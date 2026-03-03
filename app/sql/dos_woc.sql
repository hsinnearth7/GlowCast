/*
================================================================================
  dos_woc.sql — Dynamic Days of Supply (DoS) / Weeks of Coverage (WoC)
================================================================================

PURPOSE
-------
Computes per-SKU per-FC inventory coverage using a rolling 90-day trailing
average of daily sales velocity, joined to the most recent inventory snapshot.
Outputs an action flag to drive replenishment decisions.

QUERY PLAN  (EXPLAIN QUERY PLAN summary)
----------------------------------------
SQLite optimizer strategy:

  CTE RollingDailySales
    → Full scan on fact_sales (no date index; SQLite in-memory table).
      GROUP BY sku_id, fc_id, order_date collapses order-level rows to daily
      aggregates.  Window function AVG(...) OVER (PARTITION BY sku_id, fc_id
      ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) runs a frame-based pass over
      the sorted partition — SQLite uses an ephemeral B-Tree for the ORDER BY
      inside the window frame, then streams the frame aggregation in O(N) per
      partition.  No index available; expect a temp B-Tree sort node in the plan.

  CTE LatestInventory
    → Full scan on fact_inventory filtered to max(snapshot_date) per sku_id /
      fc_id.  The MAX() subquery is a correlated scalar subquery; SQLite may
      materialise fact_inventory once and perform a hash groupby.  If the table
      is large, adding an index on (sku_id, fc_id, snapshot_date DESC) would
      convert this to an index-only scan.

  CTE ProductMeta
    → Narrow scan on dim_product projecting only sku_id, concern, unit_cost.
      Expected: full scan of the dimension table (typically small, O(5k) rows).

  Final SELECT
    → Hash join: LatestInventory ⋈ RollingDailySales on (sku_id, fc_id).
      Then nested-loop join with ProductMeta on sku_id (dimension is small).
      CASE expression for action_flag evaluated in the projection phase.
      ORDER BY dos ASC forces a final sort pass — SQLite materialises the join
      result into a temp B-Tree then streams in sorted order.

PERFORMANCE NOTES
-----------------
- For production scale (500k+ fact_sales rows) consider adding:
    CREATE INDEX IF NOT EXISTS idx_sales_sku_fc_date
        ON fact_sales(sku_id, fc_id, order_date);
  This converts the window-function sort from O(N log N) to O(N) index scan.
- The 90-day window (ROWS BETWEEN 89 PRECEDING AND CURRENT ROW) is intentionally
  trailing-only to avoid look-ahead bias in real-time operational use.
- dos = NULLIF(avg_daily_sales, 0) guard prevents division-by-zero for zero-
  velocity SKUs; they surface with dos=NULL and action_flag='No_Velocity'.

COLUMNS RETURNED
----------------
  sku_id          TEXT     — SKU identifier
  fc_id           TEXT     — Fulfillment centre identifier
  concern         TEXT     — Skin concern segment
  snapshot_date   TEXT     — Most recent inventory snapshot date (ISO-8601)
  units_on_hand   INT      — Total units on hand at snapshot
  avg_daily_sales REAL     — 90-day trailing average daily sales velocity
  dos             REAL     — Days of Supply  = units_on_hand / avg_daily_sales
  woc             REAL     — Weeks of Coverage = dos / 7
  unit_cost       REAL     — Unit cost from Dim_Product (USD)
  inventory_value REAL     — units_on_hand × unit_cost
  action_flag     TEXT     — Replenishment urgency signal:
                              'No_Velocity' | 'Expired' | 'High_Risk'
                              | 'Monitor'   | 'Healthy'
================================================================================
*/

WITH

-- ── CTE 1: Daily sales aggregated per SKU × FC × date ────────────────────────
-- Collapses multiple order rows per day into a single daily units_sold total,
-- then computes a 90-day trailing window average (avg_daily_sales).
-- ROWS BETWEEN 89 PRECEDING AND CURRENT ROW = exactly 90 days (current + 89 prior).
RollingDailySales AS (
    SELECT
        sku_id,
        fc_id,
        order_date,
        SUM(units_sold)                                          AS daily_units,
        AVG(SUM(units_sold)) OVER (
            PARTITION BY sku_id, fc_id
            ORDER BY     order_date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        )                                                        AS avg_daily_sales_90d
    FROM  fact_sales
    WHERE is_return = 0                  -- exclude return transactions
    GROUP BY
        sku_id,
        fc_id,
        order_date
),

-- ── CTE 2: Most-recent trailing average per SKU × FC ─────────────────────────
-- Picks the last available date's rolling average to represent current velocity.
-- Using MAX(order_date) as the anchor avoids needing a rank/row_number pass.
LatestVelocity AS (
    SELECT
        rds.sku_id,
        rds.fc_id,
        rds.avg_daily_sales_90d          AS avg_daily_sales
    FROM  RollingDailySales rds
    WHERE rds.order_date = (
        SELECT MAX(r2.order_date)
        FROM   RollingDailySales r2
        WHERE  r2.sku_id = rds.sku_id
          AND  r2.fc_id  = rds.fc_id
    )
),

-- ── CTE 3: Latest inventory snapshot per SKU × FC ────────────────────────────
-- Aggregates all batch-level units into a single on-hand total for the most
-- recent snapshot date, and surfaces the earliest expiry date across batches
-- (FIFO: the soonest-to-expire batch governs risk classification).
LatestInventory AS (
    SELECT
        fi.sku_id,
        fi.fc_id,
        fi.snapshot_date,
        SUM(fi.units_on_hand)            AS units_on_hand,
        MIN(fi.expiry_date)              AS soonest_expiry
    FROM  fact_inventory fi
    WHERE fi.snapshot_date = (
        SELECT MAX(fi2.snapshot_date)
        FROM   fact_inventory fi2
        WHERE  fi2.sku_id = fi.sku_id
          AND  fi2.fc_id  = fi.fc_id
    )
    GROUP BY
        fi.sku_id,
        fi.fc_id,
        fi.snapshot_date
)

-- ── Final projection ──────────────────────────────────────────────────────────
SELECT
    li.sku_id,
    li.fc_id,
    dp.concern,
    li.snapshot_date,
    li.units_on_hand,
    ROUND(COALESCE(lv.avg_daily_sales, 0.0), 4)                  AS avg_daily_sales,
    -- Days of Supply: guard against zero-velocity division
    CASE
        WHEN COALESCE(lv.avg_daily_sales, 0.0) = 0.0 THEN NULL
        ELSE ROUND(li.units_on_hand / lv.avg_daily_sales, 1)
    END                                                           AS dos,
    -- Weeks of Coverage = DoS / 7
    CASE
        WHEN COALESCE(lv.avg_daily_sales, 0.0) = 0.0 THEN NULL
        ELSE ROUND(li.units_on_hand / lv.avg_daily_sales / 7.0, 2)
    END                                                           AS woc,
    ROUND(dp.unit_cost, 4)                                        AS unit_cost,
    ROUND(li.units_on_hand * dp.unit_cost, 2)                    AS inventory_value,
    li.soonest_expiry,
    -- ── Action flag ──────────────────────────────────────────────────────
    -- Priority order: velocity check → expiry check → DoS thresholds
    CASE
        WHEN COALESCE(lv.avg_daily_sales, 0.0) = 0.0
             THEN 'No_Velocity'
        WHEN julianday(li.soonest_expiry) - julianday('now') <= 0
             THEN 'Expired'
        WHEN (li.units_on_hand / lv.avg_daily_sales)
                > (julianday(li.soonest_expiry) - julianday('now'))
             THEN 'High_Risk'     -- DoS exceeds shelf-life remaining → scrap risk
        WHEN (li.units_on_hand / lv.avg_daily_sales) <= 14
             THEN 'High_Risk'     -- Less than 2 weeks of stock
        WHEN (li.units_on_hand / lv.avg_daily_sales) <= 30
             THEN 'Monitor'       -- 2–4 weeks: watch closely
        ELSE 'Healthy'
    END                                                           AS action_flag
FROM  LatestInventory   li
LEFT  JOIN LatestVelocity lv
        ON lv.sku_id = li.sku_id
       AND lv.fc_id  = li.fc_id
LEFT  JOIN dim_product  dp
        ON dp.sku_id = li.sku_id
ORDER BY
    dos ASC NULLS LAST,
    li.units_on_hand DESC;
