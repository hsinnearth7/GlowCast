/*
================================================================================
  scrap_risk.sql — FIFO Shelf-Life Scrap Risk Matrix
================================================================================

PURPOSE
-------
Identifies inventory batches at risk of expiring before they can be sold,
quantifies the potential financial loss using FIFO batch-level data, and
ranks batches by urgency (days_remaining ASC, potential_loss DESC).

The scrap risk model:
  potential_loss = GREATEST(0, (at_risk_qty - avg_daily_sales × 90)) × unit_cost

  Where:
    at_risk_qty     = units_on_hand for batches expiring within the risk window
    avg_daily_sales = 90-day trailing average computed from fact_sales
    90              = sell-through horizon (days) — matches the DoS window
    unit_cost       = from Dim_Product

  If avg_daily_sales × 90 > at_risk_qty, the SKU can sell through before expiry
  → potential_loss = 0.  Otherwise the delta is the expected unsold quantity.

QUERY PLAN  (EXPLAIN QUERY PLAN summary)
----------------------------------------
  CTE DailySalesAgg
    → Full scan on fact_sales (is_return=0 filter applied).
      GROUP BY sku_id, fc_id, order_date → ephemeral hash aggregation.
      Window function AVG(...) OVER (...) requires a sort pass (temp B-Tree)
      on the (sku_id, fc_id, order_date) key before the frame aggregation.

  CTE AvgVelocity
    → Derived from DailySalesAgg; correlated MAX subquery selects the most
      recent date row per partition.  SQLite may scan DailySalesAgg twice
      (once for outer, once for correlated subquery) unless it materialises
      the CTE first (SQLite 3.35+ WITH materialization hint is not available,
      but the planner typically materialises non-recursive CTEs).

  CTE BatchExpiry
    → Full scan on fact_inventory with filter:
        julianday(expiry_date) - julianday(snapshot_date) <= :risk_window
      No index on expiry_date → full scan.  For large tables, an index on
      (sku_id, fc_id, expiry_date) would allow an index range scan.
      GROUP BY collapses batches per (sku_id, fc_id, expiry_date).

  Final SELECT
    → Hash join: BatchExpiry ⋈ AvgVelocity on (sku_id, fc_id).
      Nested-loop join with dim_product on sku_id (small dimension).
      GREATEST() function applied in projection; no additional sort needed
      unless ORDER BY materialises a new temp B-Tree.

RISK WINDOW PARAMETER
---------------------
The risk_window_days constant (90) matches the DoS rolling window for
consistency.  Operators can override by editing the literal in BatchExpiry
or parameterising via a WITH clause constant CTE.

COLUMNS RETURNED
----------------
  sku_id             TEXT   — SKU identifier
  fc_id              TEXT   — Fulfillment centre identifier
  concern            TEXT   — Skin concern segment
  subcategory        TEXT   — Product subcategory
  unit_cost          REAL   — Unit cost (USD)
  shelf_life_days    INT    — Product shelf life (days, from Dim_Product)
  expiry_date        TEXT   — Batch expiry date (ISO-8601)
  snapshot_date      TEXT   — Inventory snapshot date
  days_remaining     REAL   — Days until expiry from snapshot date
  at_risk_qty        INT    — Total units in batches expiring within risk window
  avg_daily_sales    REAL   — 90-day trailing average daily sales velocity
  projected_sell_thru REAL  — avg_daily_sales × 90  (expected units sold in horizon)
  unsold_qty         REAL   — MAX(0, at_risk_qty - projected_sell_thru)
  potential_loss     REAL   — unsold_qty × unit_cost  (USD)
  risk_tier          TEXT   — 'Critical' | 'High' | 'Medium' | 'Low'
================================================================================
*/

WITH

-- ── CTE 1: Daily sales velocity (rolling 90-day trailing average) ─────────────
DailySalesAgg AS (
    SELECT
        sku_id,
        fc_id,
        order_date,
        SUM(units_sold)                                           AS daily_units,
        AVG(SUM(units_sold)) OVER (
            PARTITION BY sku_id, fc_id
            ORDER BY     order_date
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        )                                                         AS rolling_avg_90d
    FROM  fact_sales
    WHERE is_return = 0
    GROUP BY
        sku_id,
        fc_id,
        order_date
),

-- ── CTE 2: Most-recent velocity snapshot per SKU × FC ────────────────────────
AvgVelocity AS (
    SELECT
        d.sku_id,
        d.fc_id,
        d.rolling_avg_90d                                         AS avg_daily_sales
    FROM  DailySalesAgg d
    WHERE d.order_date = (
        SELECT MAX(d2.order_date)
        FROM   DailySalesAgg d2
        WHERE  d2.sku_id = d.sku_id
          AND  d2.fc_id  = d.fc_id
    )
),

-- ── CTE 3: Batch-level expiry risk within 90-day horizon ─────────────────────
-- Aggregates all batches sharing the same expiry date into a single risk row.
-- days_remaining is computed from snapshot_date (not 'now') for reproducibility.
BatchExpiry AS (
    SELECT
        fi.sku_id,
        fi.fc_id,
        fi.snapshot_date,
        fi.expiry_date,
        julianday(fi.expiry_date) - julianday(fi.snapshot_date)  AS days_remaining,
        SUM(fi.units_on_hand)                                     AS at_risk_qty
    FROM  fact_inventory fi
    WHERE
        -- Only batches expiring within the 90-day sell-through horizon
        julianday(fi.expiry_date) - julianday(fi.snapshot_date) <= 90
        -- Exclude already-expired batches (handled separately in dos_woc)
        AND julianday(fi.expiry_date) - julianday(fi.snapshot_date) > 0
        -- Use most recent snapshot per SKU × FC
        AND fi.snapshot_date = (
            SELECT MAX(fi2.snapshot_date)
            FROM   fact_inventory fi2
            WHERE  fi2.sku_id = fi.sku_id
              AND  fi2.fc_id  = fi.fc_id
        )
    GROUP BY
        fi.sku_id,
        fi.fc_id,
        fi.snapshot_date,
        fi.expiry_date
)

-- ── Final scrap risk matrix ───────────────────────────────────────────────────
SELECT
    be.sku_id,
    be.fc_id,
    dp.concern,
    dp.subcategory,
    ROUND(dp.unit_cost, 4)                                        AS unit_cost,
    dp.shelf_life_days,
    be.expiry_date,
    be.snapshot_date,
    ROUND(be.days_remaining, 1)                                   AS days_remaining,
    be.at_risk_qty,
    ROUND(COALESCE(av.avg_daily_sales, 0.0), 4)                  AS avg_daily_sales,
    -- Projected sell-through over the remaining days_remaining window
    ROUND(COALESCE(av.avg_daily_sales, 0.0) * be.days_remaining, 1)
                                                                  AS projected_sell_thru,
    -- Unsold quantity = MAX(0, at_risk_qty - projected_sell_thru)
    -- GREATEST() not available in all SQLite builds; use CASE for portability
    ROUND(
        CASE
            WHEN COALESCE(av.avg_daily_sales, 0.0) * be.days_remaining >= be.at_risk_qty
            THEN 0.0
            ELSE be.at_risk_qty - COALESCE(av.avg_daily_sales, 0.0) * be.days_remaining
        END,
        1
    )                                                             AS unsold_qty,
    -- potential_loss = GREATEST(0, (at_risk_qty - avg_daily_sales * days_remaining)) * unit_cost
    ROUND(
        CASE
            WHEN COALESCE(av.avg_daily_sales, 0.0) * be.days_remaining >= be.at_risk_qty
            THEN 0.0
            ELSE (be.at_risk_qty - COALESCE(av.avg_daily_sales, 0.0) * be.days_remaining)
                 * dp.unit_cost
        END,
        2
    )                                                             AS potential_loss,
    -- Risk tier based on days_remaining urgency and loss magnitude
    CASE
        WHEN be.days_remaining <= 7                                 THEN 'Critical'
        WHEN be.days_remaining <= 21                                THEN 'High'
        WHEN be.days_remaining <= 45                                THEN 'Medium'
        ELSE                                                             'Low'
    END                                                           AS risk_tier
FROM  BatchExpiry       be
LEFT  JOIN AvgVelocity  av
        ON av.sku_id = be.sku_id
       AND av.fc_id  = be.fc_id
LEFT  JOIN dim_product  dp
        ON dp.sku_id = be.sku_id
ORDER BY
    potential_loss DESC,
    days_remaining ASC;
