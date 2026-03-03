/*
================================================================================
  cross_zone_penalty.sql — Cross-Zone Fulfillment Penalty Analysis
================================================================================

PURPOSE
-------
Quantifies the financial and operational penalty incurred when orders are
fulfilled from a non-local fulfillment centre (Cross_Zone_Fulfillment), broken
down by fulfillment centre and skin-concern segment.

Penalty model:
  cross_zone_penalty_usd = SUM(
      CASE WHEN fulfillment_type = 'Cross_Zone_Fulfillment'
           THEN units_sold * 50          -- $50 flat opportunity cost per unit
           ELSE 0
      END
  )

The $50/unit figure represents the composite cost of:
  - Excess shipping vs. local rate (approx. $30–$55 cross-zone delta)
  - Extended delivery days → higher customer-churn probability
  - Carbon/sustainability penalty (regulatory compliance cost estimate)

Operators can recalibrate the 50 constant to their actual logistics contract.

QUERY PLAN  (EXPLAIN QUERY PLAN summary)
----------------------------------------
  CTE FulfillmentSummary
    → Full scan on fact_fulfillment.
      JOIN with fact_sales on order_id (many-to-many; fact_sales may have
      multiple rows per order_id — the join multiplies rows, so the SUM must
      GROUP BY fc + concern).
      SQLite will likely choose a nested-loop join: fact_fulfillment as outer
      (no index needed for full scan), fact_sales as inner indexed on order_id
      (if an index exists; otherwise inner full scan per outer row → O(N×M)).
      Adding:  CREATE INDEX idx_sales_order_id ON fact_sales(order_id, sku_id);
      converts the inner side to an index lookup O(log M) per outer row.

  CTE ConcernMapping
    → Narrow scan on dim_product (sku_id, concern columns only).
      Expected cardinality: ~5,000 rows.  Used as a lookup table in the join.

  Final SELECT
    → Hash join: FulfillmentSummary ⋈ ConcernMapping on sku_id.
      Aggregate SUM / COUNT grouped by fc_id + concern.
      ORDER BY forces a final sort pass.

CONCERN FLAG
------------
'High_Concern' flagged when cross_zone_penalty_usd > 10,000 USD per group,
indicating a structural mismatch between demand origin and stock placement
that warrants inventory rebalancing or replenishment policy review.

COLUMNS RETURNED
----------------
  fulfilled_from_fc       TEXT   — Source fulfillment centre
  concern                 TEXT   — Skin concern segment
  total_orders            INT    — Total fulfillment events in group
  total_units_sold        INT    — Total units shipped (all types)
  cross_zone_orders       INT    — Count of cross-zone events
  cross_zone_units        INT    — Units shipped cross-zone
  local_units             INT    — Units shipped locally
  cross_zone_pct          REAL   — Cross-zone share of total units (0–100)
  actual_shipping_cost    REAL   — Sum of recorded shipping_cost (USD)
  cross_zone_penalty_usd  REAL   — Model penalty (units × $50) for cross-zone
  avg_delivery_days       REAL   — Average delivery days across all types
  avg_cross_zone_days     REAL   — Average delivery days for cross-zone only
  concern_flag            TEXT   — 'High_Concern' | 'Acceptable'
================================================================================
*/

WITH

-- ── CTE 1: Join fulfillment events with SKU metadata ─────────────────────────
-- Each fulfillment record is enriched with the concern from Dim_Product via
-- the sku_id on the fulfillment record (no need to detour through fact_sales).
FulfillmentEnriched AS (
    SELECT
        ff.fulfillment_id,
        ff.fulfilled_from_fc,
        ff.sku_id,
        ff.units_sold,
        ff.fulfillment_type,
        ff.shipping_cost,
        ff.delivery_days,
        dp.concern
    FROM  fact_fulfillment ff
    LEFT  JOIN dim_product  dp
            ON dp.sku_id = ff.sku_id
),

-- ── CTE 2: Penalty calculation per FC × concern ───────────────────────────────
-- Aggregates raw fulfillment metrics and applies the $50/unit cross-zone penalty.
PenaltySummary AS (
    SELECT
        fulfilled_from_fc,
        concern,
        COUNT(*)                                                    AS total_orders,
        SUM(units_sold)                                             AS total_units_sold,
        -- Cross-zone metrics
        SUM(CASE WHEN fulfillment_type = 'Cross_Zone_Fulfillment'
                 THEN 1 ELSE 0 END)                                AS cross_zone_orders,
        SUM(CASE WHEN fulfillment_type = 'Cross_Zone_Fulfillment'
                 THEN units_sold ELSE 0 END)                       AS cross_zone_units,
        SUM(CASE WHEN fulfillment_type = 'Local_Fulfillment'
                 THEN units_sold ELSE 0 END)                       AS local_units,
        -- Shipping costs
        SUM(shipping_cost)                                          AS actual_shipping_cost,
        -- Cross-zone penalty: units_sold × $50 per cross-zone event
        SUM(
            CASE
                WHEN fulfillment_type = 'Cross_Zone_Fulfillment'
                THEN units_sold * 50
                ELSE 0
            END
        )                                                           AS cross_zone_penalty_usd,
        -- Delivery performance
        AVG(delivery_days)                                          AS avg_delivery_days,
        AVG(
            CASE
                WHEN fulfillment_type = 'Cross_Zone_Fulfillment'
                THEN CAST(delivery_days AS REAL)
                ELSE NULL
            END
        )                                                           AS avg_cross_zone_days
    FROM  FulfillmentEnriched
    WHERE concern IS NOT NULL           -- exclude records with no product match
    GROUP BY
        fulfilled_from_fc,
        concern
)

-- ── Final output ──────────────────────────────────────────────────────────────
SELECT
    fulfilled_from_fc,
    concern,
    total_orders,
    total_units_sold,
    cross_zone_orders,
    cross_zone_units,
    local_units,
    -- Cross-zone percentage (0–100 scale)
    ROUND(
        CASE
            WHEN total_units_sold = 0 THEN 0.0
            ELSE 100.0 * cross_zone_units / total_units_sold
        END,
        2
    )                                                               AS cross_zone_pct,
    ROUND(actual_shipping_cost, 2)                                  AS actual_shipping_cost,
    ROUND(cross_zone_penalty_usd, 2)                               AS cross_zone_penalty_usd,
    ROUND(avg_delivery_days, 2)                                     AS avg_delivery_days,
    ROUND(avg_cross_zone_days, 2)                                   AS avg_cross_zone_days,
    -- Concern flag: threshold at $10,000 cross-zone penalty per group
    CASE
        WHEN cross_zone_penalty_usd > 10000 THEN 'High_Concern'
        ELSE                                     'Acceptable'
    END                                                             AS concern_flag
FROM  PenaltySummary
ORDER BY
    cross_zone_penalty_usd DESC,
    fulfilled_from_fc,
    concern;
