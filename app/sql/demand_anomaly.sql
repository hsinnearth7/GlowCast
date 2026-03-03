/*
================================================================================
  demand_anomaly.sql — Rolling Z-Score Demand Anomaly Detection
================================================================================

PURPOSE
-------
Detects statistically unusual demand days using a rolling Z-score computed over
a 30-day trailing window.  Rows where |Z| > 2.0 are classified as anomalies and
surfaced with directional labels (Demand_Spike / Demand_Drop) for downstream
alerting and causal investigation.

Rolling Z-score definition (per SKU × FC, 30-day trailing window):
  mean_30d  = AVG(daily_units)  OVER (PARTITION BY sku_id, fc_id
                                      ORDER BY order_date
                                      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
  stddev_30d = estimated from variance window:
               SQRT( AVG(daily_units²) - AVG(daily_units)² )  — running Welford-style
  z_score    = (daily_units - mean_30d) / NULLIF(stddev_30d, 0)

SQLite does not have a native STDDEV window function.  We approximate population
standard deviation using the identity:
  σ = SQRT( E[X²] - E[X]² )
which is computed with two parallel AVG window frames.

QUERY PLAN  (EXPLAIN QUERY PLAN summary)
----------------------------------------
  CTE DailyDemand
    → Full scan on fact_sales (is_return=0 filter).
      GROUP BY sku_id, fc_id, order_date → hash aggregation → ephemeral table.

  CTE RollingStats
    → Reads DailyDemand (materialised CTE) sequentially.
      Two window functions share the same PARTITION BY / ORDER BY / ROWS frame
      so SQLite can compute both in a single sorted pass:
        AVG(daily_units)   → mean_30d
        AVG(daily_units²)  → mean_sq_30d
      Optimiser typically uses one temp B-Tree sort (on sku_id, fc_id,
      order_date) then streams both window aggregations simultaneously.
      Expected plan node: "USE TEMP B-TREE FOR ORDER BY" or
      "SCAN t (coroutine DailyDemand) — window sort".

  CTE ZScores
    → Single pass over RollingStats to compute stddev_30d and z_score.
      SQRT, arithmetic, and NULLIF evaluated in projection — no additional scan.

  Final SELECT
    → Filter WHERE ABS(z_score) > 2.0 and stddev_30d > 0.
      ORDER BY requires a sort pass on z_score DESC.
      Left join with dim_product (small; nested-loop on sku_id).

THRESHOLD TUNING
----------------
  Z > 2.0 corresponds to ~95.4% of a normal distribution being "expected".
  For high-frequency SKUs lower the threshold (e.g. 1.5); for intermittent
  demand SKUs raise it (e.g. 2.5–3.0) to reduce false positives.

MINIMUM WINDOW GUARD
--------------------
  window_obs >= 7 filter ensures at least 7 data points exist before flagging —
  prevents false alerts in the first few days of a new SKU's sales history.

COLUMNS RETURNED
----------------
  sku_id          TEXT   — SKU identifier
  fc_id           TEXT   — Fulfillment centre identifier
  concern         TEXT   — Skin concern segment
  order_date      TEXT   — Date of the anomalous demand observation
  daily_units     INT    — Actual units sold on this date
  mean_30d        REAL   — 30-day rolling mean (trailing)
  stddev_30d      REAL   — 30-day rolling population std dev (trailing)
  z_score         REAL   — (daily_units - mean_30d) / stddev_30d
  window_obs      INT    — Number of observations in the rolling window
  anomaly_type    TEXT   — 'Demand_Spike' (Z>2) | 'Demand_Drop' (Z<-2)
================================================================================
*/

WITH

-- ── CTE 1: Aggregate daily units per SKU × FC × date ─────────────────────────
DailyDemand AS (
    SELECT
        sku_id,
        fc_id,
        order_date,
        SUM(units_sold)                                            AS daily_units
    FROM  fact_sales
    WHERE is_return = 0
    GROUP BY
        sku_id,
        fc_id,
        order_date
),

-- ── CTE 2: Rolling 30-day statistics (mean and mean-of-squares) ───────────────
-- Both window functions share the same frame so SQLite computes them together.
-- COUNT(*) OVER same frame gives the actual window observation count,
-- which is < 30 during the warm-up period (first 29 days of a SKU's history).
RollingStats AS (
    SELECT
        sku_id,
        fc_id,
        order_date,
        daily_units,
        AVG(daily_units) OVER (
            PARTITION BY sku_id, fc_id
            ORDER BY     order_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                                          AS mean_30d,
        -- Mean of squares for variance: E[X²]
        AVG(CAST(daily_units AS REAL) * CAST(daily_units AS REAL)) OVER (
            PARTITION BY sku_id, fc_id
            ORDER BY     order_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                                          AS mean_sq_30d,
        COUNT(*) OVER (
            PARTITION BY sku_id, fc_id
            ORDER BY     order_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                                          AS window_obs
    FROM  DailyDemand
),

-- ── CTE 3: Compute population std dev and Z-score ────────────────────────────
-- Population std dev: σ = SQRT(E[X²] - (E[X])²)
-- NULLIF guards against negative radicands caused by floating-point rounding.
ZScores AS (
    SELECT
        sku_id,
        fc_id,
        order_date,
        daily_units,
        ROUND(mean_30d, 4)                                         AS mean_30d,
        ROUND(
            SQRT(
                NULLIF(
                    ABS(mean_sq_30d - (mean_30d * mean_30d)),      -- |E[X²] - E[X]²|
                    0.0
                )
            ),
            4
        )                                                          AS stddev_30d,
        window_obs,
        -- Z-score: (observed - mean) / stddev; NULL if stddev is zero
        CASE
            WHEN mean_sq_30d - (mean_30d * mean_30d) <= 0.0       THEN NULL
            ELSE ROUND(
                (daily_units - mean_30d)
                / SQRT(ABS(mean_sq_30d - (mean_30d * mean_30d))),
                3
            )
        END                                                        AS z_score
    FROM  RollingStats
)

-- ── Final anomaly output ──────────────────────────────────────────────────────
SELECT
    zs.sku_id,
    zs.fc_id,
    dp.concern,
    zs.order_date,
    zs.daily_units,
    zs.mean_30d,
    zs.stddev_30d,
    zs.z_score,
    zs.window_obs,
    CASE
        WHEN zs.z_score >  2.0 THEN 'Demand_Spike'
        WHEN zs.z_score < -2.0 THEN 'Demand_Drop'
        ELSE                        'Normal'           -- included for completeness
    END                                                            AS anomaly_type
FROM  ZScores zs
LEFT  JOIN dim_product dp
        ON dp.sku_id = zs.sku_id
WHERE
    -- Require minimum 7-day warm-up before flagging anomalies
    zs.window_obs  >= 7
    AND zs.stddev_30d > 0.0
    -- Only return detected anomalies (|Z| > 2.0)
    AND ABS(zs.z_score) > 2.0
ORDER BY
    ABS(zs.z_score) DESC,
    zs.order_date DESC;
