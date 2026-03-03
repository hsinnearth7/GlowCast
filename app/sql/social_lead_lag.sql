/*
================================================================================
  social_lead_lag.sql — Social Signal Cross-Correlation Lead/Lag Analysis
================================================================================

PURPOSE
-------
Measures the strength of the linear relationship between lagged social media
mention volume and units sold, for lags k = 1 to 14 days, broken down by skin
concern.  A positive, significant correlation at lag k means that mention_volume
on day (T−k) predicts units_sold on day T with lead time k.

This surfaces the empirical lead-lag structure that the demand generator encodes
(social lead at T−3), allowing model diagnostics and future hyperparameter tuning.

Cross-correlation at lag k (per concern):
  r(k) = Pearson(units_sold_T,  mention_volume_{T-k})
       = [Σ(x_i − x̄)(y_i − ȳ)] / [√Σ(x_i − x̄)² × √Σ(y_i − ȳ)²]

Where:
  x_i  = daily mention_volume (summed across sources) on date T−k
  y_i  = daily units_sold (summed across SKUs in the concern) on date T
  x̄,ȳ = mean over the aligned sample

The query computes this using SQL window means and cross-products; no external
statistics library is required.

QUERY PLAN  (EXPLAIN QUERY PLAN summary)
----------------------------------------
  CTE DailySalesByConcern
    → Full scan on fact_sales (is_return=0).
      JOIN with dim_product on sku_id to get concern.
      GROUP BY concern, order_date → hash aggregation.
      SQLite: nested-loop join with dim_product as inner table; if dim_product
      has an index on sku_id (SQLite adds a rowid PK automatically but not a
      column index), the join is O(N log M); otherwise O(N×M) nested scan.

  CTE DailySocialByConcern
    → Full scan on fact_social (concern is a low-cardinality column).
      GROUP BY concern, signal_date → hash aggregation.

  CTE LagPairs
    → Cross join of lag values (1..14) with a values CTE.
      For each lag k, joins DailySalesByConcern with DailySocialByConcern on
      date alignment: signal_date = order_date - k days.
      SQLite performs this as 14 separate hash joins (one per lag) via the
      VALUES clause driving the outer loop, then filters date equality.
      Adding an index on DailySocialByConcern(signal_date, concern) would
      convert each join from a full scan to a range lookup — not possible
      on a CTE; create a temp table or materialise if performance is critical.

  CTE CorrStats
    → Reads LagPairs and computes per-concern per-lag means.
      Two passes may be needed (first for means, second for deviations) unless
      SQLite materialises LagPairs.  We use a two-CTE approach (LagPairs then
      CorrStats) to avoid correlated subqueries.

  Final SELECT
    → Arithmetic in the projection to compute Pearson r.
      ORDER BY concern, lag_days.

INTERPRETATION GUIDE
--------------------
  |r| > 0.7  Strong linear relationship at this lag
  |r| > 0.5  Moderate relationship
  |r| > 0.3  Weak but potentially informative
  |r| ≤ 0.3  Negligible signal at this lag

  'Leading'  = social mentions precede sales (positive lag, positive r)
  'Lagging'  = social mentions follow sales (could indicate review-driven signal)
  'Weak'     = |r| ≤ 0.3; lag not practically useful for forecasting

COLUMNS RETURNED
----------------
  concern              TEXT   — Skin concern segment
  lag_days             INT    — Lag k in days (1–14)
  paired_obs           INT    — Number of (T, T-k) date pairs with data on both sides
  mean_social          REAL   — Mean mention_volume (lagged)
  mean_sales           REAL   — Mean units_sold (contemporaneous)
  pearson_r            REAL   — Pearson correlation coefficient (−1 to +1)
  abs_r                REAL   — |pearson_r|
  signal_direction     TEXT   — 'Positive' | 'Negative' | 'Weak'
  relationship_type    TEXT   — 'Strong_Lead' | 'Moderate_Lead' | 'Weak' | 'Lagging'
================================================================================
*/

WITH

-- ── CTE 1: Aggregate daily units sold per concern and date ────────────────────
-- Joins fact_sales with dim_product to attach concern; sums across all SKUs
-- and FCs within the concern to get a single demand signal per concern-day.
DailySalesByConcern AS (
    SELECT
        dp.concern,
        fs.order_date                                              AS sale_date,
        SUM(fs.units_sold)                                         AS daily_units_sold
    FROM  fact_sales fs
    LEFT  JOIN dim_product dp
            ON dp.sku_id = fs.sku_id
    WHERE fs.is_return = 0
      AND dp.concern IS NOT NULL
    GROUP BY
        dp.concern,
        fs.order_date
),

-- ── CTE 2: Aggregate daily mention volume per concern and date ────────────────
-- Sums mention_volume across all social sources (Reddit, TikTok, etc.) to
-- produce a single social intensity signal per concern-day.
DailySocialByConcern AS (
    SELECT
        concern,
        signal_date,
        SUM(mention_volume)                                        AS daily_mentions
    FROM  fact_social
    GROUP BY
        concern,
        signal_date
),

-- ── CTE 3: Generate lag values 1–14 via a values literal ─────────────────────
LagValues AS (
    SELECT 1  AS lag_days UNION ALL SELECT 2  UNION ALL SELECT 3  UNION ALL
    SELECT 4  AS lag_days UNION ALL SELECT 5  UNION ALL SELECT 6  UNION ALL
    SELECT 7  AS lag_days UNION ALL SELECT 8  UNION ALL SELECT 9  UNION ALL
    SELECT 10 AS lag_days UNION ALL SELECT 11 UNION ALL SELECT 12 UNION ALL
    SELECT 13 AS lag_days UNION ALL SELECT 14
),

-- ── CTE 4: Aligned pairs (mention volume at T-k, units_sold at T) ─────────────
-- For each lag k and each concern, joins the two time series on the date offset.
-- The date arithmetic uses a fixed-width string assumption (YYYY-MM-DD) aligned
-- with the ISO strings written by the executor's load_tables().
LagPairs AS (
    SELECT
        s.concern,
        lv.lag_days,
        sc.daily_mentions                                          AS mention_vol,
        s.daily_units_sold                                         AS units_sold
    FROM  DailySalesByConcern s
    JOIN  LagValues           lv  ON 1 = 1          -- cross join to all lags
    -- Match social signal at T−k: signal_date + lag_days = sale_date
    JOIN  DailySocialByConcern sc
        ON sc.concern     = s.concern
       AND sc.signal_date = date(s.sale_date, '-' || lv.lag_days || ' days')
),

-- ── CTE 5: Per-group means for Pearson computation ───────────────────────────
GroupMeans AS (
    SELECT
        concern,
        lag_days,
        COUNT(*)                                                   AS paired_obs,
        AVG(mention_vol)                                           AS mean_social,
        AVG(units_sold)                                            AS mean_sales
    FROM  LagPairs
    GROUP BY
        concern,
        lag_days
),

-- ── CTE 6: Pearson numerator and denominator components ──────────────────────
-- Joins raw pairs back to group means to compute deviations.
-- Pearson r = Σ(dx × dy) / SQRT(Σdx² × Σdy²)
PearsonComponents AS (
    SELECT
        lp.concern,
        lp.lag_days,
        gm.paired_obs,
        gm.mean_social,
        gm.mean_sales,
        SUM((lp.mention_vol  - gm.mean_social) *
            (lp.units_sold   - gm.mean_sales))                    AS cov_sum,         -- Σ(dx·dy)
        SUM((lp.mention_vol  - gm.mean_social) *
            (lp.mention_vol  - gm.mean_social))                   AS var_social_sum,  -- Σdx²
        SUM((lp.units_sold   - gm.mean_sales) *
            (lp.units_sold   - gm.mean_sales))                    AS var_sales_sum    -- Σdy²
    FROM  LagPairs   lp
    JOIN  GroupMeans gm
        ON gm.concern  = lp.concern
       AND gm.lag_days = lp.lag_days
    GROUP BY
        lp.concern,
        lp.lag_days,
        gm.paired_obs,
        gm.mean_social,
        gm.mean_sales
)

-- ── Final cross-correlation output ────────────────────────────────────────────
SELECT
    concern,
    lag_days,
    paired_obs,
    ROUND(mean_social, 2)                                          AS mean_social,
    ROUND(mean_sales, 2)                                           AS mean_sales,
    -- Pearson r: guard denominator with NULLIF to handle zero-variance series
    ROUND(
        cov_sum / NULLIF(
            SQRT(var_social_sum * var_sales_sum),
            0.0
        ),
        4
    )                                                              AS pearson_r,
    ROUND(
        ABS(cov_sum / NULLIF(SQRT(var_social_sum * var_sales_sum), 0.0)),
        4
    )                                                              AS abs_r,
    -- Signal direction
    CASE
        WHEN ABS(cov_sum / NULLIF(SQRT(var_social_sum * var_sales_sum), 0.0)) <= 0.3
             THEN 'Weak'
        WHEN cov_sum > 0 THEN 'Positive'
        ELSE                  'Negative'
    END                                                            AS signal_direction,
    -- Relationship type classification
    CASE
        WHEN ABS(cov_sum / NULLIF(SQRT(var_social_sum * var_sales_sum), 0.0)) > 0.7
             THEN 'Strong_Lead'
        WHEN ABS(cov_sum / NULLIF(SQRT(var_social_sum * var_sales_sum), 0.0)) > 0.5
             THEN 'Moderate_Lead'
        WHEN ABS(cov_sum / NULLIF(SQRT(var_social_sum * var_sales_sum), 0.0)) > 0.3
             THEN 'Weak_Lead'
        ELSE 'Negligible'
    END                                                            AS relationship_type
FROM  PearsonComponents
WHERE paired_obs >= 30           -- require at least 30 aligned date pairs for stability
ORDER BY
    concern,
    lag_days;
