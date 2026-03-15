-- Cost Anomaly Detection: Z-score based anomaly identification
-- GlowCast Cost & Commercial Analytics (SQLite-compatible)

WITH daily_costs AS (
    SELECT
        transaction_date AS date,
        sku_id,
        AVG(total_unit_cost) AS avg_daily_cost,
        SUM(volume) AS daily_volume
    FROM fact_cost_transactions
    GROUP BY 1, 2
),
rolling_stats AS (
    SELECT
        date,
        sku_id,
        avg_daily_cost,
        daily_volume,
        AVG(avg_daily_cost) OVER (
            PARTITION BY sku_id
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS rolling_mean
    FROM daily_costs
)
SELECT
    date,
    sku_id,
    avg_daily_cost,
    rolling_mean,
    CASE WHEN rolling_mean = 0 THEN 0
         ELSE ABS(avg_daily_cost - rolling_mean) / MAX(rolling_mean * 0.1, 0.01)
    END AS deviation_ratio,
    CASE
        WHEN rolling_mean > 0 AND ABS(avg_daily_cost - rolling_mean) / MAX(rolling_mean * 0.1, 0.01) > 3.0 THEN 'CRITICAL'
        WHEN rolling_mean > 0 AND ABS(avg_daily_cost - rolling_mean) / MAX(rolling_mean * 0.1, 0.01) > 2.5 THEN 'WARNING'
        ELSE 'NORMAL'
    END AS anomaly_status
FROM rolling_stats
WHERE rolling_mean > 0
  AND ABS(avg_daily_cost - rolling_mean) / MAX(rolling_mean * 0.1, 0.01) > 2.5
ORDER BY date DESC, deviation_ratio DESC;
