-- Cost Variance Analysis: Actual vs Target by Plant × Category
-- GlowCast Cost & Commercial Analytics (SQLite-compatible)

WITH monthly_costs AS (
    SELECT
        strftime('%Y-%m-01', t.transaction_date) AS period,
        t.plant_id,
        pr.category,
        AVG(t.total_unit_cost) AS avg_actual_cost,
        AVG(pr.target_cost) AS avg_target_cost,
        SUM(t.volume) AS total_volume,
        SUM(t.total_unit_cost * t.volume) AS total_actual,
        SUM(pr.target_cost * t.volume) AS total_budget
    FROM fact_cost_transactions t
    JOIN dim_plant p ON t.plant_id = p.plant_id
    JOIN dim_product pr ON t.sku_id = pr.sku_id
    GROUP BY 1, 2, 3
)
SELECT
    period,
    plant_id,
    category,
    avg_actual_cost,
    avg_target_cost,
    total_volume,
    total_actual,
    total_budget,
    CASE WHEN total_budget = 0 THEN 0
         ELSE (total_actual - total_budget) * 1.0 / total_budget
    END AS variance_pct,
    CASE
        WHEN total_budget > 0 AND (total_actual - total_budget) * 1.0 / total_budget > 0.10 THEN 'OVER_BUDGET'
        WHEN total_budget > 0 AND (total_actual - total_budget) * 1.0 / total_budget < -0.05 THEN 'UNDER_BUDGET'
        ELSE 'ON_TARGET'
    END AS variance_status
FROM monthly_costs
ORDER BY period DESC, variance_pct DESC;
