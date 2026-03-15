-- Should-Cost Gap Analysis: Identify SKUs with significant cost gaps
-- GlowCast Cost & Commercial Analytics (SQLite-compatible)

WITH sku_costs AS (
    SELECT
        t.sku_id,
        pr.category,
        pr.cost_tier,
        pr.target_cost,
        AVG(t.total_unit_cost) AS avg_actual_cost,
        COUNT(*) AS n_transactions,
        AVG(t.raw_material_cost) AS avg_raw_material,
        AVG(t.labor_cost) AS avg_labor,
        AVG(t.overhead_cost) AS avg_overhead
    FROM fact_cost_transactions t
    JOIN dim_product pr ON t.sku_id = pr.sku_id
    GROUP BY 1, 2, 3, 4
)
SELECT
    sku_id,
    category,
    cost_tier,
    target_cost,
    avg_actual_cost,
    CASE WHEN target_cost = 0 THEN 0
         ELSE (avg_actual_cost - target_cost) * 1.0 / target_cost
    END AS gap_pct,
    avg_actual_cost - target_cost AS gap_abs,
    n_transactions,
    CASE
        WHEN avg_raw_material >= avg_labor AND avg_raw_material >= avg_overhead THEN 'raw_material'
        WHEN avg_labor >= avg_raw_material AND avg_labor >= avg_overhead THEN 'labor'
        ELSE 'overhead'
    END AS largest_cost_element
FROM sku_costs
WHERE target_cost > 0 AND (avg_actual_cost - target_cost) * 1.0 / target_cost > 0.10
ORDER BY gap_pct DESC;
