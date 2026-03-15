-- Supplier Performance Analysis: Quality, Delivery, and Cost metrics
-- GlowCast Cost & Commercial Analytics (SQLite-compatible)

WITH supplier_metrics AS (
    SELECT
        po.supplier_id,
        s.supplier_name,
        s.country,
        COUNT(*) AS total_orders,
        SUM(CASE WHEN po.delivery_status = 'delivered' THEN 1 ELSE 0 END) AS delivered_count,
        SUM(CASE WHEN po.delivery_status = 'late' THEN 1 ELSE 0 END) AS late_count,
        AVG(po.unit_price) AS avg_unit_price,
        SUM(po.total_amount) AS total_spend,
        AVG(po.actual_delivery_days) AS avg_delivery_days
    FROM fact_purchase_orders po
    JOIN dim_supplier s ON po.supplier_id = s.supplier_id
    GROUP BY 1, 2, 3
),
quality_metrics AS (
    SELECT
        supplier_id,
        AVG(defect_rate) AS avg_defect_rate,
        SUM(defects_found) AS total_defects,
        SUM(batch_size) AS total_inspected
    FROM fact_quality_events
    GROUP BY 1
)
SELECT
    sm.supplier_id,
    sm.supplier_name,
    sm.country,
    sm.total_orders,
    sm.total_spend,
    ROUND(sm.delivered_count * 1.0 / MAX(sm.total_orders, 1), 3) AS delivery_rate,
    ROUND(sm.late_count * 1.0 / MAX(sm.total_orders, 1), 3) AS late_rate,
    sm.avg_unit_price,
    sm.avg_delivery_days,
    COALESCE(qm.avg_defect_rate, 0) AS avg_defect_rate,
    COALESCE(qm.total_defects, 0) AS total_defects
FROM supplier_metrics sm
LEFT JOIN quality_metrics qm ON sm.supplier_id = qm.supplier_id
ORDER BY sm.total_spend DESC;
