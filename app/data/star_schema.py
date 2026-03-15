"""Pandera schemas for GlowCast star schema (9 tables).

Enforces data contracts on all dimension and fact tables:
    Dim_Product, Dim_Supplier, Dim_Plant,
    Fact_Cost_Transactions, Fact_Supplier_Quotes, Fact_Cost_Reduction_Actions,
    Fact_Commodity_Prices, Fact_Purchase_Orders, Fact_Quality_Events
"""

from __future__ import annotations

from pandera import Check, Column, DataFrameSchema

from app.data.segment_genes import CATEGORIES, COMMODITIES, COST_REDUCTION_ACTIONS, COST_TIERS, SUPPLIERS

# ── Dimension schemas ────────────────────────────────────────────────────

Dim_Product = DataFrameSchema(
    {
        "sku_id": Column(str, Check.str_startswith("SKU_"), unique=True),
        "product_name": Column(str, nullable=False),
        "category": Column(str, Check.isin(CATEGORIES)),
        "cost_tier": Column(str, Check.isin(COST_TIERS)),
        "subcategory": Column(str, nullable=False),
        "primary_supplier": Column(str, Check.isin(SUPPLIERS)),
        "base_unit_cost": Column(float, Check.gt(0)),
        "target_cost": Column(float, Check.gt(0)),
        "commodity_exposure": Column(str, Check.isin(COMMODITIES)),
        "commodity_sensitivity": Column(float, Check.in_range(0, 1)),
        "labor_intensity": Column(float, Check.in_range(0, 1)),
        "overhead_allocation": Column(float, Check.in_range(0, 1)),
        "moq": Column(int, Check.gt(0)),
        "tariff_exposure": Column(float, Check.in_range(0, 0.5)),
    },
    name="Dim_Product",
    coerce=True,
)

Dim_Supplier = DataFrameSchema(
    {
        "supplier_id": Column(str, Check.str_startswith("SUP_"), unique=True),
        "supplier_name": Column(str, Check.isin(SUPPLIERS)),
        "country": Column(str, Check.isin(["CN", "TW", "DE", "US", "IN"])),
        "quality_score": Column(float, Check.in_range(0, 1)),
        "on_time_delivery_pct": Column(float, Check.in_range(0, 1)),
        "lead_time_days": Column(int, Check.gt(0)),
        "price_premium": Column(float, Check.in_range(-0.2, 0.5)),
    },
    name="Dim_Supplier",
    coerce=True,
)

Dim_Plant = DataFrameSchema(
    {
        "plant_id": Column(str, Check.str_startswith("PLT_"), unique=True),
        "country": Column(str, Check.isin(["CN", "TW", "DE", "US", "MX", "IN", "JP"])),
        "region": Column(str, nullable=False),
        "lat": Column(float, Check.in_range(-90, 90)),
        "lon": Column(float, Check.in_range(-180, 180)),
        "labor_rate_hourly": Column(float, Check.gt(0)),
        "overhead_rate": Column(float, Check.in_range(0, 1)),
        "capacity_utilization": Column(float, Check.in_range(0, 1)),
    },
    name="Dim_Plant",
    coerce=True,
)

# ── Fact schemas ─────────────────────────────────────────────────────────

Fact_Cost_Transactions = DataFrameSchema(
    {
        "transaction_id": Column(str, Check.str_startswith("TXN_")),
        "transaction_date": Column("datetime64[ns]"),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "plant_id": Column(str, Check.str_startswith("PLT_")),
        "supplier_id": Column(str, Check.str_startswith("SUP_")),
        "raw_material_cost": Column(float, Check.ge(0)),
        "labor_cost": Column(float, Check.ge(0)),
        "overhead_cost": Column(float, Check.ge(0)),
        "logistics_cost": Column(float, Check.ge(0)),
        "total_unit_cost": Column(float, Check.gt(0)),
        "volume": Column(int, Check.gt(0)),
    },
    name="Fact_Cost_Transactions",
    coerce=True,
)

Fact_Supplier_Quotes = DataFrameSchema(
    {
        "quote_id": Column(str, Check.str_startswith("QUO_")),
        "quote_date": Column("datetime64[ns]"),
        "supplier_id": Column(str, Check.str_startswith("SUP_")),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "quoted_price": Column(float, Check.gt(0)),
        "lead_time_days": Column(int, Check.gt(0)),
        "moq": Column(int, Check.gt(0)),
        "valid_until": Column("datetime64[ns]"),
    },
    name="Fact_Supplier_Quotes",
    coerce=True,
)

Fact_Cost_Reduction_Actions = DataFrameSchema(
    {
        "action_id": Column(str, Check.str_startswith("CRA_")),
        "action_date": Column("datetime64[ns]"),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "action_type": Column(str, Check.isin(COST_REDUCTION_ACTIONS)),
        "projected_savings_pct": Column(float, Check.in_range(0, 1)),
        "actual_savings_pct": Column(float, Check.in_range(-0.5, 1), nullable=True),
        "status": Column(str, Check.isin(["proposed", "approved", "in_progress", "completed", "cancelled"])),
    },
    name="Fact_Cost_Reduction_Actions",
    coerce=True,
)

Fact_Commodity_Prices = DataFrameSchema(
    {
        "price_date": Column("datetime64[ns]"),
        "commodity": Column(str, Check.isin(COMMODITIES)),
        "price_index": Column(float, Check.gt(0)),
        "pct_change": Column(float),
        "volatility_30d": Column(float, Check.ge(0)),
    },
    name="Fact_Commodity_Prices",
    coerce=True,
)

Fact_Purchase_Orders = DataFrameSchema(
    {
        "po_id": Column(str, Check.str_startswith("PO_")),
        "order_date": Column("datetime64[ns]"),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "supplier_id": Column(str, Check.str_startswith("SUP_")),
        "plant_id": Column(str, Check.str_startswith("PLT_")),
        "unit_price": Column(float, Check.gt(0)),
        "quantity": Column(int, Check.gt(0)),
        "total_amount": Column(float, Check.gt(0)),
        "delivery_status": Column(str, Check.isin(["pending", "shipped", "delivered", "late"])),
        "actual_delivery_days": Column(int, Check.ge(0), nullable=True),
    },
    name="Fact_Purchase_Orders",
    coerce=True,
)

Fact_Quality_Events = DataFrameSchema(
    {
        "event_id": Column(str, Check.str_startswith("QE_")),
        "event_date": Column("datetime64[ns]"),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "supplier_id": Column(str, Check.str_startswith("SUP_")),
        "plant_id": Column(str, Check.str_startswith("PLT_")),
        "defect_rate": Column(float, Check.in_range(0, 1)),
        "batch_size": Column(int, Check.gt(0)),
        "defects_found": Column(int, Check.ge(0)),
        "disposition": Column(str, Check.isin(["accepted", "rework", "scrap", "return_to_supplier"])),
    },
    name="Fact_Quality_Events",
    coerce=True,
)
