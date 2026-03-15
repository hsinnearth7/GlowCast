"""Data contracts for GlowCast — re-exports all schemas plus cost analytics I/O contracts.

Usage:
    from app.data.contracts import Dim_Product, Fact_Cost_Transactions, SHOULD_COST_SCHEMA
"""

from __future__ import annotations

from pandera import Check, Column, DataFrameSchema

# Re-export star schema contracts
from app.data.star_schema import (
    Dim_Plant,
    Dim_Product,
    Dim_Supplier,
    Fact_Commodity_Prices,
    Fact_Cost_Reduction_Actions,
    Fact_Cost_Transactions,
    Fact_Purchase_Orders,
    Fact_Quality_Events,
    Fact_Supplier_Quotes,
)

__all__ = [
    "Dim_Product", "Dim_Supplier", "Dim_Plant",
    "Fact_Cost_Transactions", "Fact_Supplier_Quotes", "Fact_Cost_Reduction_Actions",
    "Fact_Commodity_Prices", "Fact_Purchase_Orders", "Fact_Quality_Events",
    "SHOULD_COST_SCHEMA", "COST_VARIANCE_SCHEMA", "MAKE_VS_BUY_SCHEMA",
]

# ── Should-Cost model output ─────────────────────────────────────────────

SHOULD_COST_SCHEMA = DataFrameSchema(
    {
        "sku_id": Column(str, nullable=False),
        "should_cost": Column(float, Check.gt(0)),
        "actual_cost": Column(float, Check.gt(0)),
        "gap_pct": Column(float),
        "gap_abs": Column(float),
        "largest_element": Column(str, nullable=False),
    },
    name="Should_Cost_Schema",
    coerce=True,
)

# ── Cost variance analysis output ────────────────────────────────────────

COST_VARIANCE_SCHEMA = DataFrameSchema(
    {
        "period": Column("datetime64[ns]"),
        "avg_unit_cost": Column(float, Check.gt(0)),
        "total_cost": Column(float, Check.ge(0)),
        "volume": Column(int, Check.ge(0)),
        "cost_change_pct": Column(float, nullable=True),
    },
    name="Cost_Variance_Schema",
    coerce=True,
)

# ── Make-vs-Buy comparison output ────────────────────────────────────────

MAKE_VS_BUY_SCHEMA = DataFrameSchema(
    {
        "cost_change_pct": Column(float),
        "make_cost": Column(float, Check.gt(0)),
        "buy_cost": Column(float, Check.gt(0)),
        "recommendation": Column(str, Check.isin(["make", "buy"])),
    },
    name="Make_vs_Buy_Schema",
    coerce=True,
)
