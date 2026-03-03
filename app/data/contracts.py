"""Data contracts for GlowCast — re-exports all schemas plus forecast I/O contracts.

Usage:
    from app.data.contracts import Dim_Product, Fact_Sales, Y_SCHEMA, FORECAST_SCHEMA
"""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Re-export star schema contracts
from app.data.star_schema import (
    Dim_Customer,
    Dim_Location,
    Dim_Product,
    Dim_Weather,
    Fact_Inventory_Batch,
    Fact_Order_Fulfillment,
    Fact_Review_Aspects,
    Fact_Sales,
    Fact_Social_Signals,
)

__all__ = [
    "Dim_Product", "Dim_Location", "Dim_Weather", "Dim_Customer",
    "Fact_Sales", "Fact_Inventory_Batch", "Fact_Social_Signals",
    "Fact_Order_Fulfillment", "Fact_Review_Aspects",
    "Y_SCHEMA", "S_SCHEMA", "FORECAST_SCHEMA", "EVAL_SCHEMA",
]

# ── Nixtla Y format ──────────────────────────────────────────────────────

Y_SCHEMA = DataFrameSchema(
    {
        "unique_id": Column(str, nullable=False),
        "ds": Column("datetime64[ns]"),
        "y": Column(float, Check.ge(0)),
    },
    name="Y_Schema",
    coerce=True,
)

# ── Hierarchy summing matrix ─────────────────────────────────────────────

S_SCHEMA = DataFrameSchema(
    {
        "sku_id": Column(str, nullable=False),
        "fc_id": Column(str, nullable=False),
        "country": Column(str, nullable=False),
        "national": Column(str, nullable=False),
    },
    name="S_Schema",
    coerce=True,
    index=pa.Index(str, name="unique_id"),
)

# ── Forecast output ──────────────────────────────────────────────────────

FORECAST_SCHEMA = DataFrameSchema(
    {
        "unique_id": Column(str, nullable=False),
        "ds": Column("datetime64[ns]"),
        "y_hat": Column(float),
        "y_lo": Column(float, nullable=True),
        "y_hi": Column(float, nullable=True),
        "model": Column(str, nullable=False),
    },
    name="Forecast_Schema",
    coerce=True,
)

# ── Evaluation results ───────────────────────────────────────────────────

EVAL_SCHEMA = DataFrameSchema(
    {
        "model": Column(str, nullable=False),
        "fold": Column(int, Check.ge(0)),
        "mape": Column(float, Check.ge(0)),
        "rmse": Column(float, Check.ge(0)),
        "wmape": Column(float, Check.ge(0)),
        "coverage": Column(float, Check.in_range(0, 1), nullable=True),
    },
    name="Eval_Schema",
    coerce=True,
)
