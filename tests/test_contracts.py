"""Tests for data contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.contracts import (
    COST_VARIANCE_SCHEMA,
    MAKE_VS_BUY_SCHEMA,
    SHOULD_COST_SCHEMA,
    Dim_Plant,
    Dim_Product,
    Dim_Supplier,
)


class TestDimensionSchemas:

    def test_dim_product_valid(self, small_tables):
        Dim_Product.validate(small_tables["dim_product"])

    def test_dim_supplier_valid(self, small_tables):
        Dim_Supplier.validate(small_tables["dim_supplier"])

    def test_dim_plant_valid(self, small_tables):
        Dim_Plant.validate(small_tables["dim_plant"])


class TestAnalyticsSchemas:

    def test_should_cost_schema(self):
        df = pd.DataFrame({
            "sku_id": ["SKU_0001"],
            "should_cost": [42.0],
            "actual_cost": [47.0],
            "gap_pct": [0.119],
            "gap_abs": [5.0],
            "largest_element": ["raw_material"],
        })
        SHOULD_COST_SCHEMA.validate(df)

    def test_cost_variance_schema(self):
        df = pd.DataFrame({
            "period": pd.to_datetime(["2024-01-01"]),
            "avg_unit_cost": [45.0],
            "total_cost": [45000.0],
            "volume": [1000],
            "cost_change_pct": [0.05],
        })
        COST_VARIANCE_SCHEMA.validate(df)

    def test_make_vs_buy_schema(self):
        df = pd.DataFrame({
            "cost_change_pct": [0.0],
            "make_cost": [30.0],
            "buy_cost": [35.0],
            "recommendation": ["make"],
        })
        MAKE_VS_BUY_SCHEMA.validate(df)
