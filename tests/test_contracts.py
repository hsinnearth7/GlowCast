"""Tests for data contracts (Pandera schemas)."""

import pandas as pd
import pandera as pa
import pytest

from app.data.contracts import (
    EVAL_SCHEMA,
    FORECAST_SCHEMA,
    Y_SCHEMA,
)
from app.data.star_schema import (
    Dim_Location,
    Dim_Product,
    Dim_Weather,
    Fact_Inventory_Batch,
    Fact_Sales,
    Fact_Social_Signals,
)


class TestStarSchemaValidation:
    def test_dim_product_valid(self, small_tables):
        Dim_Product.validate(small_tables["dim_product"])

    def test_dim_location_valid(self, small_tables):
        Dim_Location.validate(small_tables["dim_location"])

    def test_dim_weather_valid(self, small_tables):
        Dim_Weather.validate(small_tables["dim_weather"])

    def test_fact_sales_valid(self, small_tables):
        Fact_Sales.validate(small_tables["fact_sales"])

    def test_fact_inventory_valid(self, small_tables):
        Fact_Inventory_Batch.validate(small_tables["fact_inventory"])

    def test_fact_social_valid(self, small_tables):
        Fact_Social_Signals.validate(small_tables["fact_social"])


class TestDimProductContract:
    def test_rejects_positive_elasticity(self, small_tables):
        bad = small_tables["dim_product"].copy()
        bad.loc[bad.index[0], "price_elasticity"] = 0.5
        with pytest.raises(pa.errors.SchemaError):
            Dim_Product.validate(bad)

    def test_rejects_invalid_concern(self, small_tables):
        bad = small_tables["dim_product"].copy()
        bad.loc[bad.index[0], "concern"] = "InvalidConcern"
        with pytest.raises(pa.errors.SchemaError):
            Dim_Product.validate(bad)

    def test_rejects_negative_price(self, small_tables):
        bad = small_tables["dim_product"].copy()
        bad.loc[bad.index[0], "retail_price"] = -10.0
        with pytest.raises(pa.errors.SchemaError):
            Dim_Product.validate(bad)


class TestYSchema:
    def test_valid_y_df(self, sample_Y_df):
        Y_SCHEMA.validate(sample_Y_df)

    def test_rejects_negative_y(self, sample_Y_df):
        bad = sample_Y_df.copy()
        bad.loc[bad.index[0], "y"] = -1.0
        with pytest.raises(pa.errors.SchemaError):
            Y_SCHEMA.validate(bad)


class TestForecastSchema:
    def test_valid_forecast(self):
        df = pd.DataFrame({
            "unique_id": ["SKU_1__FC_Phoenix"] * 3,
            "ds": pd.date_range("2023-01-01", periods=3),
            "y_hat": [10.0, 12.0, 11.0],
            "y_lo": [8.0, 10.0, 9.0],
            "y_hi": [12.0, 14.0, 13.0],
            "model": ["lightgbm"] * 3,
        })
        FORECAST_SCHEMA.validate(df)


class TestEvalSchema:
    def test_valid_eval(self):
        df = pd.DataFrame({
            "model": ["lightgbm", "lightgbm"],
            "fold": [0, 1],
            "mape": [0.12, 0.11],
            "rmse": [8.3, 7.9],
            "wmape": [0.11, 0.10],
            "coverage": [0.91, 0.92],
        })
        EVAL_SCHEMA.validate(df)
