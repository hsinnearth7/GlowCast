"""Tests for CostDataGenerator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.data_generator import CostDataGenerator


class TestCostDataGenerator:
    """Tests for the cost data generator."""

    def test_init_defaults(self):
        gen = CostDataGenerator()
        assert gen.n_skus == 500
        assert gen.n_days == 1095

    def test_init_custom(self):
        gen = CostDataGenerator(n_skus=50, n_days=90)
        assert gen.n_skus == 50
        assert gen.n_days == 90

    def test_generate_all_returns_9_tables(self, small_generator):
        tables = {
            "dim_product": small_generator.dim_product,
            "dim_supplier": small_generator.dim_supplier,
            "dim_plant": small_generator.dim_plant,
            "fact_cost_transactions": small_generator.fact_cost_transactions,
            "fact_supplier_quotes": small_generator.fact_supplier_quotes,
            "fact_cost_reduction_actions": small_generator.fact_cost_reduction_actions,
            "fact_commodity_prices": small_generator.fact_commodity_prices,
            "fact_purchase_orders": small_generator.fact_purchase_orders,
            "fact_quality_events": small_generator.fact_quality_events,
        }
        assert all(v is not None for v in tables.values())
        assert len(tables) == 9

    def test_dim_product_shape(self, small_generator):
        df = small_generator.dim_product
        assert len(df) <= 50
        assert "sku_id" in df.columns
        assert "category" in df.columns
        assert "cost_tier" in df.columns
        assert "base_unit_cost" in df.columns
        assert "target_cost" in df.columns
        assert all(df["base_unit_cost"] > 0)

    def test_dim_supplier_count(self, small_generator):
        df = small_generator.dim_supplier
        assert len(df) == 5

    def test_dim_plant_count(self, small_generator):
        df = small_generator.dim_plant
        assert len(df) == 12

    def test_fact_cost_transactions_not_empty(self, small_generator):
        df = small_generator.fact_cost_transactions
        assert len(df) > 0
        assert "total_unit_cost" in df.columns
        assert all(df["total_unit_cost"] > 0)
        assert all(df["volume"] > 0)

    def test_fact_commodity_prices_all_commodities(self, small_generator):
        df = small_generator.fact_commodity_prices
        assert len(df) > 0
        assert set(df["commodity"].unique()) == {"Steel", "Copper", "Resin", "Aluminum", "Silicon"}

    def test_fact_supplier_quotes_structure(self, small_generator):
        df = small_generator.fact_supplier_quotes
        assert len(df) > 0
        assert "quoted_price" in df.columns
        assert all(df["quoted_price"] > 0)

    def test_fact_cost_reduction_actions_statuses(self, small_generator):
        df = small_generator.fact_cost_reduction_actions
        assert len(df) > 0
        valid_statuses = {"proposed", "approved", "in_progress", "completed", "cancelled"}
        assert set(df["status"].unique()).issubset(valid_statuses)

    def test_fact_purchase_orders_structure(self, small_generator):
        df = small_generator.fact_purchase_orders
        assert len(df) > 0
        assert all(df["quantity"] > 0)
        assert all(df["total_amount"] > 0)

    def test_fact_quality_events_structure(self, small_generator):
        df = small_generator.fact_quality_events
        assert len(df) > 0
        assert all(df["defect_rate"] >= 0)
        assert all(df["batch_size"] > 0)

    def test_reproducibility(self):
        gen1 = CostDataGenerator(n_skus=20, n_days=30)
        gen1.generate_all()
        h1 = gen1.compute_data_hash()

        gen2 = CostDataGenerator(n_skus=20, n_days=30)
        gen2.generate_all()
        h2 = gen2.compute_data_hash()

        assert h1 == h2

    def test_summary(self, small_generator):
        s = small_generator.summary()
        assert isinstance(s, dict)
        assert all(v > 0 for v in s.values())

    def test_validate_all(self, small_generator):
        results = small_generator.validate_all()
        for name, ok in results.items():
            assert ok, f"Validation failed for {name}"
