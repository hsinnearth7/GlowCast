"""Tests for ShouldCostModel."""

from __future__ import annotations

import pandas as pd
import pytest

from app.cost.should_cost import CostBreakdown, ShouldCostModel


@pytest.fixture
def should_cost_model(small_tables):
    return ShouldCostModel(
        commodity_prices=small_tables["fact_commodity_prices"],
        plant_data=small_tables["dim_plant"],
        supplier_data=small_tables["dim_supplier"],
    )


class TestShouldCostModel:

    def test_decompose_returns_breakdown(self, should_cost_model, small_tables):
        product = small_tables["dim_product"].iloc[0]
        result = should_cost_model.decompose(product, "PLT_Shenzhen")
        assert isinstance(result, CostBreakdown)
        assert result.total_should_cost > 0
        assert len(result.cost_elements) == 5

    def test_decompose_batch(self, should_cost_model, small_tables):
        products = small_tables["dim_product"].head(5)
        results = should_cost_model.decompose_batch(products, "PLT_Shenzhen")
        assert len(results) == 5

    def test_benchmark(self, should_cost_model, small_tables):
        products = small_tables["dim_product"].head(5)
        breakdowns = should_cost_model.decompose_batch(products, "PLT_Shenzhen")
        bench = should_cost_model.benchmark(breakdowns)
        assert "gap_pct" in bench.columns
        assert len(bench) == 5

    def test_identify_gaps(self, should_cost_model, small_tables):
        products = small_tables["dim_product"].head(10)
        breakdowns = should_cost_model.decompose_batch(products, "PLT_Shenzhen")
        gaps = should_cost_model.identify_gaps(breakdowns, threshold=0.0)
        # At least some should have gaps > 0
        assert isinstance(gaps, list)

    def test_decompose_cost_elements_sum_to_total(self, should_cost_model, small_tables):
        """Total should-cost must equal the sum of its constituent elements."""
        product = small_tables["dim_product"].iloc[0]
        bd = should_cost_model.decompose(product, "PLT_Shenzhen")
        elements_sum = sum(bd.cost_elements.values())
        assert abs(bd.total_should_cost - elements_sum) < 0.01

    def test_decompose_with_unknown_plant_uses_defaults(self, should_cost_model, small_tables):
        """Unknown plant_id should fall back to default labor/overhead rates."""
        product = small_tables["dim_product"].iloc[0]
        bd = should_cost_model.decompose(product, "PLT_NONEXISTENT")
        assert isinstance(bd, CostBreakdown)
        assert bd.total_should_cost > 0

    def test_benchmark_sorted_by_gap_descending(self, should_cost_model, small_tables):
        """Benchmark DataFrame should be sorted by gap_pct descending."""
        products = small_tables["dim_product"].head(10)
        breakdowns = should_cost_model.decompose_batch(products, "PLT_Shenzhen")
        bench = should_cost_model.benchmark(breakdowns)
        gaps = bench["gap_pct"].tolist()
        assert gaps == sorted(gaps, reverse=True)

    def test_empty_commodity_prices(self, small_tables):
        """Model should handle empty commodity prices gracefully."""
        model = ShouldCostModel(
            commodity_prices=pd.DataFrame(columns=["commodity", "price_date", "price_index"]),
            plant_data=small_tables["dim_plant"],
            supplier_data=small_tables["dim_supplier"],
        )
        product = small_tables["dim_product"].iloc[0]
        bd = model.decompose(product, "PLT_Shenzhen")
        assert bd.total_should_cost > 0
