"""Tests for PriceElasticityAnalyzer."""

from __future__ import annotations

import pytest

from app.cost.price_elasticity import ElasticityResult, PriceElasticityAnalyzer


@pytest.fixture
def elasticity_analyzer(small_tables):
    return PriceElasticityAnalyzer(
        transactions=small_tables["fact_cost_transactions"],
        purchase_orders=small_tables["fact_purchase_orders"],
    )


class TestPriceElasticityAnalyzer:

    def test_estimate_elasticity(self, elasticity_analyzer, small_tables):
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = elasticity_analyzer.estimate_elasticity(sku_id)
        assert isinstance(result, ElasticityResult)
        assert result.elasticity != 0

    def test_estimate_batch(self, elasticity_analyzer, small_tables):
        sku_ids = small_tables["dim_product"]["sku_id"].head(3).tolist()
        df = elasticity_analyzer.estimate_batch(sku_ids)
        assert len(df) == 3
        assert "elasticity" in df.columns

    def test_sensitivity_curve(self, elasticity_analyzer, small_tables):
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        curve = elasticity_analyzer.sensitivity_curve(sku_id)
        assert "price_multiplier" in curve.columns
        assert len(curve) == 20

    def test_estimate_unknown_sku_returns_default(self, elasticity_analyzer):
        """SKU with no data should return default elasticity of -1.0."""
        result = elasticity_analyzer.estimate_elasticity("SKU_NONEXISTENT")
        assert result.elasticity == -1.0
        assert result.p_value == 1.0

    def test_elasticity_confidence_interval(self, elasticity_analyzer, small_tables):
        """Confidence interval low bound should be <= elasticity <= high bound."""
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = elasticity_analyzer.estimate_elasticity(sku_id)
        ci_low, ci_high = result.confidence_interval
        assert ci_low <= result.elasticity <= ci_high

    def test_sensitivity_curve_custom_points(self, elasticity_analyzer, small_tables):
        """Custom n_points should be reflected in output length."""
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        curve = elasticity_analyzer.sensitivity_curve(sku_id, n_points=10)
        assert len(curve) == 10
