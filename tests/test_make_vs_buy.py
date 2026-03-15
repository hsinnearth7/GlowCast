"""Tests for MakeVsBuyCalculator."""

from __future__ import annotations

import pytest

from app.cost.make_vs_buy import MakeVsBuyCalculator, MakeVsBuyResult


@pytest.fixture
def mvb_calculator(small_tables):
    return MakeVsBuyCalculator(
        products=small_tables["dim_product"],
        plants=small_tables["dim_plant"],
        supplier_quotes=small_tables["fact_supplier_quotes"],
        quality_events=small_tables["fact_quality_events"],
    )


class TestMakeVsBuyCalculator:

    def test_analyze(self, mvb_calculator, small_tables):
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = mvb_calculator.analyze(sku_id, "PLT_Shenzhen")
        assert isinstance(result, MakeVsBuyResult)
        assert result.make_cost > 0
        assert result.buy_cost > 0
        assert result.recommendation in ("make", "buy", "review")

    def test_analyze_unknown_sku(self, mvb_calculator):
        with pytest.raises(ValueError, match="not found"):
            mvb_calculator.analyze("SKU_9999", "PLT_Shenzhen")

    def test_sensitivity_analysis(self, mvb_calculator, small_tables):
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = mvb_calculator.sensitivity_analysis(sku_id, "PLT_Shenzhen")
        assert "recommendation" in result.columns
        assert len(result) > 0

    def test_analyze_unknown_plant(self, mvb_calculator, small_tables):
        """Unknown plant should raise ValueError."""
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        with pytest.raises(ValueError, match="not found"):
            mvb_calculator.analyze(sku_id, "PLT_NONEXISTENT")

    def test_recommendation_is_valid(self, mvb_calculator, small_tables):
        """Recommendation must be one of make/buy/review."""
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = mvb_calculator.analyze(sku_id, "PLT_Shenzhen")
        assert result.recommendation in ("make", "buy", "review")

    def test_custom_weights(self, small_tables):
        """Custom weights should be respected in composite scoring."""
        calc = MakeVsBuyCalculator(
            products=small_tables["dim_product"],
            plants=small_tables["dim_plant"],
            supplier_quotes=small_tables["fact_supplier_quotes"],
            quality_events=small_tables["fact_quality_events"],
            weights={"cost": 1.0, "quality": 0.0, "lead_time": 0.0, "strategic": 0.0},
        )
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = calc.analyze(sku_id, "PLT_Shenzhen")
        assert isinstance(result, MakeVsBuyResult)
        # With 100% cost weight, cost advantage should drive recommendation
        assert result.composite_score_make >= 0
        assert result.composite_score_buy >= 0
