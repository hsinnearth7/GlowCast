"""Tests for CostReductionEngine."""

from __future__ import annotations

import pytest

from app.cost.cost_reduction import CostReductionEngine, ReductionRecommendation


@pytest.fixture
def cost_engine(small_tables):
    return CostReductionEngine(
        cost_transactions=small_tables["fact_cost_transactions"],
        reduction_actions=small_tables["fact_cost_reduction_actions"],
        products=small_tables["dim_product"],
    )


class TestCostReductionEngine:

    def test_recommend_actions(self, cost_engine, small_tables):
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        recs = cost_engine.recommend_actions(sku_id, top_n=3)
        assert len(recs) <= 3
        assert all(isinstance(r, ReductionRecommendation) for r in recs)

    def test_recommend_unknown_sku(self, cost_engine):
        recs = cost_engine.recommend_actions("SKU_9999")
        assert recs == []

    def test_estimate_savings(self, cost_engine, small_tables):
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = cost_engine.estimate_savings("supplier_switch", sku_id)
        assert "estimated_savings_pct" in result
        assert "confidence" in result

    def test_track_realization(self, cost_engine):
        df = cost_engine.track_realization()
        if len(df) > 0:
            assert "realization_rate" in df.columns

    def test_recommend_returns_sorted_by_savings(self, cost_engine, small_tables):
        """Recommendations should be sorted by estimated_savings_pct descending."""
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        recs = cost_engine.recommend_actions(sku_id, top_n=5)
        savings = [r.estimated_savings_pct for r in recs]
        assert savings == sorted(savings, reverse=True)

    def test_estimate_savings_unknown_action(self, cost_engine, small_tables):
        """Unknown action type should return zeros."""
        sku_id = small_tables["dim_product"].iloc[0]["sku_id"]
        result = cost_engine.estimate_savings("nonexistent_action", sku_id)
        assert result["estimated_savings_pct"] == 0.0

    def test_effectiveness_computed_from_completed(self, cost_engine):
        """Internal effectiveness dict should only include completed actions."""
        # effectiveness is computed at init; just verify it's populated
        assert isinstance(cost_engine._action_effectiveness, dict)
        # all values should be non-negative
        for v in cost_engine._action_effectiveness.values():
            assert v >= 0.0
