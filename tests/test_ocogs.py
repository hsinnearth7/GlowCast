"""Tests for OCOGSTracker."""

from __future__ import annotations

import pytest

from app.cost.ocogs_tracker import CostVarianceResult, OCOGSTracker


@pytest.fixture
def ocogs_tracker(small_tables):
    return OCOGSTracker(
        cost_transactions=small_tables["fact_cost_transactions"],
        products=small_tables["dim_product"],
    )


class TestOCOGSTracker:

    def test_compute_variance(self, ocogs_tracker):
        result = ocogs_tracker.compute_variance()
        assert isinstance(result, CostVarianceResult)
        assert result.total_actual > 0

    def test_compute_variance_with_dates(self, ocogs_tracker):
        result = ocogs_tracker.compute_variance(
            period_start="2022-01-01", period_end="2022-03-31"
        )
        assert isinstance(result, CostVarianceResult)

    def test_trend_analysis(self, ocogs_tracker):
        trend = ocogs_tracker.trend_analysis(lookback_months=6)
        assert "avg_unit_cost" in trend.columns

    def test_flag_outliers(self, ocogs_tracker):
        outliers = ocogs_tracker.flag_outliers(z_threshold=2.0)
        assert "z_score" in outliers.columns

    def test_variance_favorable_flag(self, ocogs_tracker):
        """favorable should be True when actual <= budget."""
        result = ocogs_tracker.compute_variance()
        if result.total_actual <= result.total_budget:
            assert result.favorable is True
        else:
            assert result.favorable is False

    def test_compute_variance_empty_range(self, ocogs_tracker):
        """Querying a date range with no data should return zeros."""
        result = ocogs_tracker.compute_variance(
            period_start="1900-01-01", period_end="1900-01-02"
        )
        assert result.total_actual == 0.0
        assert result.total_budget == 0.0

    def test_trend_analysis_columns(self, ocogs_tracker):
        """Trend analysis should produce expected columns."""
        trend = ocogs_tracker.trend_analysis(lookback_months=12)
        for col in ("period", "avg_unit_cost", "total_cost", "volume", "cost_change_pct"):
            assert col in trend.columns

    def test_flag_outliers_high_threshold_returns_fewer(self, ocogs_tracker):
        """Higher z-threshold should return equal or fewer outliers."""
        outliers_low = ocogs_tracker.flag_outliers(z_threshold=1.5)
        outliers_high = ocogs_tracker.flag_outliers(z_threshold=3.0)
        assert len(outliers_high) <= len(outliers_low)
