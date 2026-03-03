"""Tests for fairness analysis."""

import numpy as np
import pandas as pd
import pytest

from app.explain.fairness import FairnessAnalyzer


class TestFairnessAnalyzer:
    @pytest.fixture
    def fair_data(self):
        rng = np.random.default_rng(42)
        n = 600
        y_true = rng.poisson(10, n).astype(float)
        y_pred = y_true + rng.normal(0, 1, n)
        groups = np.repeat(["FC_Phoenix", "FC_Berlin", "FC_Tokyo", "FC_Mumbai", "FC_London", "FC_Dallas"], 100)
        return y_true, y_pred, groups

    def test_per_group_mape(self, fair_data):
        y_true, y_pred, groups = fair_data
        analyzer = FairnessAnalyzer(y_true, y_pred, groups)
        result = analyzer.per_group_mape()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6
        assert "mape" in result.columns

    def test_kruskal_wallis(self, fair_data):
        y_true, y_pred, groups = fair_data
        analyzer = FairnessAnalyzer(y_true, y_pred, groups)
        result = analyzer.kruskal_wallis_test()
        assert "H" in result
        assert "p_value" in result
        # Fair data should not show significant group differences
        assert result["p_value"] > 0.01

    def test_chi_squared(self, fair_data):
        y_true, y_pred, groups = fair_data
        categories = np.repeat(["AntiAging", "Acne", "Hydrating", "Brightening", "SunProtection", "Hydrating"], 100)
        analyzer = FairnessAnalyzer(y_true, y_pred, groups)
        result = analyzer.chi_squared_test(categories)
        assert "chi2" in result
        assert "p_value" in result

    def test_slice_fairness(self, fair_data):
        y_true, y_pred, groups = fair_data
        analyzer = FairnessAnalyzer(y_true, y_pred, groups)
        segments = {
            "High_Volume": np.array([True] * 300 + [False] * 300),
            "Low_Volume": np.array([False] * 300 + [True] * 300),
        }
        result = analyzer.slice_fairness(segments)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_unfair_data_detected(self):
        rng = np.random.default_rng(42)
        y_true = rng.poisson(10, 200).astype(float)
        y_pred_good = y_true + rng.normal(0, 0.5, 200)
        y_pred_bad = y_true + rng.normal(5, 3, 200)
        y_pred = np.concatenate([y_pred_good[:100], y_pred_bad[100:]])
        groups = np.array(["good_fc"] * 100 + ["bad_fc"] * 100)
        analyzer = FairnessAnalyzer(y_true, y_pred, groups)
        result = analyzer.kruskal_wallis_test()
        assert result["p_value"] < 0.05  # should detect unfairness
