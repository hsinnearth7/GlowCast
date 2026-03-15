"""Tests for uplift modeling (metalearners)."""

import numpy as np
import pandas as pd
import pytest

from app.causal.uplift import UpliftAnalyzer


class TestUpliftAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return UpliftAnalyzer()

    def test_fit(self, analyzer, binary_treatment_data):
        X = binary_treatment_data[["X0", "X1", "X2", "X3", "X4"]].values
        treatment = binary_treatment_data["treatment"].values
        outcome = binary_treatment_data["outcome"].values
        analyzer.fit(X, treatment, outcome)

    def test_predict_cate(self, analyzer, binary_treatment_data):
        X = binary_treatment_data[["X0", "X1", "X2", "X3", "X4"]].values
        treatment = binary_treatment_data["treatment"].values
        outcome = binary_treatment_data["outcome"].values
        analyzer.fit(X, treatment, outcome)
        cate = analyzer.predict_cate(X, learner="x_learner")
        assert len(cate) == len(X)

    def test_all_learners_predict(self, analyzer, binary_treatment_data):
        X = binary_treatment_data[["X0", "X1", "X2", "X3", "X4"]].values
        treatment = binary_treatment_data["treatment"].values
        outcome = binary_treatment_data["outcome"].values
        analyzer.fit(X, treatment, outcome)
        for learner in ["s_learner", "t_learner", "x_learner"]:
            cate = analyzer.predict_cate(X, learner=learner)
            assert len(cate) == len(X)

    def test_compute_auuc(self, analyzer, binary_treatment_data):
        X = binary_treatment_data[["X0", "X1", "X2", "X3", "X4"]].values
        treatment = binary_treatment_data["treatment"].values
        outcome = binary_treatment_data["outcome"].values
        analyzer.fit(X, treatment, outcome)
        auuc = analyzer.compute_auuc(X, treatment, outcome)
        assert 0 <= auuc <= 1

    def test_ablation_study(self, analyzer, binary_treatment_data):
        X = binary_treatment_data[["X0", "X1", "X2", "X3", "X4"]].values
        treatment = binary_treatment_data["treatment"].values
        outcome = binary_treatment_data["outcome"].values
        analyzer.fit(X, treatment, outcome)
        result = analyzer.ablation_study(X, treatment, outcome, n_bootstrap=50)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3

    def test_identify_sensitive(self, analyzer, binary_treatment_data):
        X = binary_treatment_data[["X0", "X1", "X2", "X3", "X4"]].values
        treatment = binary_treatment_data["treatment"].values
        outcome = binary_treatment_data["outcome"].values
        analyzer.fit(X, treatment, outcome)
        cate = analyzer.predict_cate(X, learner="x_learner")
        mask = analyzer.identify_sensitive(cate, threshold=0.3)
        assert len(mask) == len(X)
        assert mask.dtype == bool

    def test_treatment_ratio_20_80(self, binary_treatment_data):
        treatment = binary_treatment_data["treatment"].values
        treatment_pct = treatment.mean()
        assert 0.15 < treatment_pct < 0.25  # should be ~20%

    def test_x_learner_handles_imbalance(self, analyzer):
        """X-Learner should work with 20/80 imbalanced treatment."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal((n, 3))
        treatment = rng.choice([0, 1], size=n, p=[0.8, 0.2])
        outcome = X[:, 0] + treatment * 0.5 + rng.normal(0, 0.5, n)
        analyzer.fit(X, treatment, outcome)
        cate = analyzer.predict_cate(X, learner="x_learner")
        assert not np.any(np.isnan(cate))
