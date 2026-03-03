"""Tests for SHAP and LIME explainability."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from app.explain.shap_lime import LIMEExplainer, SHAPExplainer, compare_explanations


@pytest.fixture
def trained_model():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.normal(0, 0.5, 200)
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X


@pytest.fixture
def feature_names():
    return ["social_momentum", "temperature", "humidity", "lag_1", "rolling_mean_7"]


class TestSHAPExplainer:
    def test_compute_shap_values(self, trained_model, feature_names):
        model, X = trained_model
        explainer = SHAPExplainer(model, feature_names)
        shap_vals = explainer.compute_shap_values(X[:20])
        assert shap_vals.shape == (20, 5)

    def test_feature_importance(self, trained_model, feature_names):
        model, X = trained_model
        explainer = SHAPExplainer(model, feature_names)
        explainer.compute_shap_values(X[:20])
        importance = explainer.feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 5
        assert importance.iloc[0]["mean_abs_shap"] >= importance.iloc[-1]["mean_abs_shap"]

    def test_top_feature_is_feature_0(self, trained_model, feature_names):
        """Feature 0 has coefficient 2.0, should be most important."""
        model, X = trained_model
        explainer = SHAPExplainer(model, feature_names)
        explainer.compute_shap_values(X[:50])
        importance = explainer.feature_importance()
        assert importance.iloc[0]["feature"] == "social_momentum"


class TestLIMEExplainer:
    def test_explain_instance(self, trained_model, feature_names):
        model, X = trained_model
        explainer = LIMEExplainer(model, feature_names)
        explanation = explainer.explain_instance(X[0])
        assert isinstance(explanation, dict)
        assert len(explanation) > 0


class TestCompareExplanations:
    def test_compare(self, trained_model, feature_names):
        model, X = trained_model
        shap_exp = SHAPExplainer(model, feature_names)
        lime_exp = LIMEExplainer(model, feature_names)
        comparison = compare_explanations(shap_exp, lime_exp, X[:10])
        assert isinstance(comparison, pd.DataFrame)
        assert "feature" in comparison.columns
