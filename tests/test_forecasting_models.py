"""Tests for forecasting models."""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.models import (
    ForecastModel,
    ForecastModelFactory,
    LightGBMForecaster,
    NaiveMovingAverage,
    RoutingEnsemble,
    XGBoostForecaster,
)


@pytest.fixture
def simple_Y_df():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rng = np.random.default_rng(42)
    y = 10 + 3 * np.sin(np.arange(120) / 7 * 2 * np.pi) + rng.normal(0, 1, 120)
    return pd.DataFrame({"unique_id": "SKU_1000__FC_Phoenix", "ds": dates, "y": np.maximum(0, y)})


class TestForecastModelABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ForecastModel()


class TestNaiveMovingAverage:
    def test_fit_predict(self, simple_Y_df):
        model = NaiveMovingAverage(window=30)
        model.fit(simple_Y_df)
        preds = model.predict(h=14)
        assert len(preds) == 14
        assert "y_hat" in preds.columns
        assert "ds" in preds.columns

    def test_predictions_non_negative(self, simple_Y_df):
        model = NaiveMovingAverage(window=30)
        model.fit(simple_Y_df)
        preds = model.predict(h=7)
        assert (preds["y_hat"] >= 0).all()

    def test_name_property(self):
        model = NaiveMovingAverage()
        assert model.name == "naive_ma30" or "naive" in model.name.lower()


class TestXGBoostForecaster:
    def test_fit_predict(self, simple_Y_df):
        model = XGBoostForecaster(n_estimators=10, max_depth=3)
        model.fit(simple_Y_df)
        preds = model.predict(h=14)
        assert len(preds) == 14
        assert "y_hat" in preds.columns

    def test_with_exogenous(self, simple_Y_df):
        X = simple_Y_df[["unique_id", "ds"]].copy()
        X["temperature"] = 25.0 + np.random.default_rng(42).normal(0, 5, len(X))
        X["social_momentum"] = np.random.default_rng(42).normal(0, 100, len(X))
        model = XGBoostForecaster(n_estimators=10, max_depth=3)
        model.fit(simple_Y_df, X_train=X)
        preds = model.predict(h=7)
        assert len(preds) == 7


class TestLightGBMForecaster:
    def test_fit_predict(self, simple_Y_df):
        model = LightGBMForecaster(n_estimators=10, max_depth=3, num_leaves=8)
        model.fit(simple_Y_df)
        preds = model.predict(h=14)
        assert len(preds) == 14
        assert "y_hat" in preds.columns


class TestRoutingEnsemble:
    def test_cold_start_routing(self):
        """Cold start (<60 days) should route to Chronos (or fallback)."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        rng = np.random.default_rng(42)
        Y = pd.DataFrame({
            "unique_id": "SKU_NEW__FC_Phoenix",
            "ds": dates,
            "y": rng.poisson(5, 30).astype(float),
        })
        model = RoutingEnsemble()
        model.fit(Y)
        preds = model.predict(h=7)
        assert len(preds) == 7

    def test_mature_routing(self, simple_Y_df):
        model = RoutingEnsemble()
        model.fit(simple_Y_df)
        preds = model.predict(h=14)
        assert len(preds) == 14


class TestForecastModelFactory:
    @pytest.fixture
    def factory(self):
        return ForecastModelFactory()

    def test_available_models(self, factory):
        models = factory.available_models()
        assert "naive_ma30" in models
        assert "lightgbm" in models
        assert "routing_ensemble" in models
        assert len(models) == 6

    def test_create_naive(self, factory):
        model = factory.create("naive_ma30")
        assert isinstance(model, ForecastModel)

    def test_create_unknown_raises(self, factory):
        with pytest.raises((ValueError, KeyError)):
            factory.create("nonexistent_model")

    def test_create_all(self, factory):
        models = factory.create_all()
        assert len(models) == 6
        for name, model in models.items():
            assert isinstance(model, ForecastModel)
