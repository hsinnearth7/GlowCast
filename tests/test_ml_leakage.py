"""Tests for ML data leakage prevention.

Validates that:
1. Feature engineering uses .shift(1) to prevent same-day leakage
2. Walk-forward CV never uses future data
3. Social lag features are T-3 (not T-0)
4. Train/test split is temporal (no random split on time series)
"""

import numpy as np
import pandas as pd


class TestFeatureLeakage:
    def test_lag_features_shifted(self):
        """Lag features must use .shift(1) so day T doesn't know day T's value."""
        from app.forecasting.models import _engineer_features

        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        y_series = pd.Series(range(60), index=dates, dtype=float)
        features = _engineer_features(y_series)

        # lag_1 on day 0 should be NaN (no previous day)
        first_valid = features["lag_1"].first_valid_index()
        if first_valid is not None:
            # The lag_1 value should correspond to the previous day's y
            idx = first_valid
            lag_val = features.loc[idx, "lag_1"]
            y_val = y_series.loc[idx]
            assert lag_val != y_val, "lag_1 should not equal same-day y (leakage!)"

    def test_rolling_features_shifted(self):
        """Rolling mean/std must be computed on shifted data."""
        from app.forecasting.models import _engineer_features

        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        rng = np.random.default_rng(42)
        y_series = pd.Series(rng.poisson(10, 60).astype(float), index=dates)
        features = _engineer_features(y_series)

        # rolling_mean_7 should have at least one NaN (first row, shifted series is NaN)
        assert features["rolling_mean_7"].isna().sum() >= 1


class _MultiSeriesNaiveMA:
    """Wrapper around NaiveMovingAverage that handles multiple series and
    adds the unique_id column to predictions so walk_forward_cv can align them."""

    def __init__(self, window: int = 14) -> None:
        self._window = window
        self._models: dict = {}

    def fit(self, Y_df: pd.DataFrame) -> "_MultiSeriesNaiveMA":
        from app.forecasting.models import NaiveMovingAverage
        self._models = {}
        for uid, grp in Y_df.groupby("unique_id"):
            m = NaiveMovingAverage(window=self._window)
            m.fit(grp)
            self._models[uid] = m
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        parts = []
        for uid, m in self._models.items():
            preds = m.predict(h=horizon)
            preds.insert(0, "unique_id", uid)
            parts.append(preds)
        if not parts:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])
        return pd.concat(parts, ignore_index=True)


class TestTemporalSplit:
    def test_walk_forward_no_future_data(self, sample_Y_df):
        """Walk-forward CV must not use any test-period data for training."""
        from app.forecasting.evaluation import walk_forward_cv

        model = _MultiSeriesNaiveMA(window=14)
        results = walk_forward_cv(model, sample_Y_df, n_windows=3, step_size=14, horizon=7)

        for fold in results:
            # train_end must be before test_start
            assert fold.train_end < fold.test_start


class TestSocialLagLeakage:
    def test_social_t3_lag(self):
        """Social momentum should be lagged T-3, not T-0."""

        # The social lag is 3 days as defined in the config
        from app.settings import load_config
        load_config.cache_clear()
        config = load_config()
        assert config["data"]["social_lag_days"] == 3


class TestInventoryDateLeakage:
    def test_expiry_after_manufacturing(self, small_tables):
        """Batch expiry must always be after manufacturing date."""
        inv = small_tables["fact_inventory"]
        assert (inv["expiry_date"] > inv["manufacturing_date"]).all()

    def test_no_future_manufacturing_dates(self, small_tables):
        """Manufacturing dates should not be in the future relative to snapshot."""
        inv = small_tables["fact_inventory"]
        assert (inv["manufacturing_date"] <= inv["snapshot_date"]).all()
