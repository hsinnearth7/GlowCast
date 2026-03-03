"""Tests for feature store (offline/online dual-mode)."""

import numpy as np
import pandas as pd
import pytest

from app.mlops.feature_store import FeatureStore


class TestFeatureStore:
    @pytest.fixture
    def Y_df(self):
        dates = pd.date_range("2023-01-01", periods=90, freq="D")
        rng = np.random.default_rng(42)
        records = []
        for uid in ["SKU_1000__FC_Phoenix", "SKU_1001__FC_Berlin"]:
            for d in dates:
                records.append({"unique_id": uid, "ds": d, "y": float(rng.poisson(10))})
        return pd.DataFrame(records)

    @pytest.fixture
    def store(self, Y_df):
        fs = FeatureStore()
        fs.materialize_offline(Y_df)
        return fs

    def test_materialize_offline(self, store):
        features = store.get_training_features()
        assert features is not None
        assert len(features) > 0

    def test_lag_features_present(self, store):
        features = store.get_training_features()
        assert "lag_1" in features.columns
        assert "lag_7" in features.columns

    def test_rolling_features_present(self, store):
        features = store.get_training_features()
        assert "rolling_mean_7" in features.columns

    def test_calendar_features(self, store):
        features = store.get_training_features()
        assert "day_of_week" in features.columns
        assert "month" in features.columns

    def test_filter_by_unique_id(self, store):
        features = store.get_training_features(unique_ids=["SKU_1000__FC_Phoenix"])
        assert all(features["unique_id"] == "SKU_1000__FC_Phoenix")

    def test_online_update_and_get(self, store):
        store.update_online("SKU_1000__FC_Phoenix", {"lag_1": 15.0, "rolling_mean_7": 12.5})
        result = store.get_online_features("SKU_1000__FC_Phoenix")
        assert result is not None
        assert result["lag_1"] == 15.0

    def test_online_get_missing(self, store):
        result = store.get_online_features("NONEXISTENT")
        assert result is None

    def test_leakage_prevention(self, store):
        """Lag features must not contain same-day y values."""
        features = store.get_training_features()
        # lag_1 should have NaN for first row of each group
        # Use nth(0) to get the actual first row per group (including NaN values)
        first_rows = features.groupby("unique_id", sort=False).nth(0)
        assert first_rows["lag_1"].isna().all()
