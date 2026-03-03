"""Tests for drift monitoring."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.mlops.drift_monitor import DriftMonitor


class TestDriftMonitor:
    @pytest.fixture
    def monitor(self):
        return DriftMonitor()

    def test_no_data_drift(self, monitor):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
        cur = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
        results = monitor.check_data_drift(ref, cur)
        assert isinstance(results, list)
        for r in results:
            assert not r.is_drifted

    def test_detects_data_drift(self, monitor):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"f1": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"f1": rng.normal(5, 1, 500)})  # shifted mean
        results = monitor.check_data_drift(ref, cur)
        assert any(r.is_drifted for r in results)

    def test_no_prediction_drift(self, monitor):
        rng = np.random.default_rng(42)
        ref_preds = rng.normal(10, 2, 500)
        cur_preds = rng.normal(10, 2, 500)
        result = monitor.check_prediction_drift(ref_preds, cur_preds)
        assert not result.is_drifted

    def test_detects_prediction_drift(self, monitor):
        rng = np.random.default_rng(42)
        ref_preds = rng.normal(10, 2, 500)
        cur_preds = rng.normal(20, 5, 500)
        result = monitor.check_prediction_drift(ref_preds, cur_preds)
        assert result.is_drifted

    def test_no_concept_drift(self, monitor):
        base = datetime(2023, 1, 1)
        for i in range(10):
            monitor.record_mape(base + timedelta(days=i), 0.10)
        result = monitor.check_concept_drift()
        assert not result.is_drifted

    def test_detects_concept_drift(self, monitor):
        base = datetime(2023, 1, 1)
        for i in range(10):
            monitor.record_mape(base + timedelta(days=i), 0.25)  # above 20% threshold
        result = monitor.check_concept_drift()
        assert result.is_drifted

    def test_psi_computation(self, monitor):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        psi = monitor._compute_psi(ref, cur)
        assert psi >= 0
        assert psi < 0.1  # same distribution → low PSI
