"""Tests for retrain trigger logic."""

from datetime import datetime, timedelta

import pytest

from app.mlops.drift_monitor import DriftMonitor
from app.mlops.retrain_trigger import RetrainTrigger


class TestRetrainTrigger:
    @pytest.fixture
    def trigger(self):
        monitor = DriftMonitor()
        return RetrainTrigger(monitor, mape_threshold=0.20, consecutive_days=7)

    def test_no_retrain_below_threshold(self, trigger):
        base = datetime(2023, 1, 1)
        for i in range(5):
            result = trigger.check(0.10, base + timedelta(days=i))
        assert not result["should_retrain"]

    def test_retrain_after_consecutive_high_mape(self, trigger):
        base = datetime(2023, 1, 1)
        result = None
        for i in range(10):
            result = trigger.check(0.25, base + timedelta(days=i))
        assert result["should_retrain"]

    def test_reset_clears_state(self, trigger):
        base = datetime(2023, 1, 1)
        for i in range(10):
            trigger.check(0.25, base + timedelta(days=i))
        trigger.reset()
        result = trigger.check(0.25, base + timedelta(days=11))
        assert not result["should_retrain"]  # reset counter

    def test_history_recorded(self, trigger):
        trigger.check(0.15, datetime(2023, 1, 1))
        trigger.check(0.25, datetime(2023, 1, 2))
        history = trigger.get_history()
        assert len(history) == 2

    def test_intermittent_high_mape_no_retrain(self, trigger):
        """Alternating high/low should not trigger retrain (not consecutive)."""
        base = datetime(2023, 1, 1)
        for i in range(14):
            mape = 0.25 if i % 2 == 0 else 0.10
            result = trigger.check(mape, base + timedelta(days=i))
        assert not result["should_retrain"]
