"""Tests for sequential testing (mSPRT)."""

import numpy as np

from app.experimentation.sequential import SequentialTester


class TestSequentialTester:
    def test_init(self):
        tester = SequentialTester()
        assert tester is not None

    def test_update_control(self):
        tester = SequentialTester()
        tester.update(10.0, "control")
        tester.update(12.0, "control")
        pv = tester.get_pvalue()
        assert 0 <= pv <= 1

    def test_update_both_groups(self):
        rng = np.random.default_rng(42)
        tester = SequentialTester()
        for _ in range(100):
            tester.update(float(rng.normal(10, 2)), "control")
            tester.update(float(rng.normal(10, 2)), "treatment")
        pv = tester.get_pvalue()
        assert 0 <= pv <= 1

    def test_significant_effect_detected(self):
        rng = np.random.default_rng(42)
        tester = SequentialTester()
        for _ in range(500):
            tester.update(float(rng.normal(10, 2)), "control")
            tester.update(float(rng.normal(15, 2)), "treatment")  # large effect
        assert tester.should_stop(alpha=0.05)

    def test_no_effect_not_stopped_early(self):
        rng = np.random.default_rng(42)
        tester = SequentialTester()
        for _ in range(50):
            tester.update(float(rng.normal(10, 2)), "control")
            tester.update(float(rng.normal(10, 2)), "treatment")
        assert not tester.should_stop(alpha=0.05)

    def test_confidence_sequence(self):
        rng = np.random.default_rng(42)
        tester = SequentialTester()
        for _ in range(100):
            tester.update(float(rng.normal(10, 2)), "control")
            tester.update(float(rng.normal(11, 2)), "treatment")
        cs = tester.get_confidence_sequence()
        assert isinstance(cs, (tuple, dict, list))
