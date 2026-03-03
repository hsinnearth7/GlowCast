"""Tests for power analysis."""

import pytest

from app.experimentation.power import PowerAnalyzer


class TestPowerAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return PowerAnalyzer()

    def test_required_sample_size(self, analyzer):
        result = analyzer.required_sample_size(
            baseline_mean=100, baseline_std=20, mde=0.05
        )
        assert hasattr(result, "n_per_group_raw") or isinstance(result, (dict, int))

    def test_larger_mde_needs_fewer_samples(self, analyzer):
        r1 = analyzer.required_sample_size(100, 20, 0.03)
        r2 = analyzer.required_sample_size(100, 20, 0.10)
        n1 = r1.n_per_group_raw if hasattr(r1, "n_per_group_raw") else r1["n_per_group_raw"]
        n2 = r2.n_per_group_raw if hasattr(r2, "n_per_group_raw") else r2["n_per_group_raw"]
        assert n1 > n2

    def test_mde_table(self, analyzer):
        table = analyzer.mde_table(100, 20, [0.03, 0.05, 0.10])
        assert len(table) == 3

    def test_cuped_adjusted_n(self, analyzer):
        n_cuped, _ = analyzer.cuped_adjusted_n(15200, rho=0.74)
        assert n_cuped < 15200
        # With rho=0.74, reduction should be ~55%
        reduction = 1 - n_cuped / 15200
        assert 0.40 < reduction < 0.65

    def test_higher_power_needs_more_samples(self, analyzer):
        r1 = analyzer.required_sample_size(100, 20, 0.05, power=0.80)
        r2 = analyzer.required_sample_size(100, 20, 0.05, power=0.95)
        n1 = r1.n_per_group_raw if hasattr(r1, "n_per_group_raw") else r1["n_per_group_raw"]
        n2 = r2.n_per_group_raw if hasattr(r2, "n_per_group_raw") else r2["n_per_group_raw"]
        assert n2 > n1
