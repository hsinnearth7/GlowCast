"""Tests for CUPED variance reduction."""

import numpy as np
import pytest

from app.experimentation.cuped import CUPEDAnalyzer


class TestCUPEDAnalyzer:
    @pytest.fixture
    def correlated_data(self):
        rng = np.random.default_rng(42)
        n = 5000
        pre = rng.normal(100, 20, n)
        post = 0.74 * pre + np.sqrt(1 - 0.74**2) * rng.normal(100, 20, n)
        return pre, post

    def test_fit(self, correlated_data):
        pre, post = correlated_data
        analyzer = CUPEDAnalyzer()
        analyzer.fit(pre, post)
        assert hasattr(analyzer, "theta_")
        assert hasattr(analyzer, "rho_")

    def test_rho_positive(self, correlated_data):
        pre, post = correlated_data
        analyzer = CUPEDAnalyzer()
        analyzer.fit(pre, post)
        assert analyzer.rho_ > 0.5

    def test_transform_reduces_variance(self, correlated_data):
        pre, post = correlated_data
        analyzer = CUPEDAnalyzer()
        analyzer.fit(pre, post)
        adjusted = analyzer.transform(post, pre)
        assert np.var(adjusted) < np.var(post)

    def test_variance_reduction_significant(self, correlated_data):
        pre, post = correlated_data
        analyzer = CUPEDAnalyzer()
        analyzer.fit(pre, post)
        vr = analyzer.compute_variance_reduction()
        assert vr["variance_reduction_ratio"] > 0.30  # should be ~55%

    def test_bootstrap_ci(self, correlated_data):
        pre, post = correlated_data
        analyzer = CUPEDAnalyzer()
        analyzer.fit(pre, post)
        ci = analyzer.bootstrap_ci(n=200)  # fewer for speed
        assert ci.lower < ci.upper

    def test_unfitted_transform_raises(self):
        analyzer = CUPEDAnalyzer()
        with pytest.raises((ValueError, AttributeError, RuntimeError)):
            analyzer.transform(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_zero_variance_covariate(self):
        analyzer = CUPEDAnalyzer()
        pre = np.ones(100)
        post = np.random.default_rng(42).normal(10, 2, 100)
        import warnings
        # Source issues a RuntimeWarning and sets theta=0 instead of raising
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            analyzer.fit(pre, post)
        # theta_ should be 0 (no adjustment applied) and rho_ should be 0
        assert analyzer.theta_ == 0.0
        assert analyzer.rho_ == 0.0
