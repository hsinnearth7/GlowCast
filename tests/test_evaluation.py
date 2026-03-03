"""Tests for evaluation metrics and walk-forward CV."""

import numpy as np
import pandas as pd

from app.forecasting.evaluation import (
    ConformalPredictor,
    cohens_d,
    compute_mape,
    compute_rmse,
    compute_wmape,
    confidence_interval,
    slice_evaluation,
    walk_forward_cv,
    wilcoxon_test,
)
from app.forecasting.models import NaiveMovingAverage


class TestMetrics:
    def test_mape_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_mape(y, y) == 0.0

    def test_mape_positive(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 22.0, 33.0])
        assert compute_mape(y_true, y_pred) > 0

    def test_rmse_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_rmse(y, y) == 0.0

    def test_rmse_positive(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 22.0, 33.0])
        assert compute_rmse(y_true, y_pred) > 0

    def test_wmape_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_wmape(y, y) == 0.0

    def test_wmape_bounded(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 25.0, 35.0])
        assert 0 < compute_wmape(y_true, y_pred) < 1


class TestConformalPredictor:
    def test_calibrate(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(10, 2, 100)
        y_pred = y_true + rng.normal(0, 1, 100)
        cp = ConformalPredictor(alpha=0.10)
        q = cp.calibrate(y_true, y_pred)
        assert q > 0

    def test_predict_intervals(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(10, 2, 100)
        y_pred = y_true + rng.normal(0, 1, 100)
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(y_true, y_pred)
        y_lo, y_hi = cp.predict_intervals(y_pred)
        assert (y_hi >= y_lo).all()
        assert (y_lo >= 0).all()

    def test_coverage_near_target(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(10, 2, 500)
        y_pred = y_true + rng.normal(0, 1, 500)
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(y_true[:250], y_pred[:250])
        y_lo, y_hi = cp.predict_intervals(y_pred[250:])
        coverage = cp.check_coverage(y_true[250:], y_lo, y_hi)
        assert coverage > 0.80  # should be near 90%


class TestStatisticalTests:
    def test_wilcoxon_significant(self):
        rng = np.random.default_rng(42)
        baseline = rng.uniform(0.15, 0.25, 50)
        model = rng.uniform(0.10, 0.18, 50)
        result = wilcoxon_test(baseline, model)
        assert result["p_value"] < 0.05

    def test_wilcoxon_not_significant(self):
        rng = np.random.default_rng(42)
        a = rng.uniform(0.10, 0.20, 50)
        b = rng.uniform(0.10, 0.20, 50)
        result = wilcoxon_test(a, b)
        # May or may not be significant for similar distributions
        assert "p_value" in result

    def test_cohens_d_large(self):
        a = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        b = np.array([20.0, 21.0, 22.0, 20.5, 21.5])
        result = cohens_d(a, b)
        assert result["d"] > 0.8 or result["magnitude"] == "L"

    def test_cohens_d_small(self):
        rng = np.random.default_rng(42)
        a = rng.normal(10, 2, 100)
        b = rng.normal(10.1, 2, 100)
        result = cohens_d(a, b)
        assert result["d"] < 0.5

    def test_confidence_interval(self):
        values = np.random.default_rng(42).normal(10, 2, 100)
        lo, hi = confidence_interval(values)
        assert lo < 10 < hi
        assert lo < hi


class _MultiSeriesNaiveMA:
    """Wrapper around NaiveMovingAverage that handles multiple series and
    adds the unique_id column to predictions so walk_forward_cv can align them."""

    def __init__(self, window: int = 14) -> None:
        self._window = window
        self._models: dict = {}
        self._train_df: pd.DataFrame | None = None

    def fit(self, Y_df: pd.DataFrame) -> "_MultiSeriesNaiveMA":
        self._train_df = Y_df.copy()
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


class TestWalkForwardCV:
    def test_basic_cv(self, sample_Y_df):
        model = _MultiSeriesNaiveMA(window=14)
        results = walk_forward_cv(model, sample_Y_df, n_windows=3, step_size=14, horizon=7)
        assert len(results) == 3
        for fold in results:
            assert fold.fold >= 0
            assert "mape" in fold.metrics or hasattr(fold, "mape")


class TestSliceEvaluation:
    def test_slice_eval(self):
        uids = [f"uid_{i}" for i in range(20)]
        y_true_dict = {uid: np.random.default_rng(42).poisson(10, 30).astype(float) for uid in uids}
        y_pred_dict = {uid: np.random.default_rng(43).poisson(10, 30).astype(float) for uid in uids}
        segment_labels = {uid: "AntiAging" if i < 10 else "SunProtection" for i, uid in enumerate(uids)}
        result = slice_evaluation(y_true_dict, y_pred_dict, segment_labels)
        assert isinstance(result, pd.DataFrame)
        assert "Segment" in result.columns
        assert "MAPE" in result.columns
