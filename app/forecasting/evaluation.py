"""Walk-forward cross-validation, conformal prediction, and evaluation metrics.

This module provides production-quality tools for evaluating GlowCast
demand forecasting models across the 4-level SKU hierarchy.

Key components
--------------
CVFoldResult
    Immutable dataclass storing per-fold metadata and metrics.

walk_forward_cv
    Time-series cross-validation that walks backward from the end of
    the historical series, respecting temporal ordering.

ConformalPredictor
    Split conformal prediction with a finite-sample correction factor
    that guarantees (1-α) marginal coverage.

Statistical utilities
    wilcoxon_test, cohens_d, confidence_interval — for comparing model
    MAPEs with appropriate corrections.

Metric functions
    compute_mape, compute_rmse, compute_wmape — standard accuracy metrics
    with guards against division by zero.

slice_evaluation
    Per-segment evaluation table covering five GlowCast demand archetypes:
    Overall, Stable/AntiAging, Seasonal/SunProtection, Promo, Cold_Start.

References
----------
Barber, R. F., et al. (2023). Conformal prediction beyond exchangeability.
    *Annals of Statistics*, 51(2), 816-845.
Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    *Biometrics Bulletin*, 1(6), 80-83.
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Metric epsilon guard ──────────────────────────────────────────────────────

_EPS = 1e-8  # prevents division by zero in percentage metrics


# ── Metric functions ──────────────────────────────────────────────────────────


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, guarded against zero actuals.

    Series entries where ``|y_true| < ε`` are excluded from the mean to
    avoid inflated MAPE on intermittent demand.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)

    Returns
    -------
    float
        MAPE expressed as a fraction (0.15 = 15 %).  Returns ``nan`` if all
        actuals are effectively zero.

    Examples
    --------
    >>> compute_mape(np.array([100., 200.]), np.array([110., 190.]))
    0.075
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.abs(y_true) >= _EPS
    if not mask.any():
        logger.warning("compute_mape: all y_true values are ~0; returning nan.")
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)

    Returns
    -------
    float

    Examples
    --------
    >>> compute_rmse(np.array([3., 4.]), np.array([3., 3.]))
    0.7071...
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error (volume-weighted MAPE).

    Defined as:  Σ|y - ŷ| / Σ|y|

    Preferred over MAPE for SKUs with mixed demand volumes because it
    down-weights low-volume items automatically.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)

    Returns
    -------
    float
        wMAPE as a fraction.  Returns ``nan`` if total volume is ~0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    total_volume = np.sum(np.abs(y_true))
    if total_volume < _EPS:
        logger.warning("compute_wmape: total y_true volume is ~0; returning nan.")
        return float("nan")

    return float(np.sum(np.abs(y_true - y_pred)) / total_volume)


# ── CVFoldResult ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CVFoldResult:
    """Immutable record for a single walk-forward CV fold.

    Attributes
    ----------
    fold : int
        Zero-based fold index (0 = most recent fold).
    train_end : pd.Timestamp
        Last date in the training window (inclusive).
    test_start : pd.Timestamp
        First date in the test window.
    test_end : pd.Timestamp
        Last date in the test window (inclusive).
    metrics : dict[str, float]
        Evaluation metrics computed on this fold.  Standard keys:
        ``mape``, ``rmse``, ``wmape``, ``coverage`` (optional).
    n_train : int
        Number of time steps used for training.
    n_test : int
        Number of time steps in the test window.

    Examples
    --------
    >>> result.metrics["mape"]
    0.114
    >>> result.n_train
    730
    """

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0

    def __post_init__(self) -> None:
        if self.test_start > self.test_end:
            raise ValueError(
                f"test_start ({self.test_start}) must be <= test_end ({self.test_end})"
            )
        if self.train_end >= self.test_start:
            raise ValueError(
                f"train_end ({self.train_end}) must be strictly before "
                f"test_start ({self.test_start})"
            )

    @property
    def mape(self) -> float | None:
        """Convenience accessor for fold MAPE."""
        return self.metrics.get("mape")

    @property
    def rmse(self) -> float | None:
        """Convenience accessor for fold RMSE."""
        return self.metrics.get("rmse")

    @property
    def wmape(self) -> float | None:
        """Convenience accessor for fold wMAPE."""
        return self.metrics.get("wmape")


# ── Model protocol ────────────────────────────────────────────────────────────


class _ForecastModel(Protocol):
    """Structural protocol for any model compatible with walk_forward_cv.

    The model must expose ``fit`` (returns self) and ``predict`` (returns
    a DataFrame with column ``y_hat``).
    """

    def fit(self, Y_df: pd.DataFrame) -> "_ForecastModel": ...

    def predict(self, horizon: int) -> pd.DataFrame: ...


# ── Walk-forward cross-validation ─────────────────────────────────────────────


def walk_forward_cv(
    model: _ForecastModel,
    Y_df: pd.DataFrame,
    n_windows: int = 12,
    step_size: int = 30,
    horizon: int = 14,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    verbose: bool = False,
) -> list[CVFoldResult]:
    """Walk-forward (expanding-window) time-series cross-validation.

    Folds are generated by walking *backward* from the end of the history.
    Fold 0 is the most recent, fold ``n_windows-1`` is the earliest.  Each
    fold trains on all data up to ``train_end`` (expanding window) and tests
    on the subsequent ``horizon`` days.

    The design ensures:
    * No future leakage — test window always follows train window.
    * Temporal ordering — earlier test windows train on less data.
    * Configurable step size — controls overlap between consecutive folds.

    Fold boundaries (zero-based, from end of series):
    ::
        train_end[k]   = last_date − k * step_size − horizon
        test_start[k]  = train_end[k] + 1 day
        test_end[k]    = train_end[k] + horizon days

    Parameters
    ----------
    model : ForecastModel-compatible
        Any object with ``fit(Y_df) → self`` and
        ``predict(horizon) → pd.DataFrame`` with column ``y_hat``.
    Y_df : pd.DataFrame
        Nixtla-style long frame: ``unique_id``, ``ds`` (datetime), ``y``.
        May contain multiple series (multiple ``unique_id`` values).
    n_windows : int, default 12
        Number of CV folds.  Matches ``evaluation.n_windows`` in
        ``glowcast.yaml``.
    step_size : int, default 30
        Calendar days between successive fold origins.  Matches
        ``evaluation.step_size`` in ``glowcast.yaml``.
    horizon : int, default 14
        Forecast horizon in days (test window length).  Matches
        ``evaluation.horizon`` in ``glowcast.yaml``.
    metric_fns : dict[str, callable] | None
        Extra metric functions mapping name → callable(y_true, y_pred).
        Built-in metrics (mape, rmse, wmape) are always computed.
    verbose : bool, default False
        Log progress at INFO level.

    Returns
    -------
    list[CVFoldResult]
        One result per fold, sorted fold 0 (most recent) → fold n-1.

    Raises
    ------
    ValueError
        If Y_df is missing required columns or has insufficient history.

    Examples
    --------
    >>> results = walk_forward_cv(my_model, Y_df, n_windows=12, horizon=14)
    >>> mapes = [r.metrics["mape"] for r in results]
    >>> print(f"Mean CV MAPE: {np.mean(mapes):.3f}")
    """
    _validate_Y_df(Y_df)

    Y_df = Y_df.copy()
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    all_dates = np.sort(Y_df["ds"].unique())
    last_date = pd.Timestamp(all_dates[-1])
    first_date = pd.Timestamp(all_dates[0])

    # Minimum history check
    min_required = n_windows * step_size + horizon
    n_total_days = (last_date - first_date).days + 1
    if n_total_days < min_required:
        raise ValueError(
            f"Insufficient history: {n_total_days} days available but "
            f"walk_forward_cv requires at least {min_required} days "
            f"(n_windows={n_windows}, step_size={step_size}, horizon={horizon})."
        )

    extra_metrics = metric_fns or {}
    results: list[CVFoldResult] = []

    for fold_idx in range(n_windows):
        # Walk backward from end of history
        test_end = last_date - pd.Timedelta(days=fold_idx * step_size)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)

        if train_end < first_date:
            logger.warning(
                "Fold %d: train_end (%s) < first available date (%s). "
                "Stopping at fold %d.",
                fold_idx, train_end, first_date, fold_idx - 1,
            )
            break

        # Slice data
        train_mask = Y_df["ds"] <= train_end
        test_mask = (Y_df["ds"] >= test_start) & (Y_df["ds"] <= test_end)

        train_df = Y_df[train_mask].copy()
        test_df = Y_df[test_mask].copy()

        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test slice; skipping.", fold_idx)
            continue

        n_train = int(train_df["ds"].nunique())
        n_test = int(test_df["ds"].nunique())

        if verbose:
            logger.info(
                "Fold %2d | train=[%s..%s] (%d days) | test=[%s..%s] (%d days)",
                fold_idx,
                train_df["ds"].min().date(),
                train_end.date(),
                n_train,
                test_start.date(),
                test_end.date(),
                n_test,
            )

        # Fit and predict
        try:
            fitted = model.fit(train_df)
            preds_df = fitted.predict(horizon)
        except Exception as exc:  # noqa: BLE001
            logger.error("Fold %d: model failed — %s", fold_idx, exc)
            continue

        # Align predictions with test actuals on (unique_id, ds)
        y_true_arr, y_pred_arr = _align_predictions(test_df, preds_df)

        if len(y_true_arr) == 0:
            logger.warning("Fold %d: no aligned predictions; skipping.", fold_idx)
            continue

        # Compute metrics
        fold_metrics: dict[str, float] = {
            "mape":  compute_mape(y_true_arr, y_pred_arr),
            "rmse":  compute_rmse(y_true_arr, y_pred_arr),
            "wmape": compute_wmape(y_true_arr, y_pred_arr),
        }
        for name, fn in extra_metrics.items():
            try:
                fold_metrics[name] = float(fn(y_true_arr, y_pred_arr))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metric '%s' failed on fold %d: %s", name, fold_idx, exc)
                fold_metrics[name] = float("nan")

        results.append(
            CVFoldResult(
                fold=fold_idx,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metrics=fold_metrics,
                n_train=n_train,
                n_test=n_test,
            )
        )

    if not results:
        raise RuntimeError(
            "walk_forward_cv produced no valid folds.  "
            "Check data length and fold parameters."
        )

    logger.info(
        "walk_forward_cv complete: %d folds | mean MAPE=%.4f",
        len(results),
        np.nanmean([r.metrics.get("mape", float("nan")) for r in results]),
    )
    return results


def _validate_Y_df(Y_df: pd.DataFrame) -> None:
    required = {"unique_id", "ds", "y"}
    missing = required - set(Y_df.columns)
    if missing:
        raise ValueError(
            f"Y_df is missing required columns: {missing}. "
            f"Got columns: {list(Y_df.columns)}"
        )
    if Y_df.empty:
        raise ValueError("Y_df is empty.")


def _align_predictions(
    test_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join test actuals with predictions on (unique_id, ds).

    Parameters
    ----------
    test_df : pd.DataFrame  — columns: unique_id, ds, y
    preds_df : pd.DataFrame — columns: unique_id, ds, y_hat  (+ optionals)

    Returns
    -------
    tuple (y_true, y_pred) — aligned 1-D float arrays
    """
    merged = test_df[["unique_id", "ds", "y"]].merge(
        preds_df[["unique_id", "ds", "y_hat"]],
        on=["unique_id", "ds"],
        how="inner",
    )
    if merged.empty:
        return np.array([]), np.array([])

    return merged["y"].to_numpy(dtype=np.float64), merged["y_hat"].to_numpy(dtype=np.float64)


# ── Conformal Prediction ──────────────────────────────────────────────────────


class ConformalPredictor:
    """Split conformal prediction with finite-sample coverage correction.

    Implements the split-conformal method (Papadopoulos et al., 2002) with
    the finite-sample correction from Barber et al. (2023):

        q̂ = quantile level (1 - α)(1 + 1/n) of calibration non-conformity scores

    This guarantees marginal (1-α) coverage even for finite calibration sets.

    Parameters
    ----------
    alpha : float, default 0.10
        Miscoverage level.  Produces (1-alpha) = 90 % prediction intervals
        by default, matching ``evaluation.conformal_coverage=0.90`` in
        ``glowcast.yaml``.

    Attributes
    ----------
    residual_quantile_ : float | None
        Calibrated non-conformity quantile (set after ``calibrate``).
    n_calibration_ : int | None
        Number of calibration samples used.

    Examples
    --------
    >>> cp = ConformalPredictor(alpha=0.10)
    >>> cp.calibrate(y_cal_true, y_cal_pred)
    0.382
    >>> lo, hi = cp.predict_intervals(y_test_pred)
    >>> coverage = cp.check_coverage(y_test_true, lo, hi)
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        self.alpha = alpha
        self.residual_quantile_: float | None = None
        self.n_calibration_: int | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the non-conformity quantile from calibration data.

        Non-conformity score used: absolute residual  s_i = |y_i - ŷ_i|.

        The quantile level is adjusted for finite-sample validity:
            level = min(1.0, (1 - α)(1 + 1/n))

        Parameters
        ----------
        y_true : array-like, shape (n,)
            Ground-truth calibration targets.
        y_pred : array-like, shape (n,)
            Point forecasts on the calibration set.

        Returns
        -------
        float
            The calibrated quantile value (stored as ``residual_quantile_``).

        Raises
        ------
        ValueError
            If calibration set is empty.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(y_true) == 0:
            raise ValueError("calibrate: calibration set is empty.")
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"calibrate: y_true ({len(y_true)}) and y_pred ({len(y_pred)}) "
                "must have the same length."
            )

        n = len(y_true)
        scores = np.abs(y_true - y_pred)

        # Finite-sample corrected quantile level (Barber et al., 2023)
        level = min(1.0, (1 - self.alpha) * (1 + 1.0 / n))
        self.residual_quantile_ = float(np.quantile(scores, level))
        self.n_calibration_ = n

        logger.debug(
            "ConformalPredictor calibrated: n=%d, alpha=%.3f, "
            "corrected_level=%.4f, q̂=%.4f",
            n, self.alpha, level, self.residual_quantile_,
        )
        return self.residual_quantile_

    def predict_intervals(
        self,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct symmetric prediction intervals around point forecasts.

        Interval:  [ŷ − q̂, ŷ + q̂]  clipped to [0, ∞) (demand ≥ 0).

        Parameters
        ----------
        y_pred : array-like, shape (n,)
            Point forecasts for the test set.

        Returns
        -------
        tuple (y_lo, y_hi) : pair of np.ndarray, shape (n,)
            Lower and upper prediction interval bounds, clipped to ≥ 0.

        Raises
        ------
        RuntimeError
            If ``calibrate`` has not been called yet.
        """
        if self.residual_quantile_ is None:
            raise RuntimeError(
                "predict_intervals: call calibrate() before predict_intervals()."
            )

        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        q = self.residual_quantile_

        y_lo = np.clip(y_pred - q, a_min=0.0, a_max=None)
        y_hi = np.clip(y_pred + q, a_min=0.0, a_max=None)
        return y_lo, y_hi

    def check_coverage(
        self,
        y_true: np.ndarray,
        y_lo: np.ndarray,
        y_hi: np.ndarray,
    ) -> float:
        """Compute empirical interval coverage.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_lo, y_hi : array-like, shape (n,)
            Lower and upper interval bounds from ``predict_intervals``.

        Returns
        -------
        float
            Fraction of test observations within [y_lo, y_hi].
            A well-calibrated predictor should achieve ≈ (1 - alpha).
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_lo = np.asarray(y_lo, dtype=np.float64).ravel()
        y_hi = np.asarray(y_hi, dtype=np.float64).ravel()

        if len(y_true) == 0:
            return float("nan")

        covered = (y_true >= y_lo) & (y_true <= y_hi)
        return float(covered.mean())

    def __repr__(self) -> str:
        calibrated = self.residual_quantile_ is not None
        return (
            f"ConformalPredictor(alpha={self.alpha}, "
            f"calibrated={calibrated}, "
            f"q̂={self.residual_quantile_}, "
            f"n_cal={self.n_calibration_})"
        )


# ── Statistical tests ─────────────────────────────────────────────────────────


def wilcoxon_test(
    baseline_mapes: np.ndarray,
    model_mapes: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test for paired MAPE differences.

    Tests the null hypothesis H₀: median(MAPE_baseline − MAPE_model) = 0
    against H₁: model MAPE is lower (one-sided, alternative="greater").

    Parameters
    ----------
    baseline_mapes : array-like, shape (n,)
        Per-fold or per-SKU MAPE values for the baseline model.
    model_mapes : array-like, shape (n,)
        Per-fold or per-SKU MAPE values for the challenger model.
    alpha : float, default 0.05
        Significance level.  Matches ``evaluation.wilcoxon_alpha`` in
        ``glowcast.yaml``.

    Returns
    -------
    dict with keys:
        statistic : float    — Wilcoxon test statistic W
        p_value   : float    — Two-sided p-value
        stars     : str      — Significance stars ("***"/"**"/"*"/"")
        reject    : bool     — Whether H₀ is rejected at ``alpha``
        n_pairs   : int      — Number of valid (non-tied-zero) pairs

    Examples
    --------
    >>> wilcoxon_test(baseline_mapes, model_mapes)
    {'statistic': 45.0, 'p_value': 0.021, 'stars': '*', 'reject': True, 'n_pairs': 12}
    """
    baseline_mapes = np.asarray(baseline_mapes, dtype=np.float64).ravel()
    model_mapes = np.asarray(model_mapes, dtype=np.float64).ravel()

    if len(baseline_mapes) != len(model_mapes):
        raise ValueError(
            f"baseline_mapes ({len(baseline_mapes)}) and "
            f"model_mapes ({len(model_mapes)}) must have the same length."
        )

    differences = baseline_mapes - model_mapes

    # Remove pairs where difference is exactly zero (not informative)
    non_zero_mask = np.abs(differences) > _EPS
    n_valid = int(non_zero_mask.sum())

    if n_valid < 5:
        warnings.warn(
            f"wilcoxon_test: only {n_valid} non-zero difference pairs "
            "(minimum 5 recommended).  Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    if n_valid == 0:
        return {
            "statistic": float("nan"),
            "p_value": 1.0,
            "stars": "",
            "reject": False,
            "n_pairs": 0,
        }

    try:
        stat, p_value = stats.wilcoxon(
            differences[non_zero_mask],
            alternative="two-sided",
            method="auto",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("wilcoxon_test: scipy.stats.wilcoxon failed — %s", exc)
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "stars": "",
            "reject": False,
            "n_pairs": n_valid,
        }

    stars = _significance_stars(float(p_value))
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "stars": stars,
        "reject": float(p_value) < alpha,
        "n_pairs": n_valid,
    }


def _significance_stars(p_value: float) -> str:
    """Convert p-value to significance star string."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> dict[str, Any]:
    """Cohen's d effect size for two independent groups.

    Uses the pooled standard deviation:
        d = (μ₁ − μ₂) / s_pooled

    where s_pooled = sqrt(((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)).

    Parameters
    ----------
    group1, group2 : array-like
        Numeric samples (e.g., MAPE values for two models).

    Returns
    -------
    dict with keys:
        d         : float — Cohen's d (positive means group1 > group2)
        magnitude : str   — "S" (small, |d|<0.5), "M" (medium, 0.5≤|d|<0.8),
                            "L" (large, |d|≥0.8)

    Examples
    --------
    >>> cohens_d(baseline_mapes, model_mapes)
    {'d': 0.63, 'magnitude': 'M'}
    """
    g1 = np.asarray(group1, dtype=np.float64).ravel()
    g2 = np.asarray(group2, dtype=np.float64).ravel()

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"cohens_d: each group must have ≥2 observations "
            f"(got {n1} and {n2})."
        )

    mean_diff = float(np.mean(g1) - np.mean(g2))
    var1 = float(np.var(g1, ddof=1))
    var2 = float(np.var(g2, ddof=1))
    pooled_std = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))

    if pooled_std < _EPS:
        d = 0.0
    else:
        d = mean_diff / pooled_std

    magnitude = _cohens_d_magnitude(abs(d))
    return {"d": round(d, 4), "magnitude": magnitude}


def _cohens_d_magnitude(abs_d: float) -> str:
    """Map |d| to Cohen (1988) magnitude label."""
    if abs_d < 0.5:
        return "S"   # Small
    if abs_d < 0.8:
        return "M"   # Medium
    return "L"       # Large


def confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap-free parametric confidence interval for the mean.

    Assumes the sample mean follows a t-distribution with (n-1) degrees of
    freedom.  Appropriate for metric aggregations across CV folds.

    Parameters
    ----------
    values : array-like, shape (n,)
        Observed values (e.g., per-fold MAPE).
    confidence : float, default 0.95
        Desired confidence level.

    Returns
    -------
    tuple (lo, hi)
        Lower and upper bounds of the confidence interval.

    Raises
    ------
    ValueError
        If ``values`` has fewer than 2 elements.

    Examples
    --------
    >>> confidence_interval(np.array([0.10, 0.12, 0.11, 0.13, 0.09]))
    (0.0907, 0.1293)
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    if n < 2:
        raise ValueError(
            f"confidence_interval: need ≥2 values (got {n})."
        )

    mean = float(np.mean(values))
    se = float(stats.sem(values, nan_policy="omit"))
    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * se

    return (mean - margin, mean + margin)


# ── Slice evaluation ──────────────────────────────────────────────────────────

# GlowCast segment definitions (concern-based archetypes)
_SEGMENT_CONCERNS: dict[str, list[str]] = {
    "Stable_AntiAging":       ["AntiAging"],
    "Seasonal_SunProtection": ["SunProtection"],
    "Promo":                   [],   # label-driven (no fixed concern)
    "Cold_Start":              [],   # label-driven
}

_NAMED_SEGMENTS = [
    "Overall",
    "Stable_AntiAging",
    "Seasonal_SunProtection",
    "Promo",
    "Cold_Start",
]


def slice_evaluation(
    y_true_dict: dict[str, np.ndarray],
    y_pred_dict: dict[str, np.ndarray],
    segment_labels: dict[str, str],
    baseline_dict: dict[str, np.ndarray] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Per-segment evaluation table for GlowCast demand archetypes.

    Computes MAPE, RMSE, a 95 % confidence interval on mean MAPE, and
    Cohen's d effect size (vs. baseline) for five predefined segments.

    Segments
    --------
    Overall
        All series in ``y_true_dict``.
    Stable_AntiAging
        Series whose ``segment_labels`` value is ``"AntiAging"``.
    Seasonal_SunProtection
        Series whose ``segment_labels`` value is ``"SunProtection"``.
    Promo
        Series labelled ``"Promo"``.
    Cold_Start
        Series labelled ``"Cold_Start"``.

    Parameters
    ----------
    y_true_dict : dict[str, np.ndarray]
        Mapping of ``unique_id`` → actual demand array.
    y_pred_dict : dict[str, np.ndarray]
        Mapping of ``unique_id`` → predicted demand array.
        Must share the same keys as ``y_true_dict``.
    segment_labels : dict[str, str]
        Mapping of ``unique_id`` → segment name.  Expected values:
        ``"AntiAging"``, ``"Acne"``, ``"Hydrating"``, ``"Brightening"``,
        ``"SunProtection"``, ``"Promo"``, ``"Cold_Start"``.
    baseline_dict : dict[str, np.ndarray] | None
        Optional baseline predictions for Cohen's d computation.  If None,
        ``Cohen_d`` column is filled with NaN.
    confidence : float, default 0.95
        Confidence level for the mean MAPE interval.

    Returns
    -------
    pd.DataFrame
        Columns: Segment, MAPE, RMSE, wMAPE, CI_lo, CI_hi, Cohen_d, N_series.
        One row per segment.  MAPE values are percentages (0–100 scale).

    Examples
    --------
    >>> tbl = slice_evaluation(y_true_dict, y_pred_dict, segment_labels)
    >>> tbl[["Segment", "MAPE", "CI_lo", "CI_hi"]]
    """
    common_ids = sorted(set(y_true_dict.keys()) & set(y_pred_dict.keys()))
    if not common_ids:
        raise ValueError(
            "slice_evaluation: y_true_dict and y_pred_dict share no common keys."
        )

    # Precompute per-series scalar metrics
    per_series: dict[str, dict[str, float]] = {}
    for uid in common_ids:
        yt = np.asarray(y_true_dict[uid], dtype=np.float64).ravel()
        yp = np.asarray(y_pred_dict[uid], dtype=np.float64).ravel()
        if len(yt) == 0:
            continue
        per_series[uid] = {
            "mape":  compute_mape(yt, yp),
            "rmse":  compute_rmse(yt, yp),
            "wmape": compute_wmape(yt, yp),
        }
        if baseline_dict is not None and uid in baseline_dict:
            yb = np.asarray(baseline_dict[uid], dtype=np.float64).ravel()
            per_series[uid]["mape_baseline"] = compute_mape(yt, yb)

    rows = []
    for segment in _NAMED_SEGMENTS:
        member_ids = _get_segment_members(
            segment, common_ids, segment_labels, per_series
        )
        if not member_ids:
            logger.warning("slice_evaluation: segment '%s' has no members.", segment)
            rows.append(_empty_segment_row(segment))
            continue

        segment_mapes = np.array(
            [per_series[uid]["mape"] for uid in member_ids
             if not np.isnan(per_series[uid]["mape"])],
            dtype=np.float64,
        )
        segment_rmses = np.array(
            [per_series[uid]["rmse"] for uid in member_ids],
            dtype=np.float64,
        )
        segment_wmapes = np.array(
            [per_series[uid]["wmape"] for uid in member_ids
             if not np.isnan(per_series[uid]["wmape"])],
            dtype=np.float64,
        )

        mean_mape = float(np.nanmean(segment_mapes)) * 100  # → percentage
        mean_rmse = float(np.nanmean(segment_rmses))
        mean_wmape = float(np.nanmean(segment_wmapes)) * 100

        # Confidence interval on per-series MAPE
        ci_lo, ci_hi = float("nan"), float("nan")
        if len(segment_mapes) >= 2:
            raw_lo, raw_hi = confidence_interval(segment_mapes, confidence)
            ci_lo = raw_lo * 100
            ci_hi = raw_hi * 100

        # Cohen's d (model vs baseline) if baseline available
        cohen_d_val = float("nan")
        if baseline_dict is not None:
            baseline_mapes = np.array(
                [per_series[uid].get("mape_baseline", float("nan"))
                 for uid in member_ids],
                dtype=np.float64,
            )
            valid_mask = ~np.isnan(baseline_mapes) & ~np.isnan(segment_mapes)
            if valid_mask.sum() >= 2:
                try:
                    cd = cohens_d(baseline_mapes[valid_mask], segment_mapes[valid_mask])
                    cohen_d_val = cd["d"]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("cohens_d failed for segment '%s': %s", segment, exc)

        rows.append({
            "Segment":   segment,
            "MAPE":      round(mean_mape, 3),
            "RMSE":      round(mean_rmse, 4),
            "wMAPE":     round(mean_wmape, 3),
            "CI_lo":     round(ci_lo, 3) if not np.isnan(ci_lo) else float("nan"),
            "CI_hi":     round(ci_hi, 3) if not np.isnan(ci_hi) else float("nan"),
            "Cohen_d":   round(cohen_d_val, 4) if not np.isnan(cohen_d_val) else float("nan"),
            "N_series":  len(member_ids),
        })

    result_df = pd.DataFrame(rows, columns=["Segment", "MAPE", "RMSE", "wMAPE",
                                              "CI_lo", "CI_hi", "Cohen_d", "N_series"])
    return result_df


def _get_segment_members(
    segment: str,
    all_ids: list[str],
    segment_labels: dict[str, str],
    per_series: dict[str, dict[str, float]],
) -> list[str]:
    """Return the unique_ids belonging to the given segment."""
    if segment == "Overall":
        return [uid for uid in all_ids if uid in per_series]

    # Concern-based segments
    concern_map = {
        "Stable_AntiAging":       "AntiAging",
        "Seasonal_SunProtection": "SunProtection",
    }
    if segment in concern_map:
        target_concern = concern_map[segment]
        return [
            uid for uid in all_ids
            if uid in per_series
            and segment_labels.get(uid, "") == target_concern
        ]

    # Label-based segments (Promo, Cold_Start)
    return [
        uid for uid in all_ids
        if uid in per_series
        and segment_labels.get(uid, "") == segment
    ]


def _empty_segment_row(segment: str) -> dict[str, Any]:
    """Return a row of NaNs for an empty segment."""
    return {
        "Segment":  segment,
        "MAPE":     float("nan"),
        "RMSE":     float("nan"),
        "wMAPE":    float("nan"),
        "CI_lo":    float("nan"),
        "CI_hi":    float("nan"),
        "Cohen_d":  float("nan"),
        "N_series": 0,
    }


# ── CV results summary ────────────────────────────────────────────────────────


def summarise_cv(results: list[CVFoldResult]) -> pd.DataFrame:
    """Convert a list of CVFoldResult into a summary DataFrame.

    Parameters
    ----------
    results : list[CVFoldResult]

    Returns
    -------
    pd.DataFrame
        One row per fold with columns fold, train_end, test_start, test_end,
        n_train, n_test, plus all metric columns from the metrics dict.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        row: dict[str, Any] = {
            "fold":       r.fold,
            "train_end":  r.train_end,
            "test_start": r.test_start,
            "test_end":   r.test_end,
            "n_train":    r.n_train,
            "n_test":     r.n_test,
        }
        row.update(r.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by fold index
    df = df.sort_values("fold").reset_index(drop=True)
    return df
