"""GlowCast DriftMonitor — statistical drift detection for production ML.

Three complementary drift signals are monitored:

Data drift
    Per-feature Kolmogorov-Smirnov two-sample test comparing a reference
    window (training distribution) to a current window (recent production
    data).  Threshold p < 0.05 triggers ``"ALERT"``; p < 0.01 triggers
    ``"AUTO_RETRAIN"``.

Prediction drift
    Population Stability Index (PSI) over model output distributions.
    PSI ≥ 0.1 triggers ``"ALERT"``; PSI ≥ 0.2 triggers ``"AUTO_RETRAIN"``.

Concept drift
    Rolling MAPE history.  If MAPE exceeds ``mape_threshold`` (default
    20 %) for ``consecutive_days`` (default 7) days in a row, the monitor
    emits ``"AUTO_RETRAIN"``.

Library strategy
----------------
The module attempts to import Evidently for richer HTML reports and
dataset-level drift summaries.  If Evidently is unavailable (e.g. in a
minimal Docker image), it transparently falls back to
``scipy.stats.ks_2samp`` for per-feature KS tests.

Usage
-----
    monitor = DriftMonitor()

    results = monitor.check_data_drift(ref_df, cur_df)
    for r in results:
        if r.is_drifted:
            print(r)

    pred_result = monitor.check_prediction_drift(ref_preds, cur_preds)
    monitor.record_mape(datetime.utcnow(), mape_value=0.18)
    concept = monitor.check_concept_drift()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Optional Evidently import
# ---------------------------------------------------------------------------

try:
    from evidently.metric_preset import DataDriftPreset  # type: ignore[import]  # noqa: F401
    from evidently.metrics import ColumnDriftMetric  # type: ignore[import]  # noqa: F401
    from evidently.report import Report  # type: ignore[import]  # noqa: F401
    _EVIDENTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EVIDENTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (mirrored from configs/glowcast.yaml monitoring.drift)
# ---------------------------------------------------------------------------

_KS_THRESHOLD: float = 0.05          # p-value; below this → drifted
_KS_AUTO_RETRAIN_THRESHOLD: float = 0.01  # p-value; below this → AUTO_RETRAIN
_PSI_ALERT_THRESHOLD: float = 0.1    # PSI ≥ 0.1 → alert
_PSI_RETRAIN_THRESHOLD: float = 0.2  # PSI ≥ 0.2 → AUTO_RETRAIN
_MAPE_THRESHOLD: float = 0.20        # 20 % MAPE
_CONSECUTIVE_DAYS: int = 7


# ---------------------------------------------------------------------------
# DriftResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Outcome of a single drift check.

    Attributes
    ----------
    drift_type : str
        One of ``"data"``, ``"prediction"``, ``"concept"``.
    feature_name : str
        Name of the feature or ``"predictions"`` / ``"mape"``.
    statistic : float
        Test statistic (KS-statistic, PSI, or mean MAPE).
    p_value : float
        p-value (KS test) or ``float("nan")`` for PSI / concept checks.
    threshold : float
        The threshold that was applied.
    is_drifted : bool
        ``True`` when the check has fired.
    action : str
        Recommended action: ``"OK"``, ``"ALERT"``, or ``"AUTO_RETRAIN"``.
    metadata : dict
        Additional context (e.g. Evidently drift score, consecutive count).
    """

    drift_type: str
    feature_name: str
    statistic: float
    p_value: float
    threshold: float
    is_drifted: bool
    action: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DriftResult(type={self.drift_type!r}, feature={self.feature_name!r}, "
            f"stat={self.statistic:.4f}, p={self.p_value:.4f}, "
            f"drifted={self.is_drifted}, action={self.action!r})"
        )


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Statistical drift monitor for GlowCast production forecasting.

    Parameters
    ----------
    ks_threshold : float
        KS-test p-value threshold below which a feature is flagged as
        drifted.  Defaults to 0.05 (5 % significance level).
    psi_threshold : float
        PSI threshold above which prediction distribution is flagged.
        Defaults to 0.1.
    mape_threshold : float
        MAPE percentage (expressed as a fraction, e.g. 0.20 for 20 %)
        above which a day is counted as a concept-drift event.
    consecutive_days : int
        Number of consecutive days above ``mape_threshold`` required to
        trigger ``AUTO_RETRAIN`` for concept drift.
    """

    def __init__(
        self,
        ks_threshold: float = _KS_THRESHOLD,
        psi_threshold: float = _PSI_ALERT_THRESHOLD,
        mape_threshold: float = _MAPE_THRESHOLD,
        consecutive_days: int = _CONSECUTIVE_DAYS,
    ) -> None:
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.mape_threshold = mape_threshold
        self.consecutive_days = consecutive_days

        # MAPE history: list of (timestamp, mape_value) tuples
        self._mape_history: list[dict[str, Any]] = []

        logger.info(
            "DriftMonitor initialised — evidently=%s, ks_threshold=%.3f, "
            "psi_threshold=%.3f, mape_threshold=%.2f, consecutive_days=%d",
            _EVIDENTLY_AVAILABLE,
            self.ks_threshold,
            self.psi_threshold,
            self.mape_threshold,
            self.consecutive_days,
        )

    # ------------------------------------------------------------------
    # Public — data drift
    # ------------------------------------------------------------------

    def check_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> list[DriftResult]:
        """Run per-feature KS-test to detect data distribution shift.

        If Evidently is installed, it is used to generate a full drift
        report and the per-feature Evidently drift score is stored in
        ``DriftResult.metadata["evidently_score"]``.  The KS-test is
        always run regardless, ensuring a consistent ``statistic`` and
        ``p_value`` across environments.

        Parameters
        ----------
        reference : pd.DataFrame
            Reference distribution (typically training data window).
        current : pd.DataFrame
            Current distribution (recent production data).
        columns : list[str] or None
            Subset of numeric columns to test.  If ``None``, all numeric
            columns present in both DataFrames are tested.

        Returns
        -------
        list[DriftResult]
            One ``DriftResult`` per tested feature, sorted by p-value
            ascending (most drifted first).
        """
        numeric_cols = self._select_columns(reference, current, columns)

        # Optional Evidently report — best-effort
        evidently_scores: dict[str, float] = {}
        if _EVIDENTLY_AVAILABLE:
            evidently_scores = self._run_evidently_data_drift(reference, current, numeric_cols)

        results: list[DriftResult] = []
        for col in numeric_cols:
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                logger.warning("Skipping KS test for %r — insufficient samples.", col)
                continue

            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)

            is_drifted = p_value < self.ks_threshold
            if p_value < _KS_AUTO_RETRAIN_THRESHOLD:
                action = "AUTO_RETRAIN"
            elif is_drifted:
                action = "ALERT"
            else:
                action = "OK"

            meta: dict[str, Any] = {"n_reference": len(ref_vals), "n_current": len(cur_vals)}
            if col in evidently_scores:
                meta["evidently_score"] = evidently_scores[col]

            results.append(
                DriftResult(
                    drift_type="data",
                    feature_name=col,
                    statistic=float(ks_stat),
                    p_value=float(p_value),
                    threshold=self.ks_threshold,
                    is_drifted=is_drifted,
                    action=action,
                    metadata=meta,
                )
            )

        results.sort(key=lambda r: r.p_value)
        n_drifted = sum(r.is_drifted for r in results)
        logger.info(
            "check_data_drift: tested %d features, %d drifted.",
            len(results),
            n_drifted,
        )
        return results

    # ------------------------------------------------------------------
    # Public — prediction drift
    # ------------------------------------------------------------------

    def check_prediction_drift(
        self,
        reference_preds: np.ndarray | pd.Series,
        current_preds: np.ndarray | pd.Series,
    ) -> DriftResult:
        """Detect prediction distribution shift using Population Stability Index.

        PSI is computed over a 10-bin histogram derived from the reference
        distribution.  The same bin edges are applied to the current
        distribution so the two are directly comparable.

        PSI interpretation (industry convention):
        * PSI < 0.1   → no significant shift (``"OK"``)
        * PSI 0.1–0.2 → moderate shift (``"ALERT"``)
        * PSI ≥ 0.2   → major shift (``"AUTO_RETRAIN"``)

        Parameters
        ----------
        reference_preds : array-like
            Predictions from the reference period.
        current_preds : array-like
            Predictions from the current period.

        Returns
        -------
        DriftResult
            Single result for the prediction distribution.
        """
        ref_arr = np.asarray(reference_preds, dtype=float)
        cur_arr = np.asarray(current_preds, dtype=float)

        psi_value = self._compute_psi(ref_arr, cur_arr)

        is_drifted = psi_value >= self.psi_threshold
        if psi_value >= _PSI_RETRAIN_THRESHOLD:
            action = "AUTO_RETRAIN"
        elif is_drifted:
            action = "ALERT"
        else:
            action = "OK"

        logger.info(
            "check_prediction_drift: PSI=%.4f, threshold=%.3f, action=%s",
            psi_value,
            self.psi_threshold,
            action,
        )
        return DriftResult(
            drift_type="prediction",
            feature_name="predictions",
            statistic=psi_value,
            p_value=float("nan"),
            threshold=self.psi_threshold,
            is_drifted=is_drifted,
            action=action,
            metadata={
                "n_reference": len(ref_arr),
                "n_current": len(cur_arr),
                "psi_retrain_threshold": _PSI_RETRAIN_THRESHOLD,
            },
        )

    # ------------------------------------------------------------------
    # Public — concept drift (MAPE history)
    # ------------------------------------------------------------------

    def record_mape(self, timestamp: datetime, mape_value: float) -> None:
        """Append a MAPE observation to the rolling history.

        Parameters
        ----------
        timestamp : datetime
            UTC timestamp of the evaluation window.
        mape_value : float
            MAPE expressed as a fraction (e.g. 0.15 for 15 %).
        """
        self._mape_history.append(
            {
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                "mape": float(mape_value),
                "above_threshold": mape_value > self.mape_threshold,
            }
        )
        logger.debug(
            "record_mape: timestamp=%s, mape=%.4f, above_threshold=%s",
            timestamp,
            mape_value,
            mape_value > self.mape_threshold,
        )

    def check_concept_drift(self) -> DriftResult:
        """Test whether MAPE has exceeded the threshold for consecutive days.

        The monitor counts *trailing* consecutive days where
        ``mape > mape_threshold``.  If the count reaches
        ``consecutive_days``, ``AUTO_RETRAIN`` is triggered.

        Returns
        -------
        DriftResult
            ``drift_type="concept"``, ``feature_name="mape"``.
            ``is_drifted=True`` and ``action="AUTO_RETRAIN"`` when the
            consecutive count equals or exceeds ``consecutive_days``.
        """
        if not self._mape_history:
            return DriftResult(
                drift_type="concept",
                feature_name="mape",
                statistic=0.0,
                p_value=float("nan"),
                threshold=self.mape_threshold,
                is_drifted=False,
                action="OK",
                metadata={"consecutive_count": 0, "history_length": 0},
            )

        # Count trailing consecutive above-threshold days
        consecutive = 0
        for entry in reversed(self._mape_history):
            if entry["above_threshold"]:
                consecutive += 1
            else:
                break

        recent_mapes = [e["mape"] for e in self._mape_history[-self.consecutive_days:]]
        mean_mape = float(np.mean(recent_mapes)) if recent_mapes else 0.0

        is_drifted = consecutive >= self.consecutive_days
        action = "AUTO_RETRAIN" if is_drifted else ("ALERT" if consecutive > 0 else "OK")

        logger.info(
            "check_concept_drift: consecutive=%d / %d, mean_recent_mape=%.4f, action=%s",
            consecutive,
            self.consecutive_days,
            mean_mape,
            action,
        )
        return DriftResult(
            drift_type="concept",
            feature_name="mape",
            statistic=mean_mape,
            p_value=float("nan"),
            threshold=self.mape_threshold,
            is_drifted=is_drifted,
            action=action,
            metadata={
                "consecutive_count": consecutive,
                "required_consecutive": self.consecutive_days,
                "history_length": len(self._mape_history),
            },
        )

    # ------------------------------------------------------------------
    # Private — PSI
    # ------------------------------------------------------------------

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute the Population Stability Index between two distributions.

        Uses the reference distribution to define bin edges, then maps
        both distributions onto those edges to ensure comparability.

        PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct))

        An epsilon guard of 1e-6 prevents ``log(0)`` errors when a bin
        contains zero observations in either distribution.

        Parameters
        ----------
        reference : np.ndarray
            Reference distribution values (1-D).
        current : np.ndarray
            Current distribution values (1-D).
        n_bins : int
            Number of histogram bins.  Default 10.

        Returns
        -------
        float
            PSI value.  0 = identical distributions; > 0.2 = major shift.
        """
        eps = 1e-6

        # Compute bin edges from the reference distribution
        min_val = float(np.nanmin(reference))
        max_val = float(np.nanmax(reference))

        if min_val == max_val:
            # Degenerate distribution — no information to compare
            logger.warning("_compute_psi: reference has zero variance; returning PSI=0.")
            return 0.0

        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        bin_edges[0] -= eps   # include the minimum value
        bin_edges[-1] += eps  # include the maximum value

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / (ref_counts.sum() + eps)
        cur_pct = cur_counts / (cur_counts.sum() + eps)

        # Epsilon guard: avoid log(0)
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, eps, cur_pct)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    # ------------------------------------------------------------------
    # Private — Evidently integration
    # ------------------------------------------------------------------

    def _run_evidently_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, float]:
        """Run Evidently DataDriftPreset and extract per-column drift scores.

        Parameters
        ----------
        reference, current : pd.DataFrame
            DataFrames restricted to ``columns``.
        columns : list[str]
            Numeric column names to analyse.

        Returns
        -------
        dict[str, float]
            Mapping of column name → Evidently drift score.
            Returns an empty dict on any error.
        """
        scores: dict[str, float] = {}
        try:
            ref_sub = reference[columns]
            cur_sub = current[columns]

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_sub, current_data=cur_sub)
            report_dict = report.as_dict()

            drift_results = (
                report_dict
                .get("metrics", [{}])[0]
                .get("result", {})
                .get("drift_by_columns", {})
            )
            for col, info in drift_results.items():
                if isinstance(info, dict):
                    scores[col] = float(info.get("drift_score", float("nan")))

        except Exception as exc:  # pragma: no cover
            logger.warning("Evidently report failed (%s); KS-test results are still valid.", exc)

        return scores

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_columns(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None,
    ) -> list[str]:
        """Return the list of numeric columns to test.

        Parameters
        ----------
        reference, current : pd.DataFrame
            The two DataFrames being compared.
        columns : list[str] or None
            Explicit column list, or ``None`` to auto-detect.

        Returns
        -------
        list[str]
            Sorted list of column names present in both DataFrames and
            with a numeric dtype.
        """
        common = set(reference.columns) & set(current.columns)
        if columns is not None:
            common = common & set(columns)

        numeric_cols = [
            c for c in sorted(common)
            if pd.api.types.is_numeric_dtype(reference[c])
            and pd.api.types.is_numeric_dtype(current[c])
        ]
        return numeric_cols
