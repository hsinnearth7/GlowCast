"""GlowCast RetrainTrigger — automated retraining decision layer.

The trigger is the final decision node that synthesises signals from
``DriftMonitor`` into a single binary verdict: *should the pipeline
initiate a retraining job right now?*

Decision logic
--------------
1. The current MAPE value is recorded via ``DriftMonitor.record_mape``.
2. ``DriftMonitor.check_concept_drift()`` is called to count trailing
   consecutive days above ``mape_threshold``.
3. If ``consecutive_count >= consecutive_days`` → ``should_retrain=True``
   with ``reason="CONCEPT_DRIFT_MAPE"``.
4. After a successful retrain the caller must call ``reset()`` to clear
   the MAPE history and restart the counter.

The trigger intentionally focuses on *concept drift* (degrading model
accuracy) as the primary retraining signal, because concept drift has
the highest business impact for GlowCast (missed demand peaks translate
directly to stockouts and lost margin).  Data drift results from
``DriftMonitor.check_data_drift`` can be passed in via
``check(data_drift_results=...)`` for an optional secondary signal.

Usage
-----
    monitor = DriftMonitor()
    trigger = RetrainTrigger(monitor, mape_threshold=0.20, consecutive_days=7)

    # Called once per evaluation cycle (e.g. daily cron):
    verdict = trigger.check(current_mape=0.22, timestamp=datetime.utcnow())
    if verdict["should_retrain"]:
        pipeline.retrain()
        trigger.reset()

    print(trigger.get_history())
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.mlops.drift_monitor import DriftMonitor, DriftResult

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """Automated retraining decision layer for GlowCast forecasting.

    Parameters
    ----------
    drift_monitor : DriftMonitor
        Shared ``DriftMonitor`` instance that accumulates MAPE history
        and provides concept-drift verdicts.
    mape_threshold : float
        MAPE fraction (e.g. 0.20 for 20 %) above which a single day is
        counted as a concept-drift event.  Forwarded to
        ``DriftMonitor.mape_threshold`` on construction.
    consecutive_days : int
        Number of consecutive above-threshold days required to trigger
        retraining.  Forwarded to ``DriftMonitor.consecutive_days``.

    Attributes
    ----------
    _history : list[dict]
        Append-only audit trail of every ``check`` call.  Survives
        ``reset()`` calls (the history is a full record; the MAPE series
        inside ``DriftMonitor`` is what gets cleared by ``reset()``).
    """

    def __init__(
        self,
        drift_monitor: "DriftMonitor",
        mape_threshold: float = 0.20,
        consecutive_days: int = 7,
    ) -> None:
        self.drift_monitor = drift_monitor
        self.mape_threshold = mape_threshold
        self.consecutive_days = consecutive_days

        # Sync thresholds to the shared monitor
        self.drift_monitor.mape_threshold = mape_threshold
        self.drift_monitor.consecutive_days = consecutive_days

        # Full audit trail — never cleared by reset()
        self._history: list[dict[str, Any]] = []

        logger.info(
            "RetrainTrigger initialised — mape_threshold=%.2f, consecutive_days=%d",
            mape_threshold,
            consecutive_days,
        )

    # ------------------------------------------------------------------
    # Public — primary interface
    # ------------------------------------------------------------------

    def check(
        self,
        current_mape: float,
        timestamp: datetime | None = None,
        data_drift_results: "list[DriftResult] | None" = None,
    ) -> dict[str, Any]:
        """Evaluate whether retraining should be triggered.

        Steps
        -----
        1. Record ``current_mape`` in ``DriftMonitor.record_mape``.
        2. Call ``DriftMonitor.check_concept_drift()`` for the primary
           concept-drift signal.
        3. Optionally inspect ``data_drift_results`` for secondary signal:
           if *any* feature has ``action == "AUTO_RETRAIN"``, the trigger
           fires even before consecutive days are exhausted.
        4. Append the full verdict to ``_history``.

        Parameters
        ----------
        current_mape : float
            MAPE measured in the current evaluation window, expressed as a
            fraction (e.g. 0.18 for 18 %).
        timestamp : datetime or None
            UTC timestamp of the evaluation.  Defaults to
            ``datetime.utcnow()``.
        data_drift_results : list[DriftResult] or None
            Optional list of data-drift results from
            ``DriftMonitor.check_data_drift``.  If any result has
            ``action == "AUTO_RETRAIN"``, the trigger fires with reason
            ``"DATA_DRIFT_AUTO_RETRAIN"``.

        Returns
        -------
        dict
            Keys:

            ``should_retrain`` : bool
                ``True`` if the pipeline should initiate retraining now.
            ``reason`` : str
                Human-readable reason code.  One of:

                * ``"OK"`` — no trigger condition met.
                * ``"CONCEPT_DRIFT_MAPE"`` — consecutive MAPE threshold
                  breached.
                * ``"DATA_DRIFT_AUTO_RETRAIN"`` — a feature has a severe
                  data distribution shift.
                * ``"CONCEPT_AND_DATA_DRIFT"`` — both conditions fired.
            ``consecutive_count`` : int
                Number of trailing consecutive above-threshold days at
                time of check.
            ``current_mape`` : float
                The MAPE value passed in.
            ``timestamp`` : str
                ISO-8601 UTC timestamp.
            ``data_drift_triggered`` : bool
                Whether the data drift secondary signal fired.
        """
        ts = timestamp or datetime.utcnow()
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        # ---- Record MAPE and query concept drift -----------------------
        self.drift_monitor.record_mape(ts, current_mape)
        concept_result = self.drift_monitor.check_concept_drift()

        concept_triggered = concept_result.action == "AUTO_RETRAIN"
        consecutive_count: int = concept_result.metadata.get("consecutive_count", 0)

        # ---- Optional data drift secondary signal ----------------------
        data_triggered = False
        triggering_features: list[str] = []
        if data_drift_results:
            for dr in data_drift_results:
                if dr.action == "AUTO_RETRAIN":
                    data_triggered = True
                    triggering_features.append(dr.feature_name)

        # ---- Compose verdict ------------------------------------------
        should_retrain = concept_triggered or data_triggered

        if concept_triggered and data_triggered:
            reason = "CONCEPT_AND_DATA_DRIFT"
        elif concept_triggered:
            reason = "CONCEPT_DRIFT_MAPE"
        elif data_triggered:
            reason = "DATA_DRIFT_AUTO_RETRAIN"
        else:
            reason = "OK"

        verdict: dict[str, Any] = {
            "should_retrain": should_retrain,
            "reason": reason,
            "consecutive_count": consecutive_count,
            "current_mape": current_mape,
            "timestamp": ts_str,
            "data_drift_triggered": data_triggered,
            "triggering_features": triggering_features,
            "concept_drift_statistic": concept_result.statistic,
        }

        self._history.append(verdict)

        logger.info(
            "RetrainTrigger.check: mape=%.4f, consecutive=%d/%d, "
            "concept=%s, data=%s → should_retrain=%s, reason=%r",
            current_mape,
            consecutive_count,
            self.consecutive_days,
            concept_triggered,
            data_triggered,
            should_retrain,
            reason,
        )
        return verdict

    def reset(self) -> None:
        """Clear the MAPE history in DriftMonitor after a successful retrain.

        This restarts the consecutive-days counter.  The trigger's own
        ``_history`` is **not** cleared — it is a permanent audit trail.

        Notes
        -----
        Call this method immediately after the retraining pipeline
        completes successfully.  Do not call it speculatively; a missed
        ``reset()`` is harmless (the counter will naturally reset once
        MAPE drops below threshold), but a spurious ``reset()`` discards
        valid drift signal.
        """
        self.drift_monitor._mape_history.clear()
        logger.info(
            "RetrainTrigger.reset: DriftMonitor MAPE history cleared. "
            "Consecutive counter restarted."
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Return the full audit trail of all ``check`` calls.

        Returns
        -------
        list[dict]
            Deep copy of the history list.  Each entry mirrors the dict
            returned by ``check``.  Entries are in chronological order
            (oldest first).
        """
        return list(self._history)
