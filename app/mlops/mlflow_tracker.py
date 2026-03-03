"""GlowCast ExperimentTracker — MLflow wrapper with local JSON fallback.

Architecture
------------
The tracker follows a **try-MLflow → fall-back-to-JSON** strategy:

1. On import the module attempts to import ``mlflow``.
2. If MLflow is available, all tracking calls delegate to the MLflow
   Python client (runs, metrics, artifacts, model registry).
3. If MLflow is not installed (e.g. lightweight inference container),
   all data is written to a local JSON log at
   ``~/.glowcast/mlruns/<experiment_name>.jsonl``.  Each line is a
   newline-delimited JSON record (NDJSON), making the file trivially
   parseable by any analytics tool.

Champion / Challenger promotion
--------------------------------
``promote_champion`` implements a simple rule: if *every* metric in
``challenger_metrics`` is better than (i.e. lower than for MAPE/RMSE,
higher than for others) the corresponding metric in
``champion_metrics``, the challenger is promoted to ``"Production"``
and the current champion is archived.

Usage
-----
    tracker = ExperimentTracker("glowcast_routing_ensemble")

    with tracker.start_run("xgboost_fold_3", params={"lr": 0.05}):
        tracker.log_metrics({"mape": 0.112, "rmse": 4.3})
        tracker.log_artifact("/tmp/feature_importance.png")
        tracker.register_model("xgboost_v2")

    verdict = tracker.promote_champion(
        "xgboost_v2",
        challenger_metrics={"mape": 0.099},
        champion_metrics={"mape": 0.112},
    )
    print(verdict)  # "promoted" or "retained"
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MLflow import
# ---------------------------------------------------------------------------

try:
    import mlflow  # type: ignore[import]
    import mlflow.sklearn  # type: ignore[import]
    from mlflow.tracking import MlflowClient  # type: ignore[import]
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False
    MlflowClient = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Fallback JSON log directory
# ---------------------------------------------------------------------------

_FALLBACK_LOG_DIR = Path.home() / ".glowcast" / "mlruns"

# Metrics where *lower* is better (used in champion promotion logic)
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"mape", "wmape", "rmse", "mae", "loss", "error"}
)


class ExperimentTracker:
    """MLflow experiment tracker with transparent JSON fallback.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment (created if it does not exist).
        Also used as the filename stem for the fallback JSON log.
    tracking_uri : str or None
        MLflow tracking server URI.  Defaults to the ``MLFLOW_TRACKING_URI``
        environment variable, or ``"mlruns"`` (local file store) if unset.

    Attributes
    ----------
    experiment_name : str
    _active_run_id : str or None
        Set while inside a ``start_run`` context; ``None`` otherwise.
    _fallback_log_path : Path
        Path to the NDJSON fallback log file.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self._active_run_id: str | None = None
        self._active_run_name: str | None = None

        # Fallback JSON log
        _FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._fallback_log_path: Path = _FALLBACK_LOG_DIR / f"{experiment_name}.jsonl"

        if _MLFLOW_AVAILABLE:
            uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            logger.info(
                "ExperimentTracker using MLflow — experiment=%r, uri=%r",
                experiment_name,
                uri,
            )
        else:  # pragma: no cover
            logger.warning(
                "MLflow not available. Falling back to local JSON log at %s.",
                self._fallback_log_path,
            )

    # ------------------------------------------------------------------
    # Public — run context manager
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager that wraps a single training or evaluation run.

        On enter: starts an MLflow run (or writes a ``run_start`` JSON
        record) and logs ``params`` if provided.
        On exit: ends the run (or writes a ``run_end`` JSON record).

        Parameters
        ----------
        run_name : str
            Human-readable name for the run.
        params : dict or None
            Hyperparameters / configuration to log.

        Yields
        ------
        None
            Use ``tracker.log_metrics(...)`` inside the block.

        Example
        -------
        ::

            with tracker.start_run("lgbm_fold_1", params={"n_estimators": 500}):
                tracker.log_metrics({"mape": 0.105})
        """
        self._active_run_name = run_name
        start_ts = datetime.utcnow().isoformat()

        if _MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=run_name) as run:
                self._active_run_id = run.info.run_id
                if params:
                    mlflow.log_params(params)
                logger.info("MLflow run started: %s (id=%s)", run_name, self._active_run_id)
                try:
                    yield
                finally:
                    self._active_run_id = None
                    self._active_run_name = None
                    logger.info("MLflow run ended: %s", run_name)
        else:  # pragma: no cover
            self._active_run_id = f"local_{run_name}_{start_ts}"
            self._write_fallback(
                {
                    "event": "run_start",
                    "run_name": run_name,
                    "run_id": self._active_run_id,
                    "params": params or {},
                    "timestamp": start_ts,
                }
            )
            try:
                yield
            finally:
                self._write_fallback(
                    {
                        "event": "run_end",
                        "run_name": run_name,
                        "run_id": self._active_run_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                self._active_run_id = None
                self._active_run_name = None

    # ------------------------------------------------------------------
    # Public — logging helpers
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a dictionary of scalar metrics to the active run.

        Parameters
        ----------
        metrics : dict[str, float]
            Mapping of metric name → scalar value.

        Raises
        ------
        RuntimeError
            If called outside a ``start_run`` context.
        """
        self._assert_active_run()

        if _MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics)
        else:  # pragma: no cover
            self._write_fallback(
                {
                    "event": "metrics",
                    "run_id": self._active_run_id,
                    "metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        logger.debug("log_metrics: %s", metrics)

    def log_artifact(self, filepath: str) -> None:
        """Upload a local file as a run artifact.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the file to log.

        Raises
        ------
        RuntimeError
            If called outside a ``start_run`` context.
        FileNotFoundError
            If ``filepath`` does not exist on disk.
        """
        self._assert_active_run()

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {filepath}")

        if _MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(path))
        else:  # pragma: no cover
            self._write_fallback(
                {
                    "event": "artifact",
                    "run_id": self._active_run_id,
                    "filepath": str(path.resolve()),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        logger.debug("log_artifact: %s", filepath)

    # ------------------------------------------------------------------
    # Public — model registry
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        stage: str = "staging",
    ) -> None:
        """Register the model from the active run in the MLflow Model Registry.

        Parameters
        ----------
        model_name : str
            Registry model name (e.g. ``"glowcast_routing_ensemble"``).
        stage : str
            Target stage.  Typically ``"staging"`` after evaluation,
            ``"production"`` after champion promotion.

        Raises
        ------
        RuntimeError
            If called outside a ``start_run`` context.
        """
        self._assert_active_run()

        if _MLFLOW_AVAILABLE:
            model_uri = f"runs:/{self._active_run_id}/model"
            registered = mlflow.register_model(model_uri=model_uri, name=model_name)
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=registered.version,
                stage=stage.capitalize(),
            )
            logger.info(
                "Registered model %r version=%s → stage=%s",
                model_name,
                registered.version,
                stage,
            )
        else:  # pragma: no cover
            self._write_fallback(
                {
                    "event": "register_model",
                    "run_id": self._active_run_id,
                    "model_name": model_name,
                    "stage": stage,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    def promote_champion(
        self,
        model_name: str,
        challenger_metrics: dict[str, float],
        champion_metrics: dict[str, float],
    ) -> str:
        """Compare challenger to champion and promote if challenger wins.

        Promotion rule
        ~~~~~~~~~~~~~~
        For each metric present in both ``challenger_metrics`` and
        ``champion_metrics``:

        * If the metric name contains any of the strings in
          ``_LOWER_IS_BETTER`` (e.g. ``"mape"``, ``"rmse"``), *lower*
          challenger value → better.
        * Otherwise, *higher* challenger value → better.

        The challenger is promoted if it is *strictly better* on **all**
        shared metrics.  A tie → ``"retained"`` (champion survives).

        Parameters
        ----------
        model_name : str
            Registry model name to promote.
        challenger_metrics : dict[str, float]
            Evaluation metrics for the challenger model.
        champion_metrics : dict[str, float]
            Evaluation metrics for the current champion model.

        Returns
        -------
        str
            ``"promoted"`` if the challenger became the new production
            model; ``"retained"`` if the champion was kept.
        """
        shared_keys = set(challenger_metrics) & set(champion_metrics)
        if not shared_keys:
            logger.warning(
                "promote_champion: no shared metrics between challenger %s and champion %s; "
                "retaining champion.",
                challenger_metrics,
                champion_metrics,
            )
            return "retained"

        challenger_wins_all = True
        for key in shared_keys:
            c_val = challenger_metrics[key]
            p_val = champion_metrics[key]
            lower_better = any(token in key.lower() for token in _LOWER_IS_BETTER)

            if lower_better:
                wins = c_val < p_val
            else:
                wins = c_val > p_val

            if not wins:
                challenger_wins_all = False
                logger.debug(
                    "Challenger does not win on %r: challenger=%.4f, champion=%.4f (lower_better=%s)",
                    key,
                    c_val,
                    p_val,
                    lower_better,
                )
                break

        verdict: str
        if challenger_wins_all:
            verdict = "promoted"
            if _MLFLOW_AVAILABLE:
                client = MlflowClient()
                versions = client.get_latest_versions(model_name, stages=["Production"])
                for v in versions:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived",
                    )
                staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
                for v in staging_versions:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Production",
                    )
            else:  # pragma: no cover
                self._write_fallback(
                    {
                        "event": "model_promoted",
                        "model_name": model_name,
                        "challenger_metrics": challenger_metrics,
                        "champion_metrics": champion_metrics,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
        else:
            verdict = "retained"
            if not _MLFLOW_AVAILABLE:  # pragma: no cover
                self._write_fallback(
                    {
                        "event": "model_retained",
                        "model_name": model_name,
                        "challenger_metrics": challenger_metrics,
                        "champion_metrics": champion_metrics,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        logger.info(
            "promote_champion(%r): verdict=%r | challenger=%s | champion=%s",
            model_name,
            verdict,
            challenger_metrics,
            champion_metrics,
        )
        return verdict

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_active_run(self) -> None:
        """Raise RuntimeError if no run is currently active."""
        if self._active_run_id is None:
            raise RuntimeError(
                "No active run. Wrap this call inside a `start_run` context manager."
            )

    def _write_fallback(self, record: dict[str, Any]) -> None:
        """Append a JSON record to the fallback NDJSON log file.

        Parameters
        ----------
        record : dict
            Serialisable dictionary to append as a single JSON line.
        """
        with self._fallback_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
