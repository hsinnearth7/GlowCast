"""Extended MLflow Model Registry — full lifecycle management.

Builds on ``mlflow_tracker.py``'s ExperimentTracker by adding:
- ``register_model_version``: Create a new model version with metadata
- ``transition_stage``: Move a model version through None → Staging → Production → Archived
- ``get_production_model``: Load the current production model for serving
- ``compare_models``: Statistical comparison of challenger vs champion
- ``list_model_versions``: Query all versions of a registered model

Usage
-----
    registry = ModelRegistry(tracking_uri="http://mlflow:5000")

    # Register after training
    version = registry.register_model_version(
        name="glowcast_routing_ensemble",
        run_id="abc123",
        metrics={"mape": 0.118, "rmse": 8.3},
        tags={"segment": "overall", "routing": "lightgbm"},
    )

    # Promote to production
    registry.transition_stage("glowcast_routing_ensemble", version, "Production")

    # Load for serving
    model = registry.get_production_model("glowcast_routing_ensemble")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MLflow import (mirrors mlflow_tracker.py pattern)
# ---------------------------------------------------------------------------

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    MlflowClient = None  # type: ignore[assignment,misc]
    ModelVersion = None  # type: ignore[assignment,misc]

_FALLBACK_REGISTRY_DIR = Path.home() / ".glowcast" / "registry"

# Valid stage transitions
_VALID_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"Staging", "Production"},
    "None": {"Staging", "Production"},
    "Staging": {"Production", "Archived"},
    "Production": {"Staging", "Archived"},
    "Archived": {"Staging", "Production"},
}


class ModelRegistry:
    """Full MLflow Model Registry lifecycle manager.

    Parameters
    ----------
    tracking_uri : str or None
        MLflow tracking server URI.  Defaults to MLFLOW_TRACKING_URI env var.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")

        _FALLBACK_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        self._fallback_path = _FALLBACK_REGISTRY_DIR / "registry.jsonl"

        if _MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self._uri)
            self._client = MlflowClient(tracking_uri=self._uri)
            logger.info("ModelRegistry connected to MLflow at %s", self._uri)
        else:
            self._client = None
            logger.warning(
                "MLflow not available — using fallback JSON registry at %s",
                self._fallback_path,
            )

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    def register_model_version(
        self,
        name: str,
        run_id: str,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
    ) -> str:
        """Register a new model version from a completed run.

        Parameters
        ----------
        name : str
            Registered model name (e.g. ``"glowcast_routing_ensemble"``).
        run_id : str
            MLflow run ID that produced the model artifact.
        metrics : dict
            Evaluation metrics to attach as version tags.
        tags : dict
            Additional metadata tags.
        description : str
            Human-readable description of this version.

        Returns
        -------
        str
            The version number (e.g. ``"3"``).
        """
        if _MLFLOW_AVAILABLE and self._client is not None:
            model_uri = f"runs:/{run_id}/model"

            # Ensure registered model exists
            try:
                self._client.get_registered_model(name)
            except Exception:
                self._client.create_registered_model(
                    name=name,
                    description=description or f"GlowCast model: {name}",
                )

            mv = self._client.create_model_version(
                name=name,
                source=model_uri,
                run_id=run_id,
                description=description,
            )

            # Tag with metrics
            if metrics:
                for k, v in metrics.items():
                    self._client.set_model_version_tag(name, mv.version, f"metric.{k}", str(v))

            if tags:
                for k, v in tags.items():
                    self._client.set_model_version_tag(name, mv.version, k, v)

            logger.info("Registered model %r version=%s from run=%s", name, mv.version, run_id)
            return str(mv.version)
        else:
            # Fallback
            version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            record = {
                "event": "register",
                "name": name,
                "version": version,
                "run_id": run_id,
                "metrics": metrics or {},
                "tags": tags or {},
                "description": description,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._write_fallback(record)
            logger.info("Registered model %r version=%s (fallback)", name, version)
            return version

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> str:
        """Transition a model version to a new stage.

        Parameters
        ----------
        name : str
            Registered model name.
        version : str
            Model version number.
        stage : str
            Target stage: ``"Staging"``, ``"Production"``, or ``"Archived"``.
        archive_existing : bool
            If transitioning to Production, archive the current production version.

        Returns
        -------
        str
            The new stage name.

        Raises
        ------
        ValueError
            If the stage transition is invalid.
        """
        if stage not in {"Staging", "Production", "Archived"}:
            raise ValueError(f"Invalid stage: {stage!r}. Must be Staging, Production, or Archived.")

        if _MLFLOW_AVAILABLE and self._client is not None:
            # Archive existing production model if promoting
            if archive_existing and stage == "Production":
                existing = self._client.get_latest_versions(name, stages=["Production"])
                for ev in existing:
                    if ev.version != version:
                        self._client.transition_model_version_stage(
                            name=name, version=ev.version, stage="Archived"
                        )
                        logger.info(
                            "Archived previous production model %r version=%s",
                            name, ev.version,
                        )

            self._client.transition_model_version_stage(
                name=name, version=version, stage=stage
            )
            logger.info("Transitioned %r version=%s to stage=%s", name, version, stage)
            return stage
        else:
            record = {
                "event": "transition",
                "name": name,
                "version": version,
                "stage": stage,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._write_fallback(record)
            logger.info("Transitioned %r version=%s to stage=%s (fallback)", name, version, stage)
            return stage

    # ------------------------------------------------------------------
    # Load production model
    # ------------------------------------------------------------------

    def get_production_model(self, name: str) -> Any:
        """Load the current production model for serving.

        Parameters
        ----------
        name : str
            Registered model name.

        Returns
        -------
        Any
            The loaded model object (e.g. sklearn estimator, LightGBM booster).

        Raises
        ------
        RuntimeError
            If no production model is found.
        """
        if _MLFLOW_AVAILABLE and self._client is not None:
            versions = self._client.get_latest_versions(name, stages=["Production"])
            if not versions:
                raise RuntimeError(f"No production model found for {name!r}")

            latest = versions[0]
            model_uri = f"models:/{name}/Production"
            logger.info(
                "Loading production model %r version=%s from %s",
                name, latest.version, model_uri,
            )

            # Try sklearn first, then generic pyfunc
            try:
                import mlflow.sklearn
                return mlflow.sklearn.load_model(model_uri)
            except Exception:
                return mlflow.pyfunc.load_model(model_uri)
        else:
            raise RuntimeError(
                f"Cannot load model {name!r}: MLflow not available. "
                "Use ExperimentTracker for local development."
            )

    # ------------------------------------------------------------------
    # Compare models
    # ------------------------------------------------------------------

    def compare_models(
        self,
        name: str,
        challenger_version: str,
        champion_version: str | None = None,
        metric_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare challenger version against the current champion (production).

        Parameters
        ----------
        name : str
            Registered model name.
        challenger_version : str
            Version number of the challenger.
        champion_version : str or None
            Version of the champion.  If None, uses current production version.
        metric_keys : list[str]
            Metrics to compare (reads from version tags ``metric.<key>``).

        Returns
        -------
        dict
            Comparison result with 'challenger', 'champion', 'verdict'.
        """
        if not _MLFLOW_AVAILABLE or self._client is None:
            return {"error": "MLflow not available", "verdict": "unknown"}

        # Resolve champion
        if champion_version is None:
            prod_versions = self._client.get_latest_versions(name, stages=["Production"])
            if not prod_versions:
                return {"error": "No production model to compare against", "verdict": "no_champion"}
            champion_version = prod_versions[0].version

        # Fetch tags
        challenger_mv = self._client.get_model_version(name, challenger_version)
        champion_mv = self._client.get_model_version(name, champion_version)

        keys = metric_keys or ["mape", "rmse", "wmape", "auuc"]
        lower_is_better = {"mape", "rmse", "wmape", "mae", "loss", "error"}

        challenger_metrics = {}
        champion_metrics = {}
        for key in keys:
            tag_key = f"metric.{key}"
            if tag_key in challenger_mv.tags:
                challenger_metrics[key] = float(challenger_mv.tags[tag_key])
            if tag_key in champion_mv.tags:
                champion_metrics[key] = float(champion_mv.tags[tag_key])

        # Compare
        wins = 0
        losses = 0
        comparisons = {}
        shared = set(challenger_metrics) & set(champion_metrics)
        for key in shared:
            c_val = challenger_metrics[key]
            p_val = champion_metrics[key]
            if key in lower_is_better:
                better = c_val < p_val
            else:
                better = c_val > p_val
            comparisons[key] = {
                "challenger": c_val,
                "champion": p_val,
                "challenger_wins": better,
            }
            if better:
                wins += 1
            else:
                losses += 1

        if wins > 0 and losses == 0:
            verdict = "promote"
        elif losses > 0 and wins == 0:
            verdict = "retain"
        else:
            verdict = "mixed"

        return {
            "challenger_version": challenger_version,
            "champion_version": champion_version,
            "comparisons": comparisons,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # List versions
    # ------------------------------------------------------------------

    def list_model_versions(
        self,
        name: str,
        stages: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List all versions of a registered model.

        Parameters
        ----------
        name : str
            Registered model name.
        stages : list[str]
            Filter by stages (e.g. ``["Production", "Staging"]``).

        Returns
        -------
        list[dict]
            Version metadata for each matching version.
        """
        if _MLFLOW_AVAILABLE and self._client is not None:
            if stages:
                versions = self._client.get_latest_versions(name, stages=stages)
            else:
                versions = self._client.search_model_versions(f"name='{name}'")

            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "description": v.description,
                    "tags": dict(v.tags) if v.tags else {},
                    "creation_timestamp": v.creation_timestamp,
                }
                for v in versions
            ]
        else:
            return []

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _write_fallback(self, record: dict[str, Any]) -> None:
        """Append record to fallback NDJSON file."""
        with self._fallback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
