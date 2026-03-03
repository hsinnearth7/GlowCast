"""GlowCast MLOps module — Level 2 production tooling.

Provides the full MLOps lifecycle for the GlowCast beauty & skincare
supply chain forecasting platform:

Components
----------
FeatureStore
    Dual-mode feature store with offline batch materialisation and
    online point-in-time serving.  Follows the AP > CP eventual
    consistency model: the offline store is the authoritative source of
    truth; the online cache is a best-effort, low-latency projection.

DriftMonitor
    Statistical drift detection covering data drift (KS-test),
    prediction drift (PSI), and concept drift (rolling MAPE history).
    Integrates with Evidently when available; falls back to SciPy.

ExperimentTracker
    MLflow experiment tracking with champion / challenger model
    promotion logic.  Falls back to local JSON file logging when
    MLflow is unavailable.

RetrainTrigger
    Decision layer that combines DriftMonitor signals into a single
    binary should_retrain verdict with a full audit trail.

Usage
-----
    from app.mlops import FeatureStore, DriftMonitor, ExperimentTracker, RetrainTrigger

    fs      = FeatureStore()
    monitor = DriftMonitor()
    tracker = ExperimentTracker("glowcast_v2")
    trigger = RetrainTrigger(monitor)
"""

from __future__ import annotations

from app.mlops.drift_monitor import DriftMonitor, DriftResult
from app.mlops.feature_store import FeatureStore
from app.mlops.mlflow_tracker import ExperimentTracker
from app.mlops.retrain_trigger import RetrainTrigger

__all__ = [
    "FeatureStore",
    "DriftMonitor",
    "DriftResult",
    "ExperimentTracker",
    "RetrainTrigger",
]
