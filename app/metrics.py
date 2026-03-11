"""Prometheus metrics for GlowCast.

Defines application-level metrics for monitoring forecast quality,
pipeline execution, drift detection, experimentation, and HTTP traffic.

Usage
-----
    from app.metrics import FORECAST_MAPE, observe_request, generate_latest

    FORECAST_MAPE.labels(segment="stable").set(0.08)
    observe_request(method="GET", endpoint="/api/health", status=200, duration=0.012)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest as _generate_latest,
)
from fastapi.responses import Response

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Custom registry (avoids polluting the global default with test artefacts)
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# HTTP metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "glowcast_http_requests_total",
    "Total HTTP requests processed",
    labelnames=["method", "endpoint", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "glowcast_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Forecast quality metrics
# ---------------------------------------------------------------------------

FORECAST_MAPE = Gauge(
    "glowcast_forecast_mape",
    "Current MAPE for the routing ensemble, by segment",
    labelnames=["segment"],
    registry=REGISTRY,
)

FORECAST_RMSE = Gauge(
    "glowcast_forecast_rmse",
    "Current RMSE for the routing ensemble, by segment",
    labelnames=["segment"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Uplift / causal metrics
# ---------------------------------------------------------------------------

UPLIFT_AUUC = Gauge(
    "glowcast_uplift_auuc",
    "Area Under the Uplift Curve for the active uplift model",
    labelnames=["learner"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Drift detection metrics
# ---------------------------------------------------------------------------

DRIFT_DETECTED = Gauge(
    "glowcast_drift_detected",
    "Whether drift has been detected (1 = drift, 0 = no drift)",
    labelnames=["drift_type", "segment"],
    registry=REGISTRY,
)

DRIFT_KS_PVALUE = Gauge(
    "glowcast_drift_ks_pvalue",
    "Kolmogorov-Smirnov p-value for feature drift",
    labelnames=["feature"],
    registry=REGISTRY,
)

DRIFT_PSI = Gauge(
    "glowcast_drift_psi",
    "Population Stability Index for prediction drift",
    labelnames=["segment"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Data quality metrics
# ---------------------------------------------------------------------------

DATA_QUALITY_SCORE = Gauge(
    "glowcast_data_quality_score",
    "Data quality score from Great Expectations validation (0.0-1.0)",
    labelnames=["dataset"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Pipeline metrics
# ---------------------------------------------------------------------------

PIPELINE_DURATION = Histogram(
    "glowcast_pipeline_duration_seconds",
    "Pipeline execution duration in seconds",
    labelnames=["pipeline_type"],
    buckets=(10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
    registry=REGISTRY,
)

PIPELINE_RUNS_TOTAL = Counter(
    "glowcast_pipeline_runs_total",
    "Total pipeline runs by type and outcome",
    labelnames=["pipeline_type", "status"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Experimentation metrics
# ---------------------------------------------------------------------------

EXPERIMENT_VARIANCE_REDUCTION = Gauge(
    "glowcast_experiment_variance_reduction",
    "CUPED variance reduction ratio for the active experiment",
    labelnames=["experiment_id"],
    registry=REGISTRY,
)

EXPERIMENT_SAMPLE_SIZE = Gauge(
    "glowcast_experiment_sample_size",
    "Current sample size per group",
    labelnames=["experiment_id", "group"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

ERRORS_TOTAL = Counter(
    "glowcast_errors_total",
    "Total application errors by type",
    labelnames=["error_type", "component"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Model serving metrics
# ---------------------------------------------------------------------------

MODEL_INFERENCE_DURATION = Summary(
    "glowcast_model_inference_duration_seconds",
    "Model inference latency",
    labelnames=["model_name"],
    registry=REGISTRY,
)

MODEL_INFERENCE_TOTAL = Counter(
    "glowcast_model_inference_total",
    "Total model inference requests",
    labelnames=["model_name", "status"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def observe_request(
    method: str,
    endpoint: str,
    status: int,
    duration: float,
) -> None:
    """Record an HTTP request in Prometheus metrics.

    Parameters
    ----------
    method : str
        HTTP method (GET, POST, etc.)
    endpoint : str
        Request path (e.g. "/api/health")
    status : int
        HTTP status code
    duration : float
        Request duration in seconds
    """
    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def update_forecast_metrics(segment_metrics: dict[str, dict[str, float]]) -> None:
    """Bulk-update forecast quality gauges.

    Parameters
    ----------
    segment_metrics : dict
        Mapping of segment name -> {"mape": float, "rmse": float}.
    """
    for segment, metrics in segment_metrics.items():
        if "mape" in metrics:
            FORECAST_MAPE.labels(segment=segment).set(metrics["mape"])
        if "rmse" in metrics:
            FORECAST_RMSE.labels(segment=segment).set(metrics["rmse"])


def update_drift_metrics(drift_results: list[dict]) -> None:
    """Update drift detection gauges from a list of check results.

    Parameters
    ----------
    drift_results : list[dict]
        Each dict has keys: drift_type, segment/feature, detected (bool), value.
    """
    for result in drift_results:
        drift_type = result.get("drift_type", "unknown")
        segment = result.get("segment", result.get("feature", "unknown"))
        detected = 1 if result.get("detected", False) else 0
        DRIFT_DETECTED.labels(drift_type=drift_type, segment=segment).set(detected)


def generate_latest() -> Response:
    """Generate Prometheus-format metrics output as a FastAPI Response."""
    data = _generate_latest(REGISTRY)
    return Response(
        content=data,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
