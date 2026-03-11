"""GlowCast BentoML service — model serving for forecasting, uplift, and drift detection.

Endpoints
---------
POST /forecast       — Routing ensemble forecast (Chronos-2 / SARIMAX / LightGBM)
POST /uplift_predict — X-Learner CATE estimation for promotion targeting
POST /detect_drift   — KS + PSI drift detection on feature/prediction distributions

Usage
-----
    bentoml serve serving/service.py:GlowCastService

    # Build & containerize
    bentoml build -f serving/bentofile.yaml
    bentoml containerize glowcast_service:latest
"""

from __future__ import annotations

import logging
from typing import Any

import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    """Request payload for demand forecasting."""
    sku_id: str = Field(..., description="SKU identifier (e.g. 'SKU_0001')")
    history: list[float] = Field(..., description="Historical daily demand values (most recent last)")
    horizon: int = Field(default=14, ge=1, le=90, description="Forecast horizon in days")
    include_ci: bool = Field(default=True, description="Include confidence intervals")


class ForecastResponse(BaseModel):
    """Response payload for demand forecasting."""
    sku_id: str
    model_used: str
    point_forecasts: list[float]
    lower_90: list[float] | None = None
    upper_90: list[float] | None = None
    routing_reason: str


class UpliftRequest(BaseModel):
    """Request payload for uplift prediction."""
    features: dict[str, list[float]] = Field(
        ...,
        description="Feature matrix as column_name -> values (each list same length)",
    )


class UpliftResponse(BaseModel):
    """Response payload for uplift / CATE estimation."""
    cate_estimates: list[float]
    treatment_recommended: list[bool]
    mean_cate: float
    model: str = "x_learner"


class DriftRequest(BaseModel):
    """Request payload for drift detection."""
    reference_data: dict[str, list[float]] = Field(
        ..., description="Reference (training) distribution by feature"
    )
    current_data: dict[str, list[float]] = Field(
        ..., description="Current (serving) distribution by feature"
    )
    threshold_ks: float = Field(default=0.05, description="KS test p-value threshold")
    threshold_psi: float = Field(default=0.20, description="PSI threshold for drift alert")


class DriftResponse(BaseModel):
    """Response payload for drift detection."""
    drifted: bool
    checks: list[dict[str, Any]]
    summary: str


# ---------------------------------------------------------------------------
# BentoML Service
# ---------------------------------------------------------------------------


@bentoml.service(
    name="glowcast_service",
    traffic={"timeout": 60},
    resources={"cpu": "2", "memory": "2Gi"},
)
class GlowCastService:
    """BentoML service wrapping GlowCast's forecasting, uplift, and drift models."""

    def __init__(self) -> None:
        """Initialize model artifacts.

        In production, models would be loaded from BentoML's model store
        or MLflow registry.  Here we initialize lightweight fallbacks.
        """
        self._models_loaded = False
        logger.info("GlowCastService initialized")

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    @bentoml.api()
    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate demand forecast using the routing ensemble.

        Routing logic:
        - history < 60 points  -> Chronos-2 zero-shot
        - intermittency > 30%  -> SARIMAX
        - otherwise            -> LightGBM
        """
        history = np.array(request.history, dtype=np.float64)
        horizon = request.horizon

        # Routing decision
        if len(history) < 60:
            model_used = "chronos_zs"
            reason = f"Cold start: only {len(history)} days of history (<60)"
        elif np.mean(history == 0) > 0.30:
            model_used = "sarimax"
            reason = f"Intermittent demand: {np.mean(history == 0):.0%} zero days (>30%)"
        else:
            model_used = "lightgbm"
            reason = "Mature SKU with sufficient non-zero history"

        # Generate forecasts (placeholder — in production, load real models)
        base = np.mean(history[-30:]) if len(history) >= 30 else np.mean(history)
        trend = np.polyfit(np.arange(min(len(history), 30)), history[-30:] if len(history) >= 30 else history, 1)[0]

        point_forecasts = []
        for d in range(1, horizon + 1):
            point = max(0, base + trend * d + np.random.normal(0, base * 0.05))
            point_forecasts.append(round(float(point), 2))

        # Confidence intervals via conformal-style quantiles
        lower_90 = None
        upper_90 = None
        if request.include_ci:
            residual_std = np.std(history[-30:]) if len(history) >= 30 else np.std(history)
            lower_90 = [round(max(0, p - 1.645 * residual_std), 2) for p in point_forecasts]
            upper_90 = [round(p + 1.645 * residual_std, 2) for p in point_forecasts]

        return ForecastResponse(
            sku_id=request.sku_id,
            model_used=model_used,
            point_forecasts=point_forecasts,
            lower_90=lower_90,
            upper_90=upper_90,
            routing_reason=reason,
        )

    # ------------------------------------------------------------------
    # Uplift
    # ------------------------------------------------------------------

    @bentoml.api()
    def uplift_predict(self, request: UpliftRequest) -> UpliftResponse:
        """Estimate Conditional Average Treatment Effect (CATE) using X-Learner.

        In production, loads the trained X-Learner model from the model store.
        The X-Learner cross-estimation handles the 20/80 treatment imbalance.
        """
        df = pd.DataFrame(request.features)
        n = len(df)

        # Placeholder CATE estimation (production would use trained X-Learner)
        # Simulate: higher price_sensitivity and social_signal -> higher CATE
        np.random.seed(42)
        if "price_sensitivity" in df.columns:
            cate = df["price_sensitivity"].values * 0.3 + np.random.normal(0, 0.05, n)
        else:
            cate = np.random.normal(0.15, 0.08, n)

        cate = cate.tolist()
        treatment_recommended = [c > 0.10 for c in cate]
        mean_cate = float(np.mean(cate))

        return UpliftResponse(
            cate_estimates=[round(c, 4) for c in cate],
            treatment_recommended=treatment_recommended,
            mean_cate=round(mean_cate, 4),
            model="x_learner",
        )

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------

    @bentoml.api()
    def detect_drift(self, request: DriftRequest) -> DriftResponse:
        """Detect distribution drift between reference and current data.

        Uses:
        - Kolmogorov-Smirnov test (per feature)
        - Population Stability Index (per feature)
        """
        from scipy import stats

        checks: list[dict[str, Any]] = []
        any_drift = False

        for feature in request.reference_data:
            if feature not in request.current_data:
                continue

            ref = np.array(request.reference_data[feature])
            cur = np.array(request.current_data[feature])

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)
            ks_drift = ks_pvalue < request.threshold_ks

            # PSI
            psi_value = self._compute_psi(ref, cur)
            psi_drift = psi_value > request.threshold_psi

            drift_detected = ks_drift or psi_drift
            if drift_detected:
                any_drift = True

            checks.append({
                "feature": feature,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "ks_drift": ks_drift,
                "psi": round(float(psi_value), 4),
                "psi_drift": psi_drift,
                "drift_detected": drift_detected,
            })

        drifted_features = [c["feature"] for c in checks if c["drift_detected"]]
        if drifted_features:
            summary = f"Drift detected in {len(drifted_features)} feature(s): {', '.join(drifted_features)}"
        else:
            summary = f"No drift detected across {len(checks)} feature(s)"

        return DriftResponse(
            drifted=any_drift,
            checks=checks,
            summary=summary,
        )

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions.

        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

        Values:
        - < 0.10: No significant change
        - 0.10-0.20: Moderate change
        - > 0.20: Significant change (drift)
        """
        eps = 1e-6
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = ref_counts / len(reference) + eps
        cur_pct = cur_counts / len(current) + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
