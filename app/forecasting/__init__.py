"""GlowCast forecasting module.

Provides demand forecasting models, factory, I/O contracts, hierarchical
reconciliation, and walk-forward evaluation for the beauty & skincare
supply chain platform.

Public API
----------
Models
    ForecastModel           -- Abstract base class (ABC + Strategy pattern)
    NaiveMovingAverage      -- Rolling window baseline
    SARIMAXForecaster       -- Statsmodels SARIMAX wrapper
    XGBoostForecaster       -- XGBoost with GlowCast feature engineering
    LightGBMForecaster      -- LightGBM with GlowCast feature engineering
    ChronosForecaster       -- Amazon Chronos-Bolt zero-shot wrapper
    RoutingEnsemble         -- Routing logic: cold-start / intermittent / mature

Factory
    ForecastModelFactory    -- Registry-based factory with config-driven create_all()

Contracts
    ForecastInput           -- Dataclass wrapping unique_id, Y_df, X_df, horizon
    ForecastOutput          -- Dataclass wrapping predictions DataFrame + metadata

Hierarchy
    HierarchicalReconciler  -- MinTrace reconciliation (OLS/WLS/MinT-Shrink)
    get_hierarchy_spec      -- 4-level National→Country→FC→SKU definition

Evaluation
    CVFoldResult            -- Immutable per-fold result dataclass
    walk_forward_cv         -- Expanding-window time-series cross-validation
    ConformalPredictor      -- Split conformal prediction intervals
    slice_evaluation        -- Per-segment MAPE/RMSE/CI/Cohen-d table
    wilcoxon_test           -- Paired Wilcoxon signed-rank test
    cohens_d                -- Cohen's d effect size
    confidence_interval     -- Parametric CI for the mean
    compute_mape            -- Mean Absolute Percentage Error
    compute_rmse            -- Root Mean Squared Error
    compute_wmape           -- Weighted MAPE
    summarise_cv            -- CV results → summary DataFrame

Usage
-----
    from app.forecasting import ForecastModelFactory, ForecastInput, ForecastOutput
    from app.forecasting import HierarchicalReconciler, get_hierarchy_spec
    from app.forecasting import walk_forward_cv, slice_evaluation, ConformalPredictor

    factory = ForecastModelFactory()
    models  = factory.create_all()

    inp = ForecastInput(unique_id="SKU_1001__FC_Phoenix", Y_df=y_df, horizon=14)
    model = models["routing_ensemble"]
    out   = model.fit(inp.Y_df).predict(inp.horizon)

    rec = HierarchicalReconciler()
    S   = rec.build_summing_matrix(s_df)
    reconciled = rec.reconcile(base_forecasts, method="mint_shrink")

    cv_results = walk_forward_cv(model, Y_df, n_windows=12, horizon=14)
    tbl = slice_evaluation(y_true_dict, y_pred_dict, segment_labels)
"""

from __future__ import annotations

from app.forecasting.contracts import ForecastInput, ForecastOutput
from app.forecasting.evaluation import (
    ConformalPredictor,
    CVFoldResult,
    cohens_d,
    compute_mape,
    compute_rmse,
    compute_wmape,
    confidence_interval,
    slice_evaluation,
    summarise_cv,
    walk_forward_cv,
    wilcoxon_test,
)
from app.forecasting.hierarchy import HierarchicalReconciler, get_hierarchy_spec
from app.forecasting.models import (
    ChronosForecaster,
    ForecastModel,
    ForecastModelFactory,
    LightGBMForecaster,
    NaiveMovingAverage,
    RoutingEnsemble,
    SARIMAXForecaster,
    XGBoostForecaster,
)

__all__ = [
    # ABC
    "ForecastModel",
    # Concrete models
    "NaiveMovingAverage",
    "SARIMAXForecaster",
    "XGBoostForecaster",
    "LightGBMForecaster",
    "ChronosForecaster",
    "RoutingEnsemble",
    # Factory
    "ForecastModelFactory",
    # I/O contracts
    "ForecastInput",
    "ForecastOutput",
    # Hierarchy
    "HierarchicalReconciler",
    "get_hierarchy_spec",
    # Evaluation
    "CVFoldResult",
    "ConformalPredictor",
    "walk_forward_cv",
    "slice_evaluation",
    "summarise_cv",
    "wilcoxon_test",
    "cohens_d",
    "confidence_interval",
    "compute_mape",
    "compute_rmse",
    "compute_wmape",
]
