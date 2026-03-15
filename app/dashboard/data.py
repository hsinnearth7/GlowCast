"""Dashboard data loaders — standalone simulators for cost analytics KPIs.

All data is generated deterministically (seed=42) without heavy ML libraries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(42)


# ── Cost Overview ────────────────────────────────────────────────────────

def load_cost_overview() -> dict:
    """Executive cost KPIs."""
    return {
        "avg_cost_variance_pct": 0.068,
        "should_cost_gap_pct": 0.112,
        "savings_realized_pct": 0.074,
        "supplier_on_time_pct": 0.943,
        "quality_yield_pct": 0.971,
        "ate_cost_reduction": -2.34,
        "cuped_variance_reduction": 0.55,
        "n_skus": 500,
        "n_plants": 12,
        "n_suppliers": 5,
    }


def load_should_cost_breakdown() -> pd.DataFrame:
    """Should-cost waterfall by cost element."""
    return pd.DataFrame({
        "element": ["Raw Materials", "Labor", "Overhead", "Logistics", "Tariff", "Margin/Gap"],
        "should_cost": [42.0, 25.0, 15.0, 5.0, 5.0, 0.0],
        "actual_cost": [45.5, 27.0, 16.2, 5.8, 5.5, 8.0],
        "gap": [3.5, 2.0, 1.2, 0.8, 0.5, 8.0],
    })


def load_ocogs_trend() -> pd.DataFrame:
    """Monthly OCOGS tracking over 12 months."""
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    base = 52.0
    noise = _RNG.normal(0, 1.5, 12)
    trend = np.linspace(0, -2.5, 12)
    costs = base + trend + noise
    return pd.DataFrame({
        "month": months,
        "avg_unit_cost": np.round(costs, 2),
        "target_cost": np.round(np.full(12, base * 0.92), 2),
        "volume": (_RNG.negative_binomial(20, 0.3, 12) * 100).astype(int),
    })


def load_cost_variance_heatmap() -> pd.DataFrame:
    """Plant x Category cost variance heatmap."""
    plants = ["Shenzhen", "Shanghai", "Taipei", "Munich", "Stuttgart",
              "Detroit", "Austin", "Guadalajara", "Pune", "Chennai", "Tokyo", "Osaka"]
    categories = ["RawMaterials", "Components", "Packaging", "Labor", "Overhead"]
    rows = []
    for p in plants:
        for c in categories:
            rows.append({
                "plant": p,
                "category": c,
                "variance_pct": round(float(_RNG.normal(0.05, 0.08)), 3),
            })
    return pd.DataFrame(rows)


def load_make_vs_buy_comparison() -> pd.DataFrame:
    """Make-vs-buy comparison for top 20 SKUs."""
    rows = []
    for i in range(20):
        make = 15 + _RNG.normal(0, 8)
        buy = make * (0.85 + _RNG.uniform(0, 0.4))
        rows.append({
            "sku_id": f"SKU_{i+1:04d}",
            "make_cost": round(float(max(5, make)), 2),
            "buy_cost": round(float(max(5, buy)), 2),
            "recommendation": "make" if make < buy else "buy",
            "composite_make": round(float(0.4 + _RNG.uniform(0, 0.4)), 3),
            "composite_buy": round(float(0.4 + _RNG.uniform(0, 0.4)), 3),
        })
    return pd.DataFrame(rows)


def load_supplier_performance() -> pd.DataFrame:
    """Supplier performance metrics."""
    return pd.DataFrame({
        "supplier": ["SupplierAlpha", "SupplierBeta", "SupplierGamma", "SupplierDelta", "SupplierEpsilon"],
        "quality_score": [0.92, 0.96, 0.98, 0.94, 0.88],
        "on_time_pct": [0.95, 0.98, 0.97, 0.93, 0.90],
        "price_premium": [0.00, 0.08, 0.15, 0.12, -0.05],
        "lead_time_days": [14, 10, 7, 5, 21],
        "total_orders": [1200, 800, 600, 900, 500],
    })


def load_commodity_price_index() -> pd.DataFrame:
    """Commodity price index over 12 months."""
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    commodities = ["Steel", "Copper", "Resin", "Aluminum", "Silicon"]
    rows = []
    for comm in commodities:
        base = 1.0
        for m in months:
            base = base * (1 + _RNG.normal(0, 0.03))
            rows.append({"month": m, "commodity": comm, "price_index": round(float(base), 4)})
    return pd.DataFrame(rows)


def load_cost_reduction_tracking() -> pd.DataFrame:
    """Cost reduction action tracking."""
    actions = ["supplier_switch", "process_optimization", "material_substitution",
               "volume_consolidation", "design_change", "automation", "nearshoring", "negotiate_contract"]
    rows = []
    for action in actions:
        n = int(_RNG.integers(5, 20))
        projected = round(float(_RNG.uniform(0.03, 0.12)), 3)
        realization = round(float(_RNG.uniform(0.6, 1.1)), 3)
        rows.append({
            "action_type": action,
            "count": n,
            "avg_projected_pct": projected,
            "avg_actual_pct": round(projected * realization, 3),
            "realization_rate": realization,
        })
    return pd.DataFrame(rows)


def load_cost_anomalies() -> pd.DataFrame:
    """Cost anomaly detection over time."""
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    cost = 50 + np.cumsum(_RNG.normal(0, 0.3, 365))
    z_scores = np.zeros(365)
    for i in range(30, 365):
        window = cost[i-30:i]
        z_scores[i] = (cost[i] - window.mean()) / max(window.std(), 0.01)
    return pd.DataFrame({
        "date": dates,
        "avg_cost": np.round(cost, 2),
        "z_score": np.round(z_scores, 2),
        "is_anomaly": np.abs(z_scores) > 2.5,
    })


# ── Causal & Experimentation (kept, re-contextualized) ──────────────────

def load_dowhy_results() -> dict:
    """DoWhy ATE results for cost reduction treatment."""
    return {
        "ate": -2.34,
        "ci_lower": -3.58,
        "ci_upper": -1.10,
        "p_value": 0.002,
        "treatment": "cost_reduction_action",
        "outcome": "unit_cost_change",
        "n_obs": 125_000,
        "refutations": {
            "random_common_cause": {"p_value": 0.78, "passed": True},
            "placebo_treatment": {"ate": 0.12, "passed": True},
            "data_subset": {"ate": -2.28, "passed": True},
        },
    }


def load_cuped_results() -> dict:
    """CUPED variance reduction results for cost experiments."""
    return {
        "rho": 0.74,
        "theta": -0.68,
        "variance_reduction": 0.55,
        "raw_ci": (0.72, 0.88),
        "cuped_ci": (0.42, 0.48),
        "sample_size_reduction": {
            "mde_3pct": {"raw": 8500, "cuped": 3825, "reduction": "55%"},
            "mde_5pct": {"raw": 3060, "cuped": 1377, "reduction": "55%"},
            "mde_10pct": {"raw": 765, "cuped": 345, "reduction": "55%"},
        },
    }


def load_uplift_results() -> pd.DataFrame:
    """Uplift learner AUUC comparison for cost interventions."""
    return pd.DataFrame({
        "learner": ["S-Learner", "T-Learner", "X-Learner", "Causal Forest"],
        "auuc": [0.62, 0.68, 0.74, 0.71],
        "description": [
            "Single model with treatment indicator",
            "Separate models per treatment arm",
            "4-stage with propensity weighting",
            "DML with gradient boosting",
        ],
    })


def load_uplift_curve() -> pd.DataFrame:
    """Uplift curve data for visualization."""
    n = 100
    frac = np.linspace(0, 1, n)
    x_learner = 0.74 * (1 - (1 - frac) ** 2)
    random = frac
    return pd.DataFrame({
        "fraction_treated": np.round(frac, 3),
        "x_learner_uplift": np.round(x_learner, 4),
        "random_uplift": np.round(random, 4),
    })


def load_sequential_test() -> pd.DataFrame:
    """Sequential test p-value trajectory."""
    n = 50
    days = list(range(1, n + 1))
    p_values = []
    p = 0.5
    for _ in days:
        p = max(0.001, p * (0.95 + _RNG.normal(0, 0.03)))
        p_values.append(round(float(p), 4))
    return pd.DataFrame({"day": days, "p_value": p_values, "alpha": [0.05] * n})


# ── MLOps (kept, re-contextualized) ────────────────────────────────────

def load_drift_history() -> pd.DataFrame:
    """Cost model drift monitoring over 12 months."""
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    return pd.DataFrame({
        "month": months,
        "cost_mape": np.round(0.10 + _RNG.normal(0, 0.02, 12), 3),
        "ks_p_value": np.round(np.clip(0.3 + _RNG.normal(0, 0.15, 12), 0.01, 0.99), 3),
        "psi": np.round(np.clip(0.05 + _RNG.normal(0, 0.03, 12), 0, 0.3), 3),
    })


def load_feature_importance() -> pd.DataFrame:
    """Feature importance for cost prediction model."""
    features = [
        ("commodity_index", "Commodity", 0.28),
        ("labor_rate", "Labor", 0.22),
        ("overhead_rate", "Overhead", 0.15),
        ("volume", "Volume", 0.12),
        ("supplier_quality", "Supplier", 0.08),
        ("tariff_rate", "Tariff", 0.06),
        ("capacity_util", "Plant", 0.05),
        ("lead_time", "Supplier", 0.04),
    ]
    return pd.DataFrame([
        {"feature": f, "category": c, "shap_importance": s, "lime_importance": round(s + _RNG.normal(0, 0.02), 3)}
        for f, c, s in features
    ])


def load_fairness_analysis() -> dict:
    """Fairness analysis across cost segments and plants."""
    categories = ["RawMaterials", "Components", "Packaging", "Labor", "Overhead"]
    tiers = ["Direct", "Indirect"]
    plants = ["Shenzhen", "Shanghai", "Munich", "Detroit", "Pune", "Tokyo"]

    cat_tier_mape = {}
    for cat in categories:
        for tier in tiers:
            cat_tier_mape[(cat, tier)] = round(0.08 + float(_RNG.normal(0, 0.03)), 3)

    plant_mape = {p: round(0.10 + float(_RNG.normal(0, 0.025)), 3) for p in plants}

    return {
        "category_tier_mape": cat_tier_mape,
        "plant_mape": plant_mape,
        "kruskal_wallis_p": 0.34,
        "chi2_p": 0.41,
    }


def load_platform_flow() -> pd.DataFrame:
    """Platform data flow (Sankey diagram data)."""
    return pd.DataFrame({
        "source": [
            "Cost Transactions", "Cost Transactions", "Supplier Quotes",
            "Commodity Prices", "Should-Cost Model", "OCOGS Tracker",
            "DoWhy Pipeline", "CUPED Analyzer", "Cost Reduction",
        ],
        "target": [
            "Should-Cost Model", "OCOGS Tracker", "Make-vs-Buy",
            "Should-Cost Model", "Cost Reduction", "Causal Analysis",
            "Cost Reduction", "A/B Testing", "Dashboard",
        ],
        "value": [300, 250, 200, 150, 180, 160, 140, 120, 200],
    })
