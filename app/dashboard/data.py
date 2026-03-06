"""Standalone data simulator for GlowCast dashboard.

Generates realistic KPI data matching documented target metrics
without requiring heavy ML libraries. All values are deterministic (seed=42).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Constants ──
CONCERNS = ["AntiAging", "Acne", "Hydrating", "Brightening", "SunProtection"]
TEXTURES = ["Lightweight", "Rich"]
BRANDS = ["LuxeVita", "SolGuard", "GlowRush", "PureBasics", "VelvetAura"]
COUNTRIES = {"US": 4, "DE": 1, "UK": 1, "JP": 2, "IN": 3}  # country: n_fcs
FC_IDS = [f"FC_{c}_{i+1}" for c, n in COUNTRIES.items() for i in range(n)]
PRICE_TIERS = ["Mass", "Prestige", "Luxury"]

SEGMENT_GENES = {
    ("AntiAging", "Lightweight"): dict(
        n=750, base_lambda=6, elasticity=-0.5, seasonal_amp=0.05,
        social_sens=0.15, shelf_life=730, mape_target=8.5),
    ("AntiAging", "Rich"): dict(
        n=1250, base_lambda=4, elasticity=-0.4, seasonal_amp=0.10,
        social_sens=0.08, shelf_life=730, mape_target=7.5),
    ("Acne", "Lightweight"): dict(
        n=625, base_lambda=14, elasticity=-2.0, seasonal_amp=0.30,
        social_sens=0.70, shelf_life=540, mape_target=13.0),
    ("Acne", "Rich"): dict(
        n=250, base_lambda=8, elasticity=-1.5, seasonal_amp=0.15,
        social_sens=0.40, shelf_life=730, mape_target=14.0),
    ("Hydrating", "Lightweight"): dict(
        n=500, base_lambda=16, elasticity=-1.5, seasonal_amp=0.40,
        social_sens=0.25, shelf_life=730, mape_target=10.0),
    ("Hydrating", "Rich"): dict(
        n=625, base_lambda=15, elasticity=-1.2, seasonal_amp=0.45,
        social_sens=0.15, shelf_life=910, mape_target=9.5),
    ("Brightening", "Lightweight"): dict(
        n=375, base_lambda=10, elasticity=-0.9, seasonal_amp=0.30,
        social_sens=0.50, shelf_life=450, mape_target=14.5),
    ("Brightening", "Rich"): dict(
        n=375, base_lambda=7, elasticity=-0.7, seasonal_amp=0.15,
        social_sens=0.30, shelf_life=730, mape_target=13.5),
    ("SunProtection", "Lightweight"): dict(
        n=500, base_lambda=12, elasticity=-0.6, seasonal_amp=0.90,
        social_sens=0.65, shelf_life=540, mape_target=15.5),
    ("SunProtection", "Rich"): dict(
        n=125, base_lambda=5, elasticity=-0.5, seasonal_amp=0.60,
        social_sens=0.30, shelf_life=540, mape_target=16.0),
}

MODELS = ["NaiveMA(30)", "SARIMAX", "XGBoost", "LightGBM", "Chronos-2 ZS", "Routing Ensemble"]
MODEL_COLORS = {
    "NaiveMA(30)": "#95a5a6", "SARIMAX": "#3498db", "XGBoost": "#e67e22",
    "LightGBM": "#9b59b6", "Chronos-2 ZS": "#1abc9c", "Routing Ensemble": "#e74c3c",
}

UPLIFT_LEARNERS = ["Random", "S-Learner", "T-Learner", "X-Learner", "Causal Forest"]


def load_model_comparison():
    """6-model performance comparison across 5 segments + overall."""
    data = {
        "NaiveMA(30)":      {"MAPE": 28.5, "RMSE": 22.1, "WMAPE": 27.8, "role": "Baseline"},
        "SARIMAX":          {"MAPE": 18.3, "RMSE": 15.2, "WMAPE": 17.6, "role": "Intermittent"},
        "XGBoost":          {"MAPE": 13.8, "RMSE": 10.1, "WMAPE": 13.2, "role": "Feature-rich"},
        "LightGBM":         {"MAPE": 12.5, "RMSE": 9.0,  "WMAPE": 12.0, "role": "Primary"},
        "Chronos-2 ZS":     {"MAPE": 19.0, "RMSE": 14.2, "WMAPE": 18.5, "role": "Cold-start"},
        "Routing Ensemble":  {"MAPE": 11.8, "RMSE": 8.3,  "WMAPE": 11.2, "role": "Production"},
    }
    rows = []
    for model, metrics in data.items():
        rows.append({"Model": model, **metrics})
    return pd.DataFrame(rows)


def load_segment_evaluation():
    """Slice evaluation: 5 segments with CIs and effect sizes."""
    segments = [
        {"Segment": "Overall", "MAPE": 12.0, "CI_lo": 11.2,
         "CI_hi": 12.8, "RMSE": 8.3, "Coverage": 91, "N_SKUs": 5000, "Cohen_d": None},
        {"Segment": "Stable (AntiAging)", "MAPE": 8.0, "CI_lo": 7.3,
         "CI_hi": 8.7, "RMSE": 5.1, "Coverage": 94, "N_SKUs": 3200, "Cohen_d": 1.2},
        {"Segment": "Seasonal (SunProt)", "MAPE": 15.0, "CI_lo": 13.8,
         "CI_hi": 16.2, "RMSE": 12.7, "Coverage": 89, "N_SKUs": 1200, "Cohen_d": 0.9},
        {"Segment": "Promo Days", "MAPE": 22.0, "CI_lo": 19.5,
         "CI_hi": 24.5, "RMSE": 18.4, "Coverage": 78, "N_SKUs": None, "Cohen_d": 2.5},
        {"Segment": "Cold Start (<60d)", "MAPE": 19.0, "CI_lo": 16.8,
         "CI_hi": 21.2, "RMSE": 14.2, "Coverage": 85, "N_SKUs": 600, "Cohen_d": 1.8},
    ]
    return pd.DataFrame(segments)


def load_walkforward_cv(n_folds=12):
    """12-fold walk-forward CV results per model."""
    base_date = datetime(2025, 1, 1)
    rows = []
    model_base = {"NaiveMA(30)": 28.5, "SARIMAX": 18.3, "XGBoost": 13.8,
                  "LightGBM": 12.5, "Chronos-2 ZS": 19.0, "Routing Ensemble": 11.8}
    for fold in range(n_folds):
        test_end = base_date - timedelta(days=fold * 30)
        test_start = test_end - timedelta(days=14)
        for model, base_mape in model_base.items():
            noise = np.random.normal(0, base_mape * 0.08)
            seasonal_bump = 2.0 * np.sin(fold * np.pi / 6)  # seasonal variation
            mape = max(3.0, base_mape + noise + seasonal_bump)
            rows.append({
                "Fold": fold, "Test_Start": test_start.strftime("%Y-%m-%d"),
                "Test_End": test_end.strftime("%Y-%m-%d"),
                "Model": model, "MAPE": round(mape, 1),
            })
    return pd.DataFrame(rows)


def load_inventory_dos():
    """Days of Supply / Weeks of Cover by FC and concern."""
    rows = []
    for fc in FC_IDS:
        country = fc.split("_")[1]
        for concern in CONCERNS:
            base_dos = np.random.uniform(15, 60)
            if concern == "SunProtection" and country in ["IN", "JP"]:
                base_dos *= 0.6  # faster turnover in hot climates
            if concern == "Hydrating" and country in ["DE", "UK"]:
                base_dos *= 1.3  # slower turnover
            units = int(np.random.uniform(500, 8000))
            avg_daily = units / base_dos if base_dos > 0 else 0
            woc = base_dos / 7
            if base_dos <= 14:
                action = "High_Risk"
            elif base_dos <= 30:
                action = "Monitor"
            else:
                action = "Healthy"
            rows.append({
                "FC": fc, "Country": country, "Concern": concern,
                "Units_on_Hand": units, "Avg_Daily_Sales": round(avg_daily, 1),
                "DoS": round(base_dos, 1), "WoC": round(woc, 1),
                "Inventory_Value": round(units * np.random.uniform(8, 45), 0),
                "Action": action,
            })
    return pd.DataFrame(rows)


def load_scrap_risk():
    """Scrap risk matrix by FC and concern."""
    rows = []
    for fc in FC_IDS:
        for concern in CONCERNS:
            gene = [v for k, v in SEGMENT_GENES.items() if k[0] == concern][0]
            shelf_life = gene["shelf_life"]
            days_remaining = np.random.randint(5, 120)
            at_risk_qty = int(np.random.uniform(50, 500))
            avg_daily = np.random.uniform(5, 30)
            projected = avg_daily * days_remaining
            unsold = max(0, at_risk_qty - projected)
            unit_cost = np.random.uniform(5, 40)
            loss = round(unsold * unit_cost, 2)
            if days_remaining <= 7:
                tier = "Critical"
            elif days_remaining <= 21:
                tier = "High"
            elif days_remaining <= 45:
                tier = "Medium"
            else:
                tier = "Low"
            rows.append({
                "FC": fc, "Concern": concern, "Shelf_Life": shelf_life,
                "Days_Remaining": days_remaining, "At_Risk_Qty": at_risk_qty,
                "Projected_Sell_Thru": round(projected, 0),
                "Unsold_Qty": round(unsold, 0), "Potential_Loss": loss,
                "Risk_Tier": tier,
            })
    return pd.DataFrame(rows)


def load_cross_zone():
    """Cross-zone fulfillment penalties by FC."""
    rows = []
    for fc in FC_IDS:
        country = fc.split("_")[1]
        total_orders = int(np.random.uniform(5000, 25000))
        cross_pct = np.random.uniform(0.05, 0.25)
        if country in ["IN", "JP"]:
            cross_pct *= 1.5  # more cross-zone in Asia
        cross_pct = min(cross_pct, 0.40)
        cross_orders = int(total_orders * cross_pct)
        local_orders = total_orders - cross_orders
        penalty = cross_orders * 50
        avg_local_days = np.random.uniform(1.5, 3.0)
        avg_cross_days = np.random.uniform(4.0, 8.0)
        rows.append({
            "FC": fc, "Country": country,
            "Total_Orders": total_orders,
            "Cross_Zone_Orders": cross_orders, "Local_Orders": local_orders,
            "Cross_Zone_Pct": round(cross_pct * 100, 1),
            "Penalty_USD": penalty,
            "Avg_Local_Days": round(avg_local_days, 1),
            "Avg_Cross_Days": round(avg_cross_days, 1),
        })
    return pd.DataFrame(rows)


def load_demand_anomalies(n_days=365):
    """Daily demand anomaly timeline."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        daily_units = int(np.random.poisson(500) + 200 * np.sin(d.dayofyear * 2 * np.pi / 365))
        z = np.random.normal(0, 1)
        if np.random.random() < 0.03:  # 3% viral
            z = np.random.uniform(2.5, 5.0)
            daily_units = int(daily_units * np.random.uniform(2, 4))
        anomaly = "Demand_Spike" if z > 2 else ("Demand_Drop" if z < -2 else "Normal")
        rows.append({"Date": d, "Daily_Units": daily_units, "Z_Score": round(z, 2), "Anomaly": anomaly})
    return pd.DataFrame(rows)


def load_social_lead_lag():
    """Social signal cross-correlation by concern and lag."""
    rows = []
    optimal_lags = {"AntiAging": 5, "Acne": 2, "Hydrating": 4, "Brightening": 3, "SunProtection": 3}
    for concern in CONCERNS:
        opt = optimal_lags[concern]
        for lag in range(1, 15):
            # Correlation peaks around optimal lag
            base_r = 0.7 * np.exp(-0.3 * (lag - opt) ** 2)
            r = base_r + np.random.normal(0, 0.05)
            r = np.clip(r, -0.2, 0.85)
            abs_r = abs(r)
            if abs_r > 0.7:
                rel = "Strong_Lead"
            elif abs_r > 0.5:
                rel = "Moderate_Lead"
            elif abs_r > 0.3:
                rel = "Weak_Lead"
            else:
                rel = "Negligible"
            rows.append({
                "Concern": concern, "Lag_Days": lag, "Pearson_R": round(r, 3),
                "Abs_R": round(abs_r, 3), "Relationship": rel,
            })
    return pd.DataFrame(rows)


def load_uplift_results():
    """Uplift learner comparison with bootstrap CIs."""
    learners = [
        {"Learner": "Random",        "AUUC": 0.50, "CI_lo": 0.48, "CI_hi": 0.52, "vs_Random": 0.00},
        {"Learner": "S-Learner",     "AUUC": 0.62, "CI_lo": 0.59, "CI_hi": 0.65, "vs_Random": 0.12},
        {"Learner": "T-Learner",     "AUUC": 0.68, "CI_lo": 0.64, "CI_hi": 0.72, "vs_Random": 0.18},
        {"Learner": "X-Learner",     "AUUC": 0.74, "CI_lo": 0.71, "CI_hi": 0.77, "vs_Random": 0.24},
        {"Learner": "Causal Forest", "AUUC": 0.71, "CI_lo": 0.68, "CI_hi": 0.74, "vs_Random": 0.21},
    ]
    return pd.DataFrame(learners)


def load_uplift_curve():
    """Uplift curve data (cumulative fraction vs uplift)."""
    fractions = np.linspace(0, 1, 101)
    curves = {}
    for learner, auuc in [("Random", 0.5), ("S-Learner", 0.62), ("T-Learner", 0.68),
                           ("X-Learner", 0.74), ("Causal Forest", 0.71)]:
        if learner == "Random":
            curves[learner] = fractions.copy()
        else:
            power = 1 + (1 - auuc) * 3
            curves[learner] = fractions ** (1 / power) * (auuc / 0.5)
            curves[learner] = np.clip(curves[learner], 0, 1)
    rows = []
    for i, f in enumerate(fractions):
        row = {"Fraction": round(f, 2)}
        for name, vals in curves.items():
            row[name] = round(vals[i], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def load_dowhy_results():
    """DoWhy 4-step causal inference results."""
    return {
        "treatment": "promotion_flag",
        "outcome": "units_sold",
        "ate": 3.42,
        "ci_lower": 2.18,
        "ci_upper": 4.66,
        "method": "backdoor.linear_regression",
        "n_obs": 125000,
        "refutations": [
            {"method": "random_common_cause", "new_ate": 3.38, "orig_ate": 3.42, "p_value": 0.87, "passed": True},
            {"method": "placebo_treatment",   "new_ate": 0.02, "orig_ate": 3.42, "p_value": 0.94, "passed": True},
            {"method": "data_subset",         "new_ate": 3.29, "orig_ate": 3.42, "p_value": 0.71, "passed": True},
        ],
    }


def load_cuped_results():
    """CUPED variance reduction analysis."""
    return {
        "rho": 0.74,
        "variance_reduction": 0.55,
        "theta": 0.68,
        "table": pd.DataFrame([
            {"MDE": "3%",  "n_raw": 42000, "n_cuped": 18900, "reduction": 55},
            {"MDE": "5%",  "n_raw": 15200, "n_cuped": 6840,  "reduction": 55},
            {"MDE": "10%", "n_raw": 3800,  "n_cuped": 1710,  "reduction": 55},
        ]),
    }


def load_sequential_test(n_steps=200):
    """Simulated sequential test trajectory."""
    rows = []
    true_delta = 0.03
    for t in range(1, n_steps + 1):
        n = t * 50
        obs_delta = true_delta + np.random.normal(0, 0.1 / np.sqrt(t))
        # Simplified p-value decay
        p = min(1.0, np.exp(-0.05 * t * obs_delta ** 2 * 50))
        stopped = p < 0.05 and t > 20
        rows.append({
            "Step": t, "N_total": n * 2,
            "Observed_Delta": round(obs_delta, 4),
            "P_value": round(p, 4), "Stopped": stopped,
        })
    return pd.DataFrame(rows)


def load_drift_history(n_days=90):
    """Drift monitoring daily history."""
    dates = pd.date_range("2024-10-01", periods=n_days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        mape = 11.8 + np.random.normal(0, 1.5)
        # Simulate drift event around day 60-70
        if 60 <= i <= 75:
            mape += (i - 60) * 0.8
        mape = max(5, mape)

        ks_p = np.random.uniform(0.01, 0.99)
        if 55 <= i <= 70:
            ks_p *= 0.1  # drift signal
        psi = np.random.uniform(0.01, 0.08)
        if 60 <= i <= 72:
            psi += 0.12

        rows.append({
            "Date": d, "MAPE": round(mape, 1),
            "KS_p_value": round(ks_p, 4), "PSI": round(psi, 3),
            "Data_Drift": ks_p < 0.05,
            "Pred_Drift": psi > 0.1,
            "Concept_Drift": mape > 20,
        })
    return pd.DataFrame(rows)


def load_feature_importance():
    """SHAP-style feature importance rankings."""
    features = [
        {"Feature": "lag_1",              "SHAP_mean": 0.42, "LIME_mean": 0.38, "Category": "Demand"},
        {"Feature": "rolling_mean_7",     "SHAP_mean": 0.35, "LIME_mean": 0.33, "Category": "Demand"},
        {"Feature": "lag_7",              "SHAP_mean": 0.28, "LIME_mean": 0.25, "Category": "Demand"},
        {"Feature": "rolling_std_7",      "SHAP_mean": 0.22, "LIME_mean": 0.20, "Category": "Demand"},
        {"Feature": "social_momentum_t3", "SHAP_mean": 0.18, "LIME_mean": 0.21, "Category": "Social"},
        {"Feature": "temperature",        "SHAP_mean": 0.15, "LIME_mean": 0.14, "Category": "Climate"},
        {"Feature": "month",              "SHAP_mean": 0.14, "LIME_mean": 0.12, "Category": "Calendar"},
        {"Feature": "lag_28",             "SHAP_mean": 0.12, "LIME_mean": 0.10, "Category": "Demand"},
        {"Feature": "rolling_mean_28",    "SHAP_mean": 0.10, "LIME_mean": 0.09, "Category": "Demand"},
        {"Feature": "humidity",           "SHAP_mean": 0.08, "LIME_mean": 0.07, "Category": "Climate"},
        {"Feature": "day_of_week",        "SHAP_mean": 0.06, "LIME_mean": 0.08, "Category": "Calendar"},
        {"Feature": "lag_14",             "SHAP_mean": 0.05, "LIME_mean": 0.06, "Category": "Demand"},
    ]
    return pd.DataFrame(features)


def load_fairness_results():
    """Fairness analysis across segments and FCs."""
    segment_rows = []
    for concern in CONCERNS:
        for texture in TEXTURES:
            key = (concern, texture)
            gene = SEGMENT_GENES[key]
            mape = gene["mape_target"] + np.random.normal(0, 0.5)
            ci_lo = mape - np.random.uniform(0.8, 1.5)
            ci_hi = mape + np.random.uniform(0.8, 1.5)
            segment_rows.append({
                "Concern": concern, "Texture": texture,
                "Segment": f"{concern}/{texture}",
                "MAPE": round(mape, 1),
                "CI_lo": round(ci_lo, 1), "CI_hi": round(ci_hi, 1),
                "N_SKUs": gene["n"],
            })

    fc_rows = []
    for fc in FC_IDS:
        country = fc.split("_")[1]
        base = 12.0
        if country in ["IN", "JP"]:
            base += np.random.uniform(0, 2)
        mape = base + np.random.normal(0, 1.0)
        fc_rows.append({"FC": fc, "Country": country, "MAPE": round(mape, 1)})

    return {
        "segments": pd.DataFrame(segment_rows),
        "fcs": pd.DataFrame(fc_rows),
        "kruskal_wallis": {"H": 2.31, "p": 0.51},
        "chi_squared": {"chi2": 8.7, "p": 0.12},
    }


def load_business_impact():
    """Business impact KPIs."""
    return pd.DataFrame([
        {"Metric": "Forecast MAPE",    "Before": "28.5% (Naive)", "After": "11.8% (Ensemble)", "Delta": "-59%"},
        {"Metric": "Scrap Rate",       "Before": "~15% (industry)", "After": "<2% (FIFO)",     "Delta": "-87%"},
        {"Metric": "Cross-Zone Ship",  "Before": "35% (uniform)",  "After": "<5% (climate)",    "Delta": "-86%"},
        {"Metric": "A/B Sample Size",  "Before": "15,200/group",   "After": "6,840/group",      "Delta": "-55%"},
        {"Metric": "Promo Budget",     "Before": "$X",             "After": "$0.7X (targeted)",  "Delta": "-30%"},
    ])


def load_ablation_study():
    """LightGBM feature ablation study results."""
    return pd.DataFrame([
        {"Configuration": "Full Model (LightGBM)",  "MAPE": 12.5, "Delta": 0.0,  "p_value": None,  "Significance": "Baseline"},
        {"Configuration": "- Lag (1,7,14,28)",       "MAPE": 15.7, "Delta": 3.2,  "p_value": 0.001, "Significance": "***"},
        {"Configuration": "- Promo features",        "MAPE": 14.2, "Delta": 1.7,  "p_value": 0.008, "Significance": "**"},
        {"Configuration": "- Price elasticity",      "MAPE": 13.3, "Delta": 0.8,  "p_value": 0.041, "Significance": "*"},
        {"Configuration": "- Social momentum",       "MAPE": 13.0, "Delta": 0.5,  "p_value": 0.112, "Significance": "n.s."},
        {"Configuration": "- Weather",               "MAPE": 12.7, "Delta": 0.2,  "p_value": 0.312, "Significance": "n.s."},
    ])


def load_platform_flow():
    """Sankey diagram data: platform data flow with metric values."""
    return {
        "labels": [
            "Raw Data (9 tables)",       # 0
            "Feature Store",             # 1
            "6 Forecast Models",         # 2
            "Routing Ensemble",          # 3
            "DoWhy Causal",              # 4
            "CUPED A/B Tests",           # 5
            "MAPE 11.8%",               # 6
            "Scrap <2%",                 # 7
            "Cross-Zone <5%",            # 8
            "AUUC 0.74",                # 9
            "Sample -55%",              # 10
            "Drift Monitor",            # 11
            "Auto-Retrain",             # 12
            "SQL Analytics",            # 13
            "Uplift Modeling",          # 14
        ],
        "sources": [0, 0, 1, 1, 1, 2, 3, 3, 4, 5, 0, 11, 0, 4],
        "targets": [1, 13, 2, 4, 5, 3, 6, 7, 9, 10, 11, 12, 8, 14],
        "values":  [40, 15, 30, 15, 10, 30, 25, 10, 10, 10, 8, 5, 8, 10],
        "colors": [
            "#3498db", "#1abc9c", "#9b59b6", "#e74c3c", "#e67e22",
            "#3498db", "#27ae60", "#27ae60", "#27ae60", "#27ae60",
            "#f39c12", "#f39c12", "#3498db", "#e67e22",
        ],
    }
