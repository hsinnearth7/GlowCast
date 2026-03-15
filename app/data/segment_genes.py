"""Product taxonomy and cost segment definitions for GlowCast.

Domain model: Category × Cost Tier → 10 cost-behavior segments.
Each segment encodes cost genetics: base unit cost, volatility,
commodity sensitivity, labor intensity, and overhead allocation.

References:
    - v4.0 portfolio spec (Cost & Commercial Analytics)
"""

from __future__ import annotations

import numpy as np

# ── Primary axes ──────────────────────────────────────────────────────────

CATEGORIES = ["RawMaterials", "Components", "Packaging", "Labor", "Overhead"]
COST_TIERS = ["Direct", "Indirect"]

# ── Commodity groups ─────────────────────────────────────────────────────

COMMODITIES = ["Steel", "Copper", "Resin", "Aluminum", "Silicon"]

COMMODITY_BASE_PRICES = {
    "Steel": 850.0,       # $/ton
    "Copper": 8500.0,     # $/ton
    "Resin": 1200.0,      # $/ton
    "Aluminum": 2400.0,   # $/ton
    "Silicon": 3200.0,    # $/ton
}

# ── Suppliers ────────────────────────────────────────────────────────────

SUPPLIERS = ["SupplierAlpha", "SupplierBeta", "SupplierGamma", "SupplierDelta", "SupplierEpsilon"]

SUPPLIER_PROFILES = {
    "SupplierAlpha":   {"country": "CN", "quality_score": 0.92, "on_time_pct": 0.95, "lead_time_days": 14, "price_premium": 0.00},
    "SupplierBeta":    {"country": "TW", "quality_score": 0.96, "on_time_pct": 0.98, "lead_time_days": 10, "price_premium": 0.08},
    "SupplierGamma":   {"country": "DE", "quality_score": 0.98, "on_time_pct": 0.97, "lead_time_days": 7,  "price_premium": 0.15},
    "SupplierDelta":   {"country": "US", "quality_score": 0.94, "on_time_pct": 0.93, "lead_time_days": 5,  "price_premium": 0.12},
    "SupplierEpsilon": {"country": "IN", "quality_score": 0.88, "on_time_pct": 0.90, "lead_time_days": 21, "price_premium": -0.05},
}

# ── Segment Genes (10 segments: 5 categories × 2 cost tiers) ────────────

SEGMENT_GENES = {
    ("RawMaterials", "Direct"): {
        "sku_count": 25,
        "subcategories": ["Sheet Metal", "Polymer Pellets", "Wire Stock", "Chemical Compounds"],
        "base_unit_cost": 12.50,
        "cost_volatility": 0.25,
        "commodity_sensitivity": 0.80,
        "labor_intensity": 0.10,
        "overhead_allocation": 0.05,
        "seasonal_amplitude": 0.15,
        "seasonal_direction": "q4_peak",
        "tariff_exposure": 0.12,
        "quality_rejection_rate": 0.02,
        "moq": 1000,
    },
    ("RawMaterials", "Indirect"): {
        "sku_count": 15,
        "subcategories": ["Cleaning Solvents", "Lubricants", "Safety Supplies"],
        "base_unit_cost": 5.00,
        "cost_volatility": 0.10,
        "commodity_sensitivity": 0.30,
        "labor_intensity": 0.05,
        "overhead_allocation": 0.15,
        "seasonal_amplitude": 0.05,
        "seasonal_direction": "flat",
        "tariff_exposure": 0.05,
        "quality_rejection_rate": 0.01,
        "moq": 500,
    },
    ("Components", "Direct"): {
        "sku_count": 30,
        "subcategories": ["PCB Assemblies", "Connectors", "Sensors", "Motor Units", "Display Modules"],
        "base_unit_cost": 45.00,
        "cost_volatility": 0.18,
        "commodity_sensitivity": 0.60,
        "labor_intensity": 0.30,
        "overhead_allocation": 0.10,
        "seasonal_amplitude": 0.20,
        "seasonal_direction": "q3_peak",
        "tariff_exposure": 0.18,
        "quality_rejection_rate": 0.03,
        "moq": 200,
    },
    ("Components", "Indirect"): {
        "sku_count": 10,
        "subcategories": ["Test Fixtures", "Calibration Tools"],
        "base_unit_cost": 120.00,
        "cost_volatility": 0.08,
        "commodity_sensitivity": 0.20,
        "labor_intensity": 0.15,
        "overhead_allocation": 0.25,
        "seasonal_amplitude": 0.05,
        "seasonal_direction": "flat",
        "tariff_exposure": 0.10,
        "quality_rejection_rate": 0.005,
        "moq": 50,
    },
    ("Packaging", "Direct"): {
        "sku_count": 20,
        "subcategories": ["Corrugated Boxes", "Blister Packs", "Foam Inserts", "Shrink Wrap"],
        "base_unit_cost": 2.80,
        "cost_volatility": 0.12,
        "commodity_sensitivity": 0.50,
        "labor_intensity": 0.15,
        "overhead_allocation": 0.08,
        "seasonal_amplitude": 0.30,
        "seasonal_direction": "q4_peak",
        "tariff_exposure": 0.08,
        "quality_rejection_rate": 0.015,
        "moq": 5000,
    },
    ("Packaging", "Indirect"): {
        "sku_count": 10,
        "subcategories": ["Labels", "Desiccants", "Pallet Wrap"],
        "base_unit_cost": 0.50,
        "cost_volatility": 0.06,
        "commodity_sensitivity": 0.20,
        "labor_intensity": 0.05,
        "overhead_allocation": 0.10,
        "seasonal_amplitude": 0.10,
        "seasonal_direction": "flat",
        "tariff_exposure": 0.03,
        "quality_rejection_rate": 0.005,
        "moq": 10000,
    },
    ("Labor", "Direct"): {
        "sku_count": 20,
        "subcategories": ["Assembly Line", "Welding", "Quality Inspection", "Machine Operation"],
        "base_unit_cost": 28.00,
        "cost_volatility": 0.08,
        "commodity_sensitivity": 0.05,
        "labor_intensity": 0.90,
        "overhead_allocation": 0.15,
        "seasonal_amplitude": 0.10,
        "seasonal_direction": "q4_peak",
        "tariff_exposure": 0.00,
        "quality_rejection_rate": 0.04,
        "moq": 1,
    },
    ("Labor", "Indirect"): {
        "sku_count": 15,
        "subcategories": ["Maintenance", "Logistics", "Supervision"],
        "base_unit_cost": 35.00,
        "cost_volatility": 0.05,
        "commodity_sensitivity": 0.02,
        "labor_intensity": 0.85,
        "overhead_allocation": 0.30,
        "seasonal_amplitude": 0.05,
        "seasonal_direction": "flat",
        "tariff_exposure": 0.00,
        "quality_rejection_rate": 0.01,
        "moq": 1,
    },
    ("Overhead", "Direct"): {
        "sku_count": 15,
        "subcategories": ["Equipment Depreciation", "Tooling", "Energy Consumption"],
        "base_unit_cost": 8.00,
        "cost_volatility": 0.15,
        "commodity_sensitivity": 0.40,
        "labor_intensity": 0.10,
        "overhead_allocation": 0.70,
        "seasonal_amplitude": 0.20,
        "seasonal_direction": "q1_peak",
        "tariff_exposure": 0.05,
        "quality_rejection_rate": 0.00,
        "moq": 1,
    },
    ("Overhead", "Indirect"): {
        "sku_count": 15,
        "subcategories": ["Rent Allocation", "IT Systems", "Insurance"],
        "base_unit_cost": 15.00,
        "cost_volatility": 0.04,
        "commodity_sensitivity": 0.10,
        "labor_intensity": 0.05,
        "overhead_allocation": 0.85,
        "seasonal_amplitude": 0.03,
        "seasonal_direction": "flat",
        "tariff_exposure": 0.02,
        "quality_rejection_rate": 0.00,
        "moq": 1,
    },
}

# ── Direction → phase mapping ────────────────────────────────────────────

DIRECTION_PHASE = {
    "q1_peak": -np.pi / 2,   # Peak in Q1 (January)
    "q3_peak": np.pi / 2,    # Peak in Q3 (July)
    "q4_peak": np.pi,         # Peak in Q4 (October)
    "flat": 0.0,
}

# ── Commodity price parameters ─────────────────────────────────────────

COMMODITY_VOLATILITY = {
    "Steel": 0.15,
    "Copper": 0.25,
    "Resin": 0.20,
    "Aluminum": 0.18,
    "Silicon": 0.22,
}

COMMODITY_SEASONAL_PHASE = {
    "Steel": 0.0,
    "Copper": np.pi / 3,
    "Resin": -np.pi / 4,
    "Aluminum": np.pi / 6,
    "Silicon": np.pi / 2,
}

# ── Plant (Manufacturing) definitions ─────────────────────────────────

PLANT_DEFINITIONS = {
    "PLT_Shenzhen":   {"country": "CN", "region": "Shenzhen",   "lat": 22.543, "lon": 114.058, "labor_rate": 8.50,  "overhead_rate": 0.25, "capacity_util": 0.88},
    "PLT_Shanghai":   {"country": "CN", "region": "Shanghai",   "lat": 31.230, "lon": 121.474, "labor_rate": 10.00, "overhead_rate": 0.28, "capacity_util": 0.85},
    "PLT_Taipei":     {"country": "TW", "region": "Taipei",     "lat": 25.033, "lon": 121.565, "labor_rate": 14.00, "overhead_rate": 0.30, "capacity_util": 0.82},
    "PLT_Munich":     {"country": "DE", "region": "Munich",     "lat": 48.137, "lon": 11.576,  "labor_rate": 32.00, "overhead_rate": 0.40, "capacity_util": 0.78},
    "PLT_Stuttgart":  {"country": "DE", "region": "Stuttgart",  "lat": 48.776, "lon": 9.183,   "labor_rate": 30.00, "overhead_rate": 0.38, "capacity_util": 0.80},
    "PLT_Detroit":    {"country": "US", "region": "Detroit",    "lat": 42.331, "lon": -83.046, "labor_rate": 25.00, "overhead_rate": 0.35, "capacity_util": 0.75},
    "PLT_Austin":     {"country": "US", "region": "Austin",     "lat": 30.267, "lon": -97.743, "labor_rate": 28.00, "overhead_rate": 0.32, "capacity_util": 0.83},
    "PLT_Guadalajara":{"country": "MX", "region": "Guadalajara","lat": 20.659, "lon": -103.349,"labor_rate": 6.50,  "overhead_rate": 0.22, "capacity_util": 0.90},
    "PLT_Pune":       {"country": "IN", "region": "Pune",       "lat": 18.520, "lon": 73.857,  "labor_rate": 4.50,  "overhead_rate": 0.20, "capacity_util": 0.87},
    "PLT_Chennai":    {"country": "IN", "region": "Chennai",    "lat": 13.083, "lon": 80.270,  "labor_rate": 4.00,  "overhead_rate": 0.18, "capacity_util": 0.91},
    "PLT_Tokyo":      {"country": "JP", "region": "Tokyo",      "lat": 35.682, "lon": 139.759, "labor_rate": 22.00, "overhead_rate": 0.35, "capacity_util": 0.79},
    "PLT_Osaka":      {"country": "JP", "region": "Osaka",      "lat": 34.694, "lon": 135.502, "labor_rate": 20.00, "overhead_rate": 0.33, "capacity_util": 0.81},
}

PLANT_WEIGHTS = {
    "PLT_Shenzhen": 0.15, "PLT_Shanghai": 0.10, "PLT_Taipei": 0.08,
    "PLT_Munich": 0.06, "PLT_Stuttgart": 0.05, "PLT_Detroit": 0.07, "PLT_Austin": 0.08,
    "PLT_Guadalajara": 0.10, "PLT_Pune": 0.10, "PLT_Chennai": 0.08,
    "PLT_Tokyo": 0.07, "PLT_Osaka": 0.06,
}

# ── Cost reduction action types ─────────────────────────────────────────

COST_REDUCTION_ACTIONS = [
    "supplier_switch",
    "design_change",
    "volume_consolidation",
    "process_optimization",
    "material_substitution",
    "nearshoring",
    "automation",
    "negotiate_contract",
]

# ── Make-vs-Buy thresholds ──────────────────────────────────────────────

MAKE_VS_BUY_FACTORS = {
    "volume_breakeven_multiplier": 1.15,   # Make is cheaper above this volume ratio
    "quality_weight": 0.30,
    "lead_time_weight": 0.20,
    "cost_weight": 0.35,
    "strategic_weight": 0.15,
}
