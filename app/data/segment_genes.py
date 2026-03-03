"""Product taxonomy and segment gene definitions for GlowCast.

Domain model: Concern × Texture → 10 demand-behavior segments.
Each segment encodes supply chain genetics: demand shape, seasonality,
social sensitivity, shelf-life constraints, and replenishment parameters.

References:
    - v5.0 domain design (Section 3.3)
    - beauty_product_archetype_research.md (price research)
"""

from __future__ import annotations

import numpy as np

# ── Primary axes ──────────────────────────────────────────────────────────

CONCERNS = ["AntiAging", "Acne", "Hydrating", "Brightening", "SunProtection"]
TEXTURES = ["Lightweight", "Rich"]

# ── Price tiers ───────────────────────────────────────────────────────────

PRICE_TIERS = ["Mass", "Prestige", "Luxury"]

PRICE_TIER_PARAMS = {
    "Mass":     {"range": (5, 25),   "cogs_pct": (0.30, 0.40), "margin": (0.50, 0.60)},
    "Prestige": {"range": (25, 80),  "cogs_pct": (0.20, 0.30), "margin": (0.65, 0.75)},
    "Luxury":   {"range": (80, 280), "cogs_pct": (0.10, 0.15), "margin": (0.74, 0.82)},
}

# ── Brands ────────────────────────────────────────────────────────────────

BRANDS = ["LuxeVita", "SolGuard", "GlowRush", "PureBasics", "VelvetAura"]

BRAND_CONCERN_WEIGHTS = {
    "LuxeVita":   {"AntiAging": 0.50, "Hydrating": 0.20, "Brightening": 0.15, "SunProtection": 0.10, "Acne": 0.05},
    "SolGuard":   {"SunProtection": 0.45, "Acne": 0.25, "Hydrating": 0.15, "Brightening": 0.10, "AntiAging": 0.05},
    "GlowRush":   {"Acne": 0.30, "Brightening": 0.30, "Hydrating": 0.20, "SunProtection": 0.15, "AntiAging": 0.05},
    "PureBasics": {"Hydrating": 0.45, "Acne": 0.20, "SunProtection": 0.20, "Brightening": 0.10, "AntiAging": 0.05},
    "VelvetAura": {"AntiAging": 0.35, "Brightening": 0.25, "Hydrating": 0.20, "SunProtection": 0.10, "Acne": 0.10},
}

# ── Segment Genes (10 segments) ──────────────────────────────────────────

SEGMENT_GENES = {
    ("AntiAging", "Lightweight"): {
        "sku_count": 15,
        "subcategories": ["Firming Gel Serum", "Light Repair Essence", "Anti-Wrinkle Ampoule"],
        "base_demand_lambda": 6,
        "price_elasticity": -0.5,
        "seasonal_amplitude": 0.05,
        "seasonal_direction": "neutral",
        "social_sensitivity": 0.15,
        "return_rate": 0.025,
        "shelf_life_days": 730,
        "replenish_cycle": (60, 90),
        "trend_annual_pct": 0.03,
    },
    ("AntiAging", "Rich"): {
        "sku_count": 25,
        "subcategories": ["Intensive Night Cream", "Rejuvenating Eye Balm", "Ultra Repair Concentrate"],
        "base_demand_lambda": 4,
        "price_elasticity": -0.4,
        "seasonal_amplitude": 0.10,
        "seasonal_direction": "winter",
        "social_sensitivity": 0.08,
        "return_rate": 0.020,
        "shelf_life_days": 730,
        "replenish_cycle": (90, 120),
        "trend_annual_pct": 0.03,
    },
    ("Acne", "Lightweight"): {
        "sku_count": 25,
        "subcategories": ["Oil Control Gel", "Salicylic Acid Toner", "Clear Skin Serum", "Mattifying Fluid"],
        "base_demand_lambda": 14,
        "price_elasticity": -2.0,
        "seasonal_amplitude": 0.30,
        "seasonal_direction": "summer",
        "social_sensitivity": 0.70,
        "return_rate": 0.035,
        "shelf_life_days": 540,
        "replenish_cycle": (45, 75),
        "trend_annual_pct": 0.02,
    },
    ("Acne", "Rich"): {
        "sku_count": 10,
        "subcategories": ["Post-Acne Repair Cream", "Gentle Cleansing Balm"],
        "base_demand_lambda": 8,
        "price_elasticity": -1.5,
        "seasonal_amplitude": 0.15,
        "seasonal_direction": "neutral",
        "social_sensitivity": 0.40,
        "return_rate": 0.030,
        "shelf_life_days": 730,
        "replenish_cycle": (60, 90),
        "trend_annual_pct": 0.01,
    },
    ("Hydrating", "Lightweight"): {
        "sku_count": 20,
        "subcategories": ["Water Gel Moisturizer", "Hydra Mist Spray", "Cooling Gel Mask", "Aqua Toner"],
        "base_demand_lambda": 16,
        "price_elasticity": -1.5,
        "seasonal_amplitude": 0.40,
        "seasonal_direction": "summer",
        "social_sensitivity": 0.25,
        "return_rate": 0.020,
        "shelf_life_days": 730,
        "replenish_cycle": (45, 75),
        "trend_annual_pct": 0.04,
    },
    ("Hydrating", "Rich"): {
        "sku_count": 25,
        "subcategories": ["Barrier Repair Cream", "Rich Night Balm", "Hand & Lip Treatment", "Sleeping Mask"],
        "base_demand_lambda": 15,
        "price_elasticity": -1.2,
        "seasonal_amplitude": 0.45,
        "seasonal_direction": "winter",
        "social_sensitivity": 0.15,
        "return_rate": 0.018,
        "shelf_life_days": 910,
        "replenish_cycle": (60, 90),
        "trend_annual_pct": 0.02,
    },
    ("Brightening", "Lightweight"): {
        "sku_count": 15,
        "subcategories": ["Vitamin C Serum", "Brightening Essence Water", "Dark Spot Ampoule"],
        "base_demand_lambda": 10,
        "price_elasticity": -0.9,
        "seasonal_amplitude": 0.30,
        "seasonal_direction": "spring",
        "social_sensitivity": 0.50,
        "return_rate": 0.035,
        "shelf_life_days": 450,
        "replenish_cycle": (60, 90),
        "trend_annual_pct": 0.06,
    },
    ("Brightening", "Rich"): {
        "sku_count": 15,
        "subcategories": ["Whitening Night Cream", "Radiance Eye Cream", "Glow Repair Mask"],
        "base_demand_lambda": 7,
        "price_elasticity": -0.7,
        "seasonal_amplitude": 0.15,
        "seasonal_direction": "neutral",
        "social_sensitivity": 0.30,
        "return_rate": 0.030,
        "shelf_life_days": 730,
        "replenish_cycle": (75, 120),
        "trend_annual_pct": 0.04,
    },
    ("SunProtection", "Lightweight"): {
        "sku_count": 20,
        "subcategories": ["UV Aqua Gel SPF50", "Sport Sunscreen Spray", "Mineral Sun Fluid", "Daily UV Shield"],
        "base_demand_lambda": 12,
        "price_elasticity": -0.6,
        "seasonal_amplitude": 0.90,
        "seasonal_direction": "summer",
        "social_sensitivity": 0.65,
        "return_rate": 0.025,
        "shelf_life_days": 540,
        "replenish_cycle": (30, 45),
        "trend_annual_pct": 0.05,
    },
    ("SunProtection", "Rich"): {
        "sku_count": 5,
        "subcategories": ["Moisturizing Sun Cream SPF30"],
        "base_demand_lambda": 5,
        "price_elasticity": -0.5,
        "seasonal_amplitude": 0.60,
        "seasonal_direction": "summer",
        "social_sensitivity": 0.30,
        "return_rate": 0.020,
        "shelf_life_days": 540,
        "replenish_cycle": (45, 60),
        "trend_annual_pct": 0.03,
    },
}

# ── Direction → phase mapping ────────────────────────────────────────────

DIRECTION_PHASE = {
    "summer": 0.0,
    "winter": np.pi,
    "spring": -np.pi / 2,
    "neutral": 0.0,
}

# ── Social signal parameters ─────────────────────────────────────────────

CONCERN_PHASE = {
    "AntiAging":      np.pi / 2,
    "Acne":          -np.pi / 4,
    "Hydrating":      0.0,
    "Brightening":    np.pi / 3,
    "SunProtection": -np.pi / 3,
}

CONCERN_BASE_VOL = {
    "AntiAging": 400,
    "Acne": 700,
    "Hydrating": 600,
    "Brightening": 500,
    "SunProtection": 650,
}

GLOBAL_SOURCES = ["Reddit", "TikTok", "Amazon_Reviews", "YouTube", "@cosme", "Sephora_Reviews", "Instagram"]
SOURCE_WEIGHTS = [0.20, 0.25, 0.20, 0.10, 0.10, 0.08, 0.07]

# ── NLP aspect taxonomy ──────────────────────────────────────────────────

BEAUTY_ASPECTS = {
    "texture":       ["texture", "consistency", "smooth", "thick", "thin",
                      "creamy", "watery", "gel", "lightweight", "sticky"],
    "scent":         ["smell", "scent", "fragrance", "odor", "perfume",
                      "aroma", "unscented", "stinky"],
    "packaging":     ["bottle", "pump", "tube", "packaging", "container",
                      "cap", "dispenser", "leak", "broken"],
    "efficacy":      ["effective", "works", "results", "improvement",
                      "difference", "before and after", "useless"],
    "ingredients":   ["ingredients", "chemical", "natural", "organic",
                      "paraben", "sulfate", "retinol", "hyaluronic"],
    "skin_reaction": ["breakout", "irritation", "rash", "allergy",
                      "burning", "redness", "sensitive", "acne"],
    "value":         ["price", "expensive", "cheap", "worth", "value",
                      "affordable", "overpriced"],
    "shelf_life":    ["expired", "shelf life", "spoil", "old", "fresh",
                      "expiration", "rancid"],
}

# ── Fulfillment Center definitions ───────────────────────────────────────

FC_DEFINITIONS = {
    "FC_Phoenix":    {"country": "US", "region": "Phoenix",    "lat": 33.448, "lon": -112.074,
                      "climate_zone": "Hot_Dry"},
    "FC_Miami":      {"country": "US", "region": "Miami",      "lat": 25.762, "lon": -80.192,
                      "climate_zone": "Hot_Humid"},
    "FC_Seattle":    {"country": "US", "region": "Seattle",    "lat": 47.606, "lon": -122.332,
                      "climate_zone": "Cool_Damp"},
    "FC_Dallas":     {"country": "US", "region": "Dallas",     "lat": 32.777, "lon": -96.797,
                      "climate_zone": "Humid_Subtropical"},
    "FC_Berlin":     {"country": "DE", "region": "Berlin",     "lat": 52.520, "lon": 13.405,
                      "climate_zone": "Cold_Continental"},
    "FC_London":     {"country": "UK", "region": "London",     "lat": 51.507, "lon": -0.128,
                      "climate_zone": "Mild_Oceanic"},
    "FC_Manchester": {"country": "UK", "region": "Manchester", "lat": 53.483, "lon": -2.244,
                      "climate_zone": "Cold_Damp"},
    "FC_Tokyo":      {"country": "JP", "region": "Tokyo",      "lat": 35.682, "lon": 139.759,
                      "climate_zone": "Humid_Subtropical"},
    "FC_Osaka":      {"country": "JP", "region": "Osaka",      "lat": 34.694, "lon": 135.502,
                      "climate_zone": "Humid_Subtropical"},
    "FC_Mumbai":     {"country": "IN", "region": "Mumbai",     "lat": 19.076, "lon": 72.878,
                      "climate_zone": "Tropical_Monsoon"},
    "FC_Delhi":      {"country": "IN", "region": "Delhi",      "lat": 28.614, "lon": 77.209,
                      "climate_zone": "Extreme_Swing"},
    "FC_Bangalore":  {"country": "IN", "region": "Bangalore",  "lat": 12.972, "lon": 77.594,
                      "climate_zone": "Tropical_Plateau"},
}

FC_WEIGHTS = {
    "FC_Phoenix": 0.12, "FC_Miami": 0.10, "FC_Seattle": 0.10, "FC_Dallas": 0.08,
    "FC_Berlin": 0.08, "FC_London": 0.06, "FC_Manchester": 0.04,
    "FC_Tokyo": 0.10, "FC_Osaka": 0.07,
    "FC_Mumbai": 0.10, "FC_Delhi": 0.08, "FC_Bangalore": 0.07,
}

# ── FC climate parameters ────────────────────────────────────────────────
# Format: (base_temp_C, temp_amplitude, base_humidity_%, humidity_amplitude, temp_noise, humidity_noise)

FC_CLIMATE_PARAMS = {
    "Phoenix":    (25, 16, 30, 8,  2.0, 4),
    "Miami":      (26, 7,  75, 8,  1.5, 5),
    "Seattle":    (12, 9,  76, 6,  1.5, 4),
    "Dallas":     (20, 14, 65, 10, 2.0, 5),
    "Berlin":     (10, 12, 75, 8,  2.5, 5),
    "London":     (11, 8,  78, 5,  1.5, 4),
    "Manchester": (10, 7,  82, 4,  1.5, 3),
    "Tokyo":      (16, 14, 65, 12, 2.0, 5),
    "Osaka":      (17, 13, 65, 10, 1.8, 5),
    "Mumbai":     (28, 4,  75, 12, 1.0, 6),
    "Delhi":      (25, 18, 55, 20, 2.5, 8),
    "Bangalore":  (25, 5,  60, 15, 1.0, 5),
}

# ── Temperature-Humidity Index ────────────────────────────────────────────

TEMPERATURE_SWITCH_POINT = 23.5  # Celsius — above: Lightweight demand up; below: Rich demand up


def compute_thi(temperature: float, humidity: float) -> float:
    """Compute Temperature-Humidity Index (NWS Livestock Weather Safety Index, Celsius)."""
    return temperature - 0.55 * (1 - humidity / 100) * (temperature - 14.5)
