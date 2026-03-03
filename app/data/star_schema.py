"""Pandera schemas for GlowCast star schema (9 tables).

Enforces data contracts on all dimension and fact tables:
    Dim_Product, Dim_Location, Dim_Weather, Dim_Customer,
    Fact_Sales, Fact_Inventory_Batch, Fact_Social_Signals,
    Fact_Order_Fulfillment, Fact_Review_Aspects
"""

from __future__ import annotations

from pandera import Check, Column, DataFrameSchema

from app.data.segment_genes import BRANDS, CONCERNS, PRICE_TIERS, TEXTURES

# ── Dimension schemas ────────────────────────────────────────────────────

Dim_Product = DataFrameSchema(
    {
        "sku_id": Column(str, Check.str_startswith("SKU_"), unique=True),
        "product_name": Column(str, nullable=False),
        "brand": Column(str, Check.isin(BRANDS)),
        "concern": Column(str, Check.isin(CONCERNS)),
        "texture": Column(str, Check.isin(TEXTURES)),
        "price_tier": Column(str, Check.isin(PRICE_TIERS)),
        "subcategory": Column(str, nullable=False),
        "unit_cost": Column(float, Check.gt(0)),
        "retail_price": Column(float, Check.gt(0)),
        "gross_margin": Column(float, Check.in_range(0, 1)),
        "shelf_life_days": Column(int, Check.gt(0)),
        "price_elasticity": Column(float, Check.lt(0)),
        "social_sensitivity": Column(float, Check.in_range(0, 1)),
        "seasonal_amplitude": Column(float, Check.ge(0)),
        "seasonal_direction": Column(str, Check.isin(["summer", "winter", "spring", "neutral"])),
        "base_return_rate": Column(float, Check.in_range(0, 0.1)),
    },
    name="Dim_Product",
    coerce=True,
)

Dim_Location = DataFrameSchema(
    {
        "fc_id": Column(str, Check.str_startswith("FC_"), unique=True),
        "country": Column(str, Check.isin(["US", "DE", "UK", "JP", "IN"])),
        "region": Column(str, nullable=False),
        "lat": Column(float, Check.in_range(-90, 90)),
        "lon": Column(float, Check.in_range(-180, 180)),
        "climate_zone": Column(str, nullable=False),
        "storage_capacity": Column(int, Check.gt(0)),
        "avg_ship_days": Column(float, Check.gt(0)),
    },
    name="Dim_Location",
    coerce=True,
)

Dim_Weather = DataFrameSchema(
    {
        "weather_id": Column(int, unique=True),
        "date": Column("datetime64[ns]"),
        "region": Column(str, nullable=False),
        "temperature_celsius": Column(float),
        "humidity_pct": Column(float, Check.in_range(10, 100)),
        "temp_humidity_index": Column(float),
        "season": Column(str, Check.isin(["Spring", "Summer", "Autumn", "Winter"])),
    },
    name="Dim_Weather",
    coerce=True,
)

Dim_Customer = DataFrameSchema(
    {
        "customer_id": Column(str, Check.str_startswith("CUST_"), unique=True),
        "age_group": Column(str, Check.isin(["18-24", "25-34", "35-44", "45+"])),
        "skin_type": Column(str, Check.isin(["Oily_Skin", "Dry_Skin", "Combo_Skin", "Mature_Skin"])),
        "region": Column(str, nullable=False),
        "acquisition_channel": Column(str, Check.isin(["Online", "Retail", "Marketplace"])),
        "first_purchase_date": Column("datetime64[ns]"),
    },
    name="Dim_Customer",
    coerce=True,
)

# ── Fact schemas ─────────────────────────────────────────────────────────

Fact_Sales = DataFrameSchema(
    {
        "order_id": Column(str, Check.str_startswith("ORD_")),
        "order_date": Column("datetime64[ns]"),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "fc_id": Column(str, Check.str_startswith("FC_")),
        "customer_id": Column(str, Check.str_startswith("CUST_")),
        "units_sold": Column(int, Check.ge(0)),
        "discount_rate": Column(float, Check.in_range(0, 1)),
        "revenue": Column(float, Check.ge(0)),
        "is_return": Column(int, Check.isin([0, 1])),
        "channel": Column(str, Check.isin(["Online", "Retail", "Marketplace"])),
        "skin_type": Column(str, Check.isin(["Oily_Skin", "Dry_Skin", "Combo_Skin", "Mature_Skin"])),
        "age_group": Column(str, Check.isin(["18-24", "25-34", "35-44", "45+"])),
    },
    name="Fact_Sales",
    coerce=True,
)

Fact_Inventory_Batch = DataFrameSchema(
    {
        "snapshot_date": Column("datetime64[ns]"),
        "fc_id": Column(str, Check.str_startswith("FC_")),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "batch_id": Column(str, nullable=False),
        "manufacturing_date": Column("datetime64[ns]"),
        "expiry_date": Column("datetime64[ns]"),
        "units_on_hand": Column(int, Check.gt(0)),
    },
    name="Fact_Inventory_Batch",
    coerce=True,
)

Fact_Social_Signals = DataFrameSchema(
    {
        "signal_id": Column(int),
        "signal_date": Column("datetime64[ns]"),
        "concern": Column(str, Check.isin(CONCERNS)),
        "source": Column(str, nullable=False),
        "mention_volume": Column(int, Check.ge(0)),
        "sentiment_score": Column(float, Check.in_range(-1, 1)),
        "is_viral": Column(int, Check.isin([0, 1])),
        "net_momentum": Column(float),
    },
    name="Fact_Social_Signals",
    coerce=True,
)

Fact_Order_Fulfillment = DataFrameSchema(
    {
        "fulfillment_id": Column(str, nullable=False),
        "order_id": Column(str, Check.str_startswith("ORD_")),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "customer_region": Column(str, nullable=False),
        "fulfilled_from_fc": Column(str, Check.str_startswith("FC_")),
        "units_sold": Column(int, Check.ge(0)),
        "fulfillment_type": Column(str, Check.isin(["Local_Fulfillment", "Cross_Zone_Fulfillment"])),
        "shipping_cost": Column(float, Check.ge(0)),
        "delivery_days": Column(int, Check.gt(0)),
    },
    name="Fact_Order_Fulfillment",
    coerce=True,
)

Fact_Review_Aspects = DataFrameSchema(
    {
        "review_id": Column(str, nullable=False),
        "sku_id": Column(str, Check.str_startswith("SKU_")),
        "review_date": Column("datetime64[ns]"),
        "star_rating": Column(int, Check.in_range(1, 5)),
        "aspect": Column(str, Check.isin([
            "texture", "scent", "packaging", "efficacy",
            "ingredients", "skin_reaction", "value", "shelf_life",
        ])),
        "aspect_sentiment": Column(float, Check.in_range(-1, 1)),
    },
    name="Fact_Review_Aspects",
    coerce=True,
)
