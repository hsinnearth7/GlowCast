"""GlowCast synthetic data generator engine.

Generates a complete star schema dataset for beauty/skincare supply chain simulation:
- 5,000 SKUs across 10 concern×texture segments
- 12 fulfillment centers across 5 countries (US/DE/UK/JP/IN)
- 3 years of daily demand with Negative Binomial distribution
- Social signals with viral bursts and T-3 leading indicator
- Climate-driven seasonality with 23.5°C texture switching
- FIFO inventory batches with shelf-life constraints

Usage:
    python -m app.data.data_generator [--validate-only]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import pandera

from app.data.segment_genes import (
    BEAUTY_ASPECTS,
    BRAND_CONCERN_WEIGHTS,
    BRANDS,
    CONCERN_BASE_VOL,
    CONCERN_PHASE,
    CONCERNS,
    DIRECTION_PHASE,
    FC_CLIMATE_PARAMS,
    FC_DEFINITIONS,
    FC_WEIGHTS,
    GLOBAL_SOURCES,
    PRICE_TIER_PARAMS,
    SEGMENT_GENES,
    SOURCE_WEIGHTS,
    TEMPERATURE_SWITCH_POINT,
)
from app.data.star_schema import (
    Dim_Customer,
    Dim_Location,
    Dim_Product,
    Dim_Weather,
    Fact_Inventory_Batch,
    Fact_Order_Fulfillment,
    Fact_Review_Aspects,
    Fact_Sales,
    Fact_Social_Signals,
)
from app.seed import GLOBAL_SEED


class GlowCastDataGenerator:
    """Generates the complete GlowCast star schema dataset."""

    def __init__(
        self,
        n_skus: int = 5000,
        n_days: int = 1095,
        start_date: str = "2022-01-01",
        seed: int = GLOBAL_SEED,
    ):
        self.n_skus = n_skus
        self.n_days = n_days
        self.start_date = pd.Timestamp(start_date)
        self.rng = np.random.default_rng(seed)
        self.dates = pd.date_range(self.start_date, periods=n_days, freq="D")

        # Generated tables
        self.dim_product: pd.DataFrame | None = None
        self.dim_location: pd.DataFrame | None = None
        self.dim_weather: pd.DataFrame | None = None
        self.dim_customer: pd.DataFrame | None = None
        self.fact_sales: pd.DataFrame | None = None
        self.fact_inventory: pd.DataFrame | None = None
        self.fact_social: pd.DataFrame | None = None
        self.fact_fulfillment: pd.DataFrame | None = None
        self.fact_reviews: pd.DataFrame | None = None

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all star schema tables."""
        self._generate_dim_product()
        self._generate_dim_location()
        self._generate_dim_weather()
        self._generate_fact_social()
        self._generate_fact_sales()
        self._generate_dim_customer()
        self._generate_fact_inventory()
        self._generate_fact_fulfillment()
        self._generate_fact_reviews()

        return {
            "dim_product": self.dim_product,
            "dim_location": self.dim_location,
            "dim_weather": self.dim_weather,
            "dim_customer": self.dim_customer,
            "fact_sales": self.fact_sales,
            "fact_inventory": self.fact_inventory,
            "fact_social": self.fact_social,
            "fact_fulfillment": self.fact_fulfillment,
            "fact_reviews": self.fact_reviews,
        }

    def _generate_dim_product(self) -> None:
        """Generate Dim_Product with proportional segment allocation."""
        total_base = sum(g["sku_count"] for g in SEGMENT_GENES.values())
        records = []
        sku_idx = 1000

        for (concern, texture), genes in SEGMENT_GENES.items():
            seg_skus = max(1, round(genes["sku_count"] / total_base * self.n_skus))

            # Assign brands with concern-weighted probabilities
            brand_weights = np.array([BRAND_CONCERN_WEIGHTS[b][concern] for b in BRANDS])
            brand_weights = brand_weights / brand_weights.sum()

            for _ in range(seg_skus):
                brand = self.rng.choice(BRANDS, p=brand_weights)
                subcat = self.rng.choice(genes["subcategories"])
                tier = self.rng.choice(
                    ["Mass", "Prestige", "Luxury"], p=[0.40, 0.40, 0.20]
                )
                tp = PRICE_TIER_PARAMS[tier]
                retail_price = round(self.rng.uniform(*tp["range"]), 2)
                cogs_pct = self.rng.uniform(*tp["cogs_pct"])
                unit_cost = round(retail_price * cogs_pct, 2)
                gross_margin = round((retail_price - unit_cost) / retail_price, 4)

                records.append({
                    "sku_id": f"SKU_{sku_idx}",
                    "product_name": f"{brand} {subcat}",
                    "brand": brand,
                    "concern": concern,
                    "texture": texture,
                    "price_tier": tier,
                    "subcategory": subcat,
                    "unit_cost": unit_cost,
                    "retail_price": retail_price,
                    "gross_margin": gross_margin,
                    "shelf_life_days": genes["shelf_life_days"],
                    "price_elasticity": genes["price_elasticity"],
                    "social_sensitivity": genes["social_sensitivity"],
                    "seasonal_amplitude": genes["seasonal_amplitude"],
                    "seasonal_direction": genes["seasonal_direction"],
                    "base_return_rate": genes["return_rate"],
                })
                sku_idx += 1

        self.dim_product = pd.DataFrame(records[: self.n_skus])

    def _generate_dim_location(self) -> None:
        """Generate Dim_Location from FC definitions."""
        records = []
        for fc_id, fc in FC_DEFINITIONS.items():
            records.append({
                "fc_id": fc_id,
                "country": fc["country"],
                "region": fc["region"],
                "lat": fc["lat"],
                "lon": fc["lon"],
                "climate_zone": fc["climate_zone"],
                "storage_capacity": int(self.rng.integers(80000, 200001)),
                "avg_ship_days": round(self.rng.uniform(1.0, 5.0), 1),
            })
        self.dim_location = pd.DataFrame(records)

    def _generate_dim_weather(self) -> None:
        """Generate daily weather per region using sine-wave climate models."""
        records = []
        wid = 0
        for region, (t_base, t_amp, h_base, h_amp, t_noise, h_noise) in FC_CLIMATE_PARAMS.items():
            for date in self.dates:
                doy = date.day_of_year
                phase = (doy - 100) / 365 * 2 * np.pi

                # Southern hemisphere regions would need phase flip;
                # all GlowCast FCs are in northern hemisphere
                temp = t_base + t_amp * np.sin(phase) + self.rng.normal(0, t_noise)
                humidity = h_base + h_amp * np.sin(phase) + self.rng.normal(0, h_noise)
                humidity = float(np.clip(humidity, 10, 100))
                thi = temp - 0.55 * (1 - humidity / 100) * (temp - 14.5)

                month = date.month
                if month in (3, 4, 5):
                    season = "Spring"
                elif month in (6, 7, 8):
                    season = "Summer"
                elif month in (9, 10, 11):
                    season = "Autumn"
                else:
                    season = "Winter"

                records.append({
                    "weather_id": wid,
                    "date": date,
                    "region": region,
                    "temperature_celsius": round(temp, 1),
                    "humidity_pct": round(humidity, 1),
                    "temp_humidity_index": round(thi, 1),
                    "season": season,
                })
                wid += 1

        self.dim_weather = pd.DataFrame(records)

    def _generate_fact_social(self) -> None:
        """Generate social signal time series per concern."""
        records = []
        sid = 0
        for concern in CONCERNS:
            base_vol = CONCERN_BASE_VOL[concern]
            phase = CONCERN_PHASE[concern]

            for date in self.dates:
                doy = date.day_of_year
                seasonal_vol = base_vol * (1 + 0.3 * np.sin(doy / 365 * 2 * np.pi + phase))
                is_viral = int(self.rng.random() < 0.03)

                if is_viral:
                    volume = int(seasonal_vol + self.rng.integers(3000, 10001))
                    sentiment = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.5, 0.95))
                else:
                    volume = max(0, int(seasonal_vol + self.rng.normal(0, base_vol * 0.1)))
                    sentiment = round(float(self.rng.uniform(-0.2, 0.5)), 3)

                source = self.rng.choice(GLOBAL_SOURCES, p=SOURCE_WEIGHTS)
                momentum = volume * sentiment

                records.append({
                    "signal_id": sid,
                    "signal_date": date,
                    "concern": concern,
                    "source": source,
                    "mention_volume": volume,
                    "sentiment_score": round(sentiment, 3),
                    "is_viral": is_viral,
                    "net_momentum": round(momentum, 2),
                })
                sid += 1

        self.fact_social = pd.DataFrame(records)

    def _generate_fact_sales(self) -> None:
        """Generate daily sales with Negative Binomial demand.

        Demand model:
            base (NegBin from segment lambda)
            × seasonal (direction-aware sinusoidal)
            × temperature-texture interaction (23.5°C switch)
            × social lead (T-3 momentum)
            × price elasticity (discount effect)
            + viral bursts
            × trend (annual growth)
        """
        if self.dim_product is None or self.dim_weather is None or self.fact_social is None:
            raise RuntimeError("Must generate dim_product, dim_weather, fact_social first")

        # Pre-compute social momentum lookup: concern → date → momentum
        social_lookup: dict[str, dict[pd.Timestamp, float]] = {}
        for concern in CONCERNS:
            mask = self.fact_social["concern"] == concern
            concern_social = self.fact_social[mask].set_index("signal_date")
            social_lookup[concern] = concern_social["net_momentum"].to_dict()

        # Pre-compute weather lookup: region → date → temp
        weather_lookup: dict[str, dict[pd.Timestamp, float]] = {}
        for region in FC_CLIMATE_PARAMS:
            mask = self.dim_weather["region"] == region
            rw = self.dim_weather[mask].set_index("date")
            weather_lookup[region] = rw["temperature_celsius"].to_dict()

        fc_ids = list(FC_DEFINITIONS.keys())
        fc_weights = np.array([FC_WEIGHTS[fc] for fc in fc_ids])

        # Mark 30% of SKUs as intermittent
        n_intermittent = int(self.n_skus * 0.30)
        intermittent_skus = set(
            self.rng.choice(self.dim_product["sku_id"].values, size=n_intermittent, replace=False)
        )

        records = []
        order_idx = 0

        discount_rates = [1.0, 0.85, 0.75, 0.60]
        discount_probs = [0.65, 0.20, 0.10, 0.05]
        channels = ["Online", "Retail", "Marketplace"]
        channel_probs = [0.50, 0.30, 0.20]
        skin_types = ["Oily_Skin", "Dry_Skin", "Combo_Skin", "Mature_Skin"]
        skin_probs = [0.30, 0.25, 0.30, 0.15]
        age_groups = ["18-24", "25-34", "35-44", "45+"]
        age_probs = [0.25, 0.35, 0.25, 0.15]

        # Sample a subset of dates per SKU to keep row count manageable
        # Average ~100 sales records per SKU → ~500K rows total
        for _, prod in self.dim_product.iterrows():
            sku_id = prod["sku_id"]
            concern = prod["concern"]
            texture = prod["texture"]
            seg_key = (concern, texture)
            genes = SEGMENT_GENES[seg_key]
            base_lambda = genes["base_demand_lambda"]
            amp = genes["seasonal_amplitude"]
            direction = genes["seasonal_direction"]
            sensitivity = genes["social_sensitivity"]
            elasticity = genes["price_elasticity"]
            trend_pct = genes["trend_annual_pct"]
            return_rate = genes["return_rate"]
            retail_price = prod["retail_price"]
            phase = DIRECTION_PHASE[direction]
            is_intermittent = sku_id in intermittent_skus

            # For intermittent SKUs, only generate demand on ~40% of days
            if is_intermittent:
                active_mask = self.rng.random(self.n_days) < 0.40
            else:
                active_mask = np.ones(self.n_days, dtype=bool)

            # Assign to FCs proportionally
            sku_fcs = self.rng.choice(fc_ids, size=min(4, len(fc_ids)), replace=False, p=fc_weights)

            for fc_id in sku_fcs:
                region = FC_DEFINITIONS[fc_id]["region"]

                for day_idx, date in enumerate(self.dates):
                    if not active_mask[day_idx]:
                        continue

                    # Seasonal component
                    doy = date.day_of_year
                    seasonal = 1 + amp * np.sin(doy / 365 * 2 * np.pi + phase)

                    # Temperature-texture interaction
                    temp = weather_lookup.get(region, {}).get(date, 20.0)
                    if texture == "Lightweight":
                        temp_effect = 1.0 + 0.15 * max(0, (temp - TEMPERATURE_SWITCH_POINT) / 10)
                    else:
                        temp_effect = 1.0 + 0.15 * max(0, (TEMPERATURE_SWITCH_POINT - temp) / 10)

                    # Social lead (T-3)
                    lead_date = date - timedelta(days=3)
                    momentum = social_lookup.get(concern, {}).get(lead_date, 0)
                    social_effect = 1.0 + sensitivity * np.clip(momentum / max(abs(CONCERN_BASE_VOL[concern] * 0.3), 1e-6), -1, 3)

                    # Trend
                    year_frac = day_idx / 365
                    trend = 1.0 + trend_pct * year_frac

                    # Discount / price elasticity
                    discount = float(self.rng.choice(discount_rates, p=discount_probs))
                    price_effect = 1.0 + elasticity * (discount - 1.0) if discount < 1.0 else 1.0

                    # Final lambda
                    lam = max(0.1, base_lambda * seasonal * temp_effect * social_effect * trend * price_effect)

                    # Negative Binomial draw (NegBin parameterized as n, p)
                    n_param = max(1, int(lam))
                    p_param = n_param / (n_param + lam) if lam > 0 else 0.5
                    units = int(self.rng.negative_binomial(n_param, max(0.01, min(0.99, p_param))))

                    if units <= 0:
                        continue

                    is_return = int(self.rng.random() < return_rate)
                    revenue = round(units * retail_price * discount * (1 - is_return), 2)

                    records.append({
                        "order_id": f"ORD_{order_idx:08d}",
                        "order_date": date,
                        "sku_id": sku_id,
                        "fc_id": fc_id,
                        "customer_id": f"CUST_{self.rng.integers(10000, 99999):05d}",
                        "units_sold": units,
                        "discount_rate": discount,
                        "revenue": revenue,
                        "is_return": is_return,
                        "channel": self.rng.choice(channels, p=channel_probs),
                        "skin_type": self.rng.choice(skin_types, p=skin_probs),
                        "age_group": self.rng.choice(age_groups, p=age_probs),
                    })
                    order_idx += 1

        self.fact_sales = pd.DataFrame(records)

    def _generate_dim_customer(self) -> None:
        """Generate Dim_Customer from unique customer IDs in sales."""
        if self.fact_sales is None:
            raise RuntimeError("Must generate fact_sales first")

        unique_customers = self.fact_sales["customer_id"].unique()
        records = []
        for cid in unique_customers:
            cust_sales = self.fact_sales[self.fact_sales["customer_id"] == cid]
            first_row = cust_sales.iloc[0]
            records.append({
                "customer_id": cid,
                "age_group": first_row["age_group"],
                "skin_type": first_row["skin_type"],
                "region": FC_DEFINITIONS[first_row["fc_id"]]["region"],
                "acquisition_channel": first_row["channel"],
                "first_purchase_date": cust_sales["order_date"].min(),
            })

        self.dim_customer = pd.DataFrame(records)

    def _generate_fact_inventory(self) -> None:
        """Generate FIFO inventory batches (monthly snapshots)."""
        if self.dim_product is None:
            raise RuntimeError("Must generate dim_product first")

        records = []
        snapshot_dates = pd.date_range(self.start_date, periods=self.n_days // 30, freq="ME")
        fc_ids = list(FC_DEFINITIONS.keys())

        for snap_date in snapshot_dates:
            # Sample a subset of SKUs per snapshot
            sample_size = min(200, len(self.dim_product))
            sampled_skus = self.rng.choice(
                self.dim_product["sku_id"].values, size=sample_size, replace=False
            )

            for sku_id in sampled_skus:
                prod = self.dim_product[self.dim_product["sku_id"] == sku_id].iloc[0]
                shelf_life = int(prod["shelf_life_days"])

                for fc_id in self.rng.choice(fc_ids, size=self.rng.integers(1, 4), replace=False):
                    n_batches = int(self.rng.integers(2, 7))
                    for b in range(n_batches):
                        days_ago = int(self.rng.integers(30, min(700, shelf_life)))
                        mfg_date = snap_date - timedelta(days=days_ago)
                        expiry_date = mfg_date + timedelta(days=shelf_life)

                        records.append({
                            "snapshot_date": snap_date,
                            "fc_id": fc_id,
                            "sku_id": sku_id,
                            "batch_id": f"B-{mfg_date.strftime('%Y%m')}-{b:02d}",
                            "manufacturing_date": mfg_date,
                            "expiry_date": expiry_date,
                            "units_on_hand": int(self.rng.integers(20, 801)),
                        })

        self.fact_inventory = pd.DataFrame(records)

    def _generate_fact_fulfillment(self) -> None:
        """Generate order fulfillment records with cross-zone tracking."""
        if self.fact_sales is None:
            raise RuntimeError("Must generate fact_sales first")

        records = []
        fc_ids = list(FC_DEFINITIONS.keys())

        # Sample a subset of orders for fulfillment tracking
        sample_size = min(100000, len(self.fact_sales))
        sampled = self.fact_sales.sample(n=sample_size, random_state=GLOBAL_SEED)

        for idx, (_, row) in enumerate(sampled.iterrows()):
            order_fc = row["fc_id"]
            customer_region = FC_DEFINITIONS[order_fc]["region"]

            # 85% local fulfillment, 15% cross-zone
            if self.rng.random() < 0.85:
                fulfilled_fc = order_fc
                ftype = "Local_Fulfillment"
                ship_cost = round(self.rng.uniform(3, 12), 2)
                delivery = int(self.rng.integers(1, 4))
            else:
                other_fcs = [f for f in fc_ids if f != order_fc]
                fulfilled_fc = self.rng.choice(other_fcs)
                ftype = "Cross_Zone_Fulfillment"
                ship_cost = round(self.rng.uniform(15, 65), 2)
                delivery = int(self.rng.integers(3, 10))

            records.append({
                "fulfillment_id": f"FUL_{idx:08d}",
                "order_id": row["order_id"],
                "sku_id": row["sku_id"],
                "customer_region": customer_region,
                "fulfilled_from_fc": fulfilled_fc,
                "units_sold": row["units_sold"],
                "fulfillment_type": ftype,
                "shipping_cost": ship_cost,
                "delivery_days": delivery,
            })

        self.fact_fulfillment = pd.DataFrame(records)

    def _generate_fact_reviews(self) -> None:
        """Generate review aspect analysis records."""
        if self.fact_sales is None:
            raise RuntimeError("Must generate fact_sales first")

        records = []
        aspects = list(BEAUTY_ASPECTS.keys())

        # ~5% of orders generate reviews
        review_mask = self.rng.random(len(self.fact_sales)) < 0.05
        reviewed_orders = self.fact_sales[review_mask]

        for idx, (_, row) in enumerate(reviewed_orders.iterrows()):
            n_aspects = int(self.rng.integers(1, 4))
            chosen_aspects = self.rng.choice(aspects, size=n_aspects, replace=False)
            star = int(self.rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.15, 0.32, 0.40]))

            for aspect in chosen_aspects:
                # Sentiment correlates with star rating
                base_sent = (star - 3) / 2
                sentiment = float(np.clip(base_sent + self.rng.normal(0, 0.2), -1, 1))

                records.append({
                    "review_id": f"REV_{idx:06d}_{aspect}",
                    "sku_id": row["sku_id"],
                    "review_date": row["order_date"] + timedelta(days=int(self.rng.integers(1, 30))),
                    "star_rating": star,
                    "aspect": aspect,
                    "aspect_sentiment": round(sentiment, 3),
                })

        self.fact_reviews = pd.DataFrame(records)

    def validate_all(self) -> dict[str, bool]:
        """Validate all tables against Pandera schemas."""
        schemas = {
            "dim_product": (self.dim_product, Dim_Product),
            "dim_location": (self.dim_location, Dim_Location),
            "dim_weather": (self.dim_weather, Dim_Weather),
            "dim_customer": (self.dim_customer, Dim_Customer),
            "fact_sales": (self.fact_sales, Fact_Sales),
            "fact_inventory": (self.fact_inventory, Fact_Inventory_Batch),
            "fact_social": (self.fact_social, Fact_Social_Signals),
            "fact_fulfillment": (self.fact_fulfillment, Fact_Order_Fulfillment),
            "fact_reviews": (self.fact_reviews, Fact_Review_Aspects),
        }

        results = {}
        for name, (df, schema) in schemas.items():
            if df is None:
                results[name] = False
                continue
            try:
                schema.validate(df)
                results[name] = True
            except pandera.errors.SchemaError as e:
                print(f"FAIL {name}: {e}")
                results[name] = False

        return results

    def to_nixtla_format(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert to Nixtla (unique_id, ds, y) format + hierarchy S_df.

        Returns:
            Y_df: DataFrame with columns [unique_id, ds, y]
            S_df: Hierarchy summing matrix
        """
        if self.fact_sales is None or self.dim_product is None:
            raise RuntimeError("Must generate data first")

        # Aggregate daily sales per SKU×FC
        agg = (
            self.fact_sales.groupby(["sku_id", "fc_id", "order_date"])["units_sold"]
            .sum()
            .reset_index()
        )
        agg["unique_id"] = agg["sku_id"] + "__" + agg["fc_id"]
        Y_df = agg.rename(columns={"order_date": "ds", "units_sold": "y"})[["unique_id", "ds", "y"]]

        # Build hierarchy: National → Country → FC → SKU
        hierarchy_records = []
        for uid in Y_df["unique_id"].unique():
            sku_id, fc_id = uid.split("__")
            country = FC_DEFINITIONS[fc_id]["country"]
            hierarchy_records.append({
                "unique_id": uid,
                "sku_id": sku_id,
                "fc_id": fc_id,
                "country": country,
                "national": "Total",
            })

        S_df = pd.DataFrame(hierarchy_records).set_index("unique_id")

        return Y_df, S_df

    def compute_data_hash(self) -> str:
        """Compute SHA-256 hash of all generated tables for reproducibility."""
        hasher = hashlib.sha256()
        tables = [
            self.dim_product, self.dim_location, self.dim_weather,
            self.dim_customer, self.fact_sales, self.fact_inventory,
            self.fact_social, self.fact_fulfillment, self.fact_reviews,
        ]
        for df in tables:
            if df is not None:
                hasher.update(pd.util.hash_pandas_object(df).values.tobytes())

        return hasher.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Return summary statistics of generated data."""
        tables = {
            "dim_product": self.dim_product,
            "dim_location": self.dim_location,
            "dim_weather": self.dim_weather,
            "dim_customer": self.dim_customer,
            "fact_sales": self.fact_sales,
            "fact_inventory": self.fact_inventory,
            "fact_social": self.fact_social,
            "fact_fulfillment": self.fact_fulfillment,
            "fact_reviews": self.fact_reviews,
        }
        return {
            name: len(df) if df is not None else 0
            for name, df in tables.items()
        }


def main():
    parser = argparse.ArgumentParser(description="GlowCast Data Generator")
    parser.add_argument("--validate-only", action="store_true", help="Generate and validate only (no file output)")
    parser.add_argument("--n-skus", type=int, default=200, help="Number of SKUs (default 200 for quick run)")
    parser.add_argument("--n-days", type=int, default=730, help="Number of days (default 730)")
    args = parser.parse_args()

    print("GlowCast Data Generator")
    print("=" * 60)

    gen = GlowCastDataGenerator(n_skus=args.n_skus, n_days=args.n_days)

    print("Generating all tables...")
    gen.generate_all()

    summary = gen.summary()
    for table, count in summary.items():
        print(f"  {table}: {count:,} rows")

    print(f"\nData hash: {gen.compute_data_hash()[:16]}...")

    print("\nValidating schemas...")
    results = gen.validate_all()
    all_pass = all(results.values())

    for table, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {table}: {status}")

    if all_pass:
        print("\nAll validations passed!")
    else:
        print("\nSome validations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
