"""GlowCast synthetic data generator engine.

Generates a complete star schema dataset for cost & commercial analytics:
- 500 SKUs across 10 category × cost_tier segments
- 12 manufacturing plants across 7 countries
- 5 suppliers with distinct cost/quality profiles
- 3 years of daily cost transactions with commodity price sensitivity
- Supplier quotes, cost reduction actions, purchase orders, quality events

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
    COMMODITIES,
    COMMODITY_BASE_PRICES,
    COMMODITY_SEASONAL_PHASE,
    COMMODITY_VOLATILITY,
    COST_REDUCTION_ACTIONS,
    PLANT_DEFINITIONS,
    PLANT_WEIGHTS,
    SEGMENT_GENES,
    SUPPLIER_PROFILES,
    SUPPLIERS,
)
from app.data.star_schema import (
    Dim_Plant,
    Dim_Product,
    Dim_Supplier,
    Fact_Commodity_Prices,
    Fact_Cost_Reduction_Actions,
    Fact_Cost_Transactions,
    Fact_Purchase_Orders,
    Fact_Quality_Events,
    Fact_Supplier_Quotes,
)
from app.seed import GLOBAL_SEED


class CostDataGenerator:
    """Generates the complete GlowCast cost analytics star schema dataset."""

    def __init__(
        self,
        n_skus: int = 500,
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
        self.dim_supplier: pd.DataFrame | None = None
        self.dim_plant: pd.DataFrame | None = None
        self.fact_cost_transactions: pd.DataFrame | None = None
        self.fact_supplier_quotes: pd.DataFrame | None = None
        self.fact_cost_reduction_actions: pd.DataFrame | None = None
        self.fact_commodity_prices: pd.DataFrame | None = None
        self.fact_purchase_orders: pd.DataFrame | None = None
        self.fact_quality_events: pd.DataFrame | None = None

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all star schema tables."""
        self._generate_dim_supplier()
        self._generate_dim_plant()
        self._generate_dim_product()
        self._generate_fact_commodity_prices()
        self._generate_fact_cost_transactions()
        self._generate_fact_supplier_quotes()
        self._generate_fact_cost_reduction_actions()
        self._generate_fact_purchase_orders()
        self._generate_fact_quality_events()

        return {
            "dim_product": self.dim_product,
            "dim_supplier": self.dim_supplier,
            "dim_plant": self.dim_plant,
            "fact_cost_transactions": self.fact_cost_transactions,
            "fact_supplier_quotes": self.fact_supplier_quotes,
            "fact_cost_reduction_actions": self.fact_cost_reduction_actions,
            "fact_commodity_prices": self.fact_commodity_prices,
            "fact_purchase_orders": self.fact_purchase_orders,
            "fact_quality_events": self.fact_quality_events,
        }

    # ── Dimension tables ─────────────────────────────────────────────────

    def _generate_dim_supplier(self) -> None:
        """Generate supplier dimension table."""
        records = []
        for i, (name, profile) in enumerate(SUPPLIER_PROFILES.items()):
            records.append({
                "supplier_id": f"SUP_{i + 1:03d}",
                "supplier_name": name,
                "country": profile["country"],
                "quality_score": profile["quality_score"],
                "on_time_delivery_pct": profile["on_time_pct"],
                "lead_time_days": profile["lead_time_days"],
                "price_premium": profile["price_premium"],
            })
        self.dim_supplier = pd.DataFrame(records)

    def _generate_dim_plant(self) -> None:
        """Generate plant dimension table."""
        records = []
        for plant_id, info in PLANT_DEFINITIONS.items():
            records.append({
                "plant_id": plant_id,
                "country": info["country"],
                "region": info["region"],
                "lat": info["lat"],
                "lon": info["lon"],
                "labor_rate_hourly": info["labor_rate"],
                "overhead_rate": info["overhead_rate"],
                "capacity_utilization": info["capacity_util"],
            })
        self.dim_plant = pd.DataFrame(records)

    def _generate_dim_product(self) -> None:
        """Generate product dimension with cost segment assignments."""
        records = []
        sku_idx = 0

        # Calculate actual SKU count per segment (proportional)
        total_gene_skus = sum(g["sku_count"] for g in SEGMENT_GENES.values())
        supplier_ids = [f"SUP_{i + 1:03d}" for i in range(len(SUPPLIERS))]

        for (category, cost_tier), genes in SEGMENT_GENES.items():
            segment_n = max(1, int(self.n_skus * genes["sku_count"] / total_gene_skus))
            subcats = genes["subcategories"]

            for j in range(segment_n):
                sku_idx += 1
                subcat = subcats[j % len(subcats)]
                base_cost = genes["base_unit_cost"] * (1 + self.rng.normal(0, 0.15))
                base_cost = max(0.10, base_cost)

                # Target cost is typically 5-15% below current
                target_cost = base_cost * (1 - self.rng.uniform(0.05, 0.15))

                commodity = self.rng.choice(COMMODITIES)
                _supplier = self.rng.choice(supplier_ids)  # noqa: F841 (preserves RNG state)

                records.append({
                    "sku_id": f"SKU_{sku_idx:04d}",
                    "product_name": f"{category}_{subcat}_{sku_idx}",
                    "category": category,
                    "cost_tier": cost_tier,
                    "subcategory": subcat,
                    "primary_supplier": self.rng.choice(SUPPLIERS),
                    "base_unit_cost": round(base_cost, 4),
                    "target_cost": round(target_cost, 4),
                    "commodity_exposure": commodity,
                    "commodity_sensitivity": round(float(genes["commodity_sensitivity"]), 4),
                    "labor_intensity": round(float(genes["labor_intensity"]), 4),
                    "overhead_allocation": round(float(genes["overhead_allocation"]), 4),
                    "moq": genes["moq"],
                    "tariff_exposure": round(float(genes["tariff_exposure"]), 4),
                })

        self.dim_product = pd.DataFrame(records[:self.n_skus])

    # ── Fact tables ──────────────────────────────────────────────────────

    def _generate_fact_commodity_prices(self) -> None:
        """Generate daily commodity price index time series."""
        records = []
        for commodity in COMMODITIES:
            base_price = COMMODITY_BASE_PRICES[commodity]
            volatility = COMMODITY_VOLATILITY[commodity]
            phase = COMMODITY_SEASONAL_PHASE[commodity]

            # Random walk with mean reversion + seasonality
            price = base_price
            prev_idx = 1.0
            for i, date in enumerate(self.dates):
                day_of_year = date.dayofyear
                seasonal = 0.05 * np.sin(2 * np.pi * day_of_year / 365 + phase)
                shock = self.rng.normal(0, volatility / np.sqrt(365))
                mean_reversion = -0.002 * (price - base_price) / base_price
                price = price * (1 + seasonal / 365 + shock + mean_reversion)
                price = max(base_price * 0.5, min(base_price * 2.0, price))

                price_idx = price / base_price
                pct_change = 0.0 if i == 0 else (price_idx - prev_idx) / prev_idx
                prev_idx = price_idx

                # 30-day volatility (simplified)
                vol_30d = volatility * np.sqrt(30 / 365)

                records.append({
                    "price_date": date,
                    "commodity": commodity,
                    "price_index": round(price_idx, 6),
                    "pct_change": round(pct_change, 6),
                    "volatility_30d": round(vol_30d, 6),
                })

        self.fact_commodity_prices = pd.DataFrame(records)

    def _generate_fact_cost_transactions(self) -> None:
        """Generate daily cost transaction records."""
        if self.dim_product is None:
            self._generate_dim_product()

        plant_ids = list(PLANT_DEFINITIONS.keys())
        plant_weights = [PLANT_WEIGHTS[p] for p in plant_ids]
        supplier_ids = [f"SUP_{i + 1:03d}" for i in range(len(SUPPLIERS))]

        records = []
        txn_idx = 0

        # Sample a subset of SKUs per day (not all SKUs transact daily)
        n_daily_txns = max(5, self.n_skus // 10)

        for date in self.dates:
            day_of_year = date.dayofyear
            sku_sample = self.dim_product.sample(
                n=min(n_daily_txns, len(self.dim_product)),
                random_state=int(self.rng.integers(0, 2**31)),
            )

            for _, product in sku_sample.iterrows():
                txn_idx += 1
                base_cost = float(product["base_unit_cost"])
                commodity_sens = float(product["commodity_sensitivity"])
                labor_int = float(product["labor_intensity"])
                overhead_alloc = float(product["overhead_allocation"])

                # Commodity price effect
                commodity_noise = self.rng.normal(0, 0.03)
                seasonal_effect = 0.05 * np.sin(2 * np.pi * day_of_year / 365)

                # Cost components
                raw_material = base_cost * commodity_sens * (1 + commodity_noise + seasonal_effect * commodity_sens)
                labor = base_cost * labor_int * (1 + self.rng.normal(0, 0.02))
                overhead = base_cost * overhead_alloc * (1 + self.rng.normal(0, 0.01))
                logistics = base_cost * 0.05 * (1 + self.rng.normal(0, 0.05))

                total = max(0.01, raw_material + labor + overhead + logistics)
                volume = max(1, int(self.rng.negative_binomial(5, 0.3)))

                plant = self.rng.choice(plant_ids, p=plant_weights)
                supplier = self.rng.choice(supplier_ids)

                records.append({
                    "transaction_id": f"TXN_{txn_idx:08d}",
                    "transaction_date": date,
                    "sku_id": product["sku_id"],
                    "plant_id": plant,
                    "supplier_id": supplier,
                    "raw_material_cost": round(max(0, raw_material), 4),
                    "labor_cost": round(max(0, labor), 4),
                    "overhead_cost": round(max(0, overhead), 4),
                    "logistics_cost": round(max(0, logistics), 4),
                    "total_unit_cost": round(total, 4),
                    "volume": volume,
                })

        self.fact_cost_transactions = pd.DataFrame(records)

    def _generate_fact_supplier_quotes(self) -> None:
        """Generate supplier quote data (quotes arrive periodically)."""
        if self.dim_product is None:
            self._generate_dim_product()

        supplier_ids = [f"SUP_{i + 1:03d}" for i in range(len(SUPPLIERS))]
        supplier_premiums = {
            f"SUP_{i + 1:03d}": profile["price_premium"]
            for i, (_, profile) in enumerate(SUPPLIER_PROFILES.items())
        }

        records = []
        quote_idx = 0

        # Quotes come quarterly for each SKU from 2-3 suppliers
        quarter_dates = pd.date_range(self.start_date, periods=self.n_days // 90, freq="QS")

        for date in quarter_dates:
            sku_sample = self.dim_product.sample(
                n=min(self.n_skus // 2, len(self.dim_product)),
                random_state=int(self.rng.integers(0, 2**31)),
            )
            for _, product in sku_sample.iterrows():
                n_quotes = self.rng.integers(2, 4)
                quoting_suppliers = self.rng.choice(supplier_ids, size=n_quotes, replace=False)

                for sup_id in quoting_suppliers:
                    quote_idx += 1
                    base = float(product["base_unit_cost"])
                    premium = supplier_premiums.get(sup_id, 0.0)
                    noise = self.rng.normal(0, 0.05)
                    quoted_price = base * (1 + premium + noise)

                    sup_profile_idx = int(sup_id.split("_")[1]) - 1
                    sup_name = SUPPLIERS[sup_profile_idx]
                    lead_time = SUPPLIER_PROFILES[sup_name]["lead_time_days"] + self.rng.integers(-2, 3)

                    records.append({
                        "quote_id": f"QUO_{quote_idx:06d}",
                        "quote_date": date + timedelta(days=int(self.rng.integers(0, 15))),
                        "supplier_id": sup_id,
                        "sku_id": product["sku_id"],
                        "quoted_price": round(max(0.01, quoted_price), 4),
                        "lead_time_days": max(1, lead_time),
                        "moq": int(product["moq"]),
                        "valid_until": date + timedelta(days=90),
                    })

        self.fact_supplier_quotes = pd.DataFrame(records)

    def _generate_fact_cost_reduction_actions(self) -> None:
        """Generate historical cost reduction action records."""
        if self.dim_product is None:
            self._generate_dim_product()

        records = []
        action_idx = 0

        # Generate ~2-5 actions per quarter
        quarter_dates = pd.date_range(self.start_date, periods=self.n_days // 90, freq="QS")

        for date in quarter_dates:
            n_actions = self.rng.integers(2, 6)
            sku_sample = self.dim_product.sample(
                n=min(n_actions, len(self.dim_product)),
                random_state=int(self.rng.integers(0, 2**31)),
            )

            for _, product in sku_sample.iterrows():
                action_idx += 1
                action_type = self.rng.choice(COST_REDUCTION_ACTIONS)
                projected = self.rng.uniform(0.02, 0.15)

                # Status based on age
                days_ago = (self.dates[-1] - date).days
                if days_ago > 365:
                    status = self.rng.choice(["completed", "cancelled"], p=[0.75, 0.25])
                elif days_ago > 180:
                    status = self.rng.choice(["completed", "in_progress", "cancelled"], p=[0.5, 0.3, 0.2])
                else:
                    status = self.rng.choice(["proposed", "approved", "in_progress"], p=[0.3, 0.4, 0.3])

                actual_savings = None
                if status == "completed":
                    # Actual savings = projected * realization_rate + noise
                    realization = self.rng.uniform(0.5, 1.2)
                    actual_savings = round(projected * realization, 4)

                records.append({
                    "action_id": f"CRA_{action_idx:05d}",
                    "action_date": date + timedelta(days=int(self.rng.integers(0, 30))),
                    "sku_id": product["sku_id"],
                    "action_type": action_type,
                    "projected_savings_pct": round(projected, 4),
                    "actual_savings_pct": actual_savings,
                    "status": status,
                })

        self.fact_cost_reduction_actions = pd.DataFrame(records)

    def _generate_fact_purchase_orders(self) -> None:
        """Generate purchase order records."""
        if self.dim_product is None:
            self._generate_dim_product()

        supplier_ids = [f"SUP_{i + 1:03d}" for i in range(len(SUPPLIERS))]
        plant_ids = list(PLANT_DEFINITIONS.keys())
        plant_weights = [PLANT_WEIGHTS[p] for p in plant_ids]

        records = []
        po_idx = 0

        # Weekly POs for a subset of SKUs
        weekly_dates = pd.date_range(self.start_date, periods=self.n_days // 7, freq="W")

        for date in weekly_dates:
            n_orders = max(3, self.n_skus // 20)
            sku_sample = self.dim_product.sample(
                n=min(n_orders, len(self.dim_product)),
                random_state=int(self.rng.integers(0, 2**31)),
            )

            for _, product in sku_sample.iterrows():
                po_idx += 1
                base = float(product["base_unit_cost"])
                noise = self.rng.normal(0, 0.03)
                unit_price = max(0.01, base * (1 + noise))
                qty = max(1, int(self.rng.negative_binomial(3, 0.2)))
                total = unit_price * qty

                supplier = self.rng.choice(supplier_ids)
                plant = self.rng.choice(plant_ids, p=plant_weights)

                # Delivery status
                sup_profile_idx = int(supplier.split("_")[1]) - 1
                sup_name = SUPPLIERS[sup_profile_idx]
                on_time_pct = SUPPLIER_PROFILES[sup_name]["on_time_pct"]
                expected_days = SUPPLIER_PROFILES[sup_name]["lead_time_days"]

                is_delivered = self.rng.random() < 0.85
                is_on_time = self.rng.random() < on_time_pct

                if is_delivered:
                    if is_on_time:
                        status = "delivered"
                        actual_days = expected_days + self.rng.integers(-1, 2)
                    else:
                        status = "late"
                        actual_days = expected_days + self.rng.integers(3, 15)
                else:
                    status = self.rng.choice(["pending", "shipped"])
                    actual_days = None

                records.append({
                    "po_id": f"PO_{po_idx:07d}",
                    "order_date": date,
                    "sku_id": product["sku_id"],
                    "supplier_id": supplier,
                    "plant_id": plant,
                    "unit_price": round(unit_price, 4),
                    "quantity": qty,
                    "total_amount": round(total, 4),
                    "delivery_status": status,
                    "actual_delivery_days": actual_days if actual_days is not None else 0,
                })

        self.fact_purchase_orders = pd.DataFrame(records)

    def _generate_fact_quality_events(self) -> None:
        """Generate quality inspection event records."""
        if self.dim_product is None:
            self._generate_dim_product()

        supplier_ids = [f"SUP_{i + 1:03d}" for i in range(len(SUPPLIERS))]
        plant_ids = list(PLANT_DEFINITIONS.keys())
        plant_weights = [PLANT_WEIGHTS[p] for p in plant_ids]

        records = []
        event_idx = 0

        # Monthly quality inspections
        monthly_dates = pd.date_range(self.start_date, periods=self.n_days // 30, freq="MS")

        for date in monthly_dates:
            n_inspections = max(3, self.n_skus // 15)
            sku_sample = self.dim_product.sample(
                n=min(n_inspections, len(self.dim_product)),
                random_state=int(self.rng.integers(0, 2**31)),
            )

            for _, product in sku_sample.iterrows():
                event_idx += 1
                batch_size = max(10, int(self.rng.negative_binomial(5, 0.3) * 10))

                # Base defect rate from segment genes
                category = product["category"]
                cost_tier = product["cost_tier"]
                genes = SEGMENT_GENES.get((category, cost_tier), {})
                base_defect = genes.get("quality_rejection_rate", 0.02)

                defect_rate = max(0, base_defect + self.rng.normal(0, base_defect * 0.5))
                defects = int(batch_size * defect_rate)

                # Disposition based on defect severity
                if defect_rate < 0.01:
                    disposition = "accepted"
                elif defect_rate < 0.03:
                    disposition = self.rng.choice(["accepted", "rework"], p=[0.6, 0.4])
                elif defect_rate < 0.08:
                    disposition = self.rng.choice(["rework", "scrap"], p=[0.5, 0.5])
                else:
                    disposition = self.rng.choice(["scrap", "return_to_supplier"], p=[0.4, 0.6])

                supplier = self.rng.choice(supplier_ids)
                plant = self.rng.choice(plant_ids, p=plant_weights)

                records.append({
                    "event_id": f"QE_{event_idx:06d}",
                    "event_date": date + timedelta(days=int(self.rng.integers(0, 28))),
                    "sku_id": product["sku_id"],
                    "supplier_id": supplier,
                    "plant_id": plant,
                    "defect_rate": round(defect_rate, 6),
                    "batch_size": batch_size,
                    "defects_found": defects,
                    "disposition": disposition,
                })

        self.fact_quality_events = pd.DataFrame(records)

    # ── Validation ───────────────────────────────────────────────────────

    def validate_all(self) -> dict[str, bool]:
        """Validate all tables against Pandera schemas."""
        schema_map = {
            "dim_product": (self.dim_product, Dim_Product),
            "dim_supplier": (self.dim_supplier, Dim_Supplier),
            "dim_plant": (self.dim_plant, Dim_Plant),
            "fact_cost_transactions": (self.fact_cost_transactions, Fact_Cost_Transactions),
            "fact_supplier_quotes": (self.fact_supplier_quotes, Fact_Supplier_Quotes),
            "fact_cost_reduction_actions": (self.fact_cost_reduction_actions, Fact_Cost_Reduction_Actions),
            "fact_commodity_prices": (self.fact_commodity_prices, Fact_Commodity_Prices),
            "fact_purchase_orders": (self.fact_purchase_orders, Fact_Purchase_Orders),
            "fact_quality_events": (self.fact_quality_events, Fact_Quality_Events),
        }

        results = {}
        for name, (df, schema) in schema_map.items():
            if df is None:
                results[name] = False
                continue
            try:
                schema.validate(df, lazy=True)
                results[name] = True
            except pandera.errors.SchemaErrors as exc:
                print(f"  ✗ {name}: {len(exc.failure_cases)} failures")
                results[name] = False

        return results

    def compute_data_hash(self) -> str:
        """Compute SHA-256 hash of all generated data for reproducibility."""
        hasher = hashlib.sha256()
        tables = [
            self.dim_product, self.dim_supplier, self.dim_plant,
            self.fact_cost_transactions, self.fact_supplier_quotes,
            self.fact_cost_reduction_actions, self.fact_commodity_prices,
            self.fact_purchase_orders, self.fact_quality_events,
        ]
        for df in tables:
            if df is not None:
                hasher.update(pd.util.hash_pandas_object(df).values.tobytes())
        return hasher.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Return summary statistics of generated data."""
        tables = self.generate_all() if self.dim_product is None else {
            "dim_product": self.dim_product,
            "dim_supplier": self.dim_supplier,
            "dim_plant": self.dim_plant,
            "fact_cost_transactions": self.fact_cost_transactions,
            "fact_supplier_quotes": self.fact_supplier_quotes,
            "fact_cost_reduction_actions": self.fact_cost_reduction_actions,
            "fact_commodity_prices": self.fact_commodity_prices,
            "fact_purchase_orders": self.fact_purchase_orders,
            "fact_quality_events": self.fact_quality_events,
        }
        return {name: len(df) if df is not None else 0 for name, df in tables.items()}


# ── CLI entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GlowCast Cost Data Generator")
    parser.add_argument("--n-skus", type=int, default=500)
    parser.add_argument("--n-days", type=int, default=1095)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    gen = CostDataGenerator(n_skus=args.n_skus, n_days=args.n_days)

    if args.validate_only:
        gen.generate_all()
        results = gen.validate_all()
        all_ok = all(results.values())
        for name, ok in results.items():
            print(f"  {'✓' if ok else '✗'} {name}")
        sys.exit(0 if all_ok else 1)

    tables = gen.generate_all()
    print(f"Generated {len(tables)} tables:")
    for name, df in tables.items():
        print(f"  {name}: {len(df):,} rows × {len(df.columns)} cols")
    print(f"Data hash: {gen.compute_data_hash()[:12]}")


if __name__ == "__main__":
    main()
