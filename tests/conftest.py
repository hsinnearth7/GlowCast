"""Shared test fixtures for GlowCast cost analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.seed import GLOBAL_SEED


@pytest.fixture(scope="session")
def seed() -> int:
    return GLOBAL_SEED


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(GLOBAL_SEED)


@pytest.fixture(scope="session")
def small_generator():
    """Session-scoped small data generator for fast tests."""
    from app.data.data_generator import CostDataGenerator

    gen = CostDataGenerator(n_skus=50, n_days=90, seed=GLOBAL_SEED)
    gen.generate_all()
    return gen


@pytest.fixture(scope="session")
def small_tables(small_generator) -> dict[str, pd.DataFrame]:
    """Dict of all generated tables from the small generator."""
    return {
        "dim_product": small_generator.dim_product,
        "dim_supplier": small_generator.dim_supplier,
        "dim_plant": small_generator.dim_plant,
        "fact_cost_transactions": small_generator.fact_cost_transactions,
        "fact_supplier_quotes": small_generator.fact_supplier_quotes,
        "fact_cost_reduction_actions": small_generator.fact_cost_reduction_actions,
        "fact_commodity_prices": small_generator.fact_commodity_prices,
        "fact_purchase_orders": small_generator.fact_purchase_orders,
        "fact_quality_events": small_generator.fact_quality_events,
    }


@pytest.fixture
def sample_cost_df(rng) -> pd.DataFrame:
    """Sample cost transaction data for unit tests."""
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "transaction_id": [f"TXN_{i:06d}" for i in range(n)],
        "transaction_date": np.tile(dates[:100], 5),
        "sku_id": np.repeat([f"SKU_{i:04d}" for i in range(1, 6)], 100),
        "plant_id": rng.choice(["PLT_Shenzhen", "PLT_Munich", "PLT_Detroit"], n),
        "supplier_id": rng.choice(["SUP_001", "SUP_002", "SUP_003"], n),
        "raw_material_cost": rng.uniform(5, 30, n).round(2),
        "labor_cost": rng.uniform(3, 15, n).round(2),
        "overhead_cost": rng.uniform(2, 10, n).round(2),
        "logistics_cost": rng.uniform(0.5, 3, n).round(2),
        "total_unit_cost": rng.uniform(15, 60, n).round(2),
        "volume": rng.integers(1, 100, n),
    })


@pytest.fixture
def binary_treatment_data(rng) -> pd.DataFrame:
    """Synthetic data for causal / uplift tests (kept from original)."""
    n = 1000
    X = rng.standard_normal((n, 5))
    treatment = (rng.random(n) < 0.2).astype(int)
    tau = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    y = 2.0 + X[:, 0] + 0.5 * X[:, 1] + treatment * tau + rng.standard_normal(n) * 0.5

    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(5)])
    df["treatment"] = treatment
    df["outcome"] = y
    return df
