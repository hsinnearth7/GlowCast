"""Shared test fixtures for GlowCast test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.data_generator import GlowCastDataGenerator
from app.seed import set_global_seed


@pytest.fixture(scope="session")
def seed():
    """Set global seed for all tests."""
    set_global_seed(42)
    return 42


@pytest.fixture(scope="session")
def rng():
    """Provide a seeded random generator."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_generator(seed):
    """Create a small data generator for testing (50 SKUs, 90 days)."""
    gen = GlowCastDataGenerator(n_skus=50, n_days=90, seed=42)
    gen.generate_all()
    return gen


@pytest.fixture(scope="session")
def small_tables(small_generator):
    """Return all generated tables from small generator."""
    return {
        "dim_product": small_generator.dim_product,
        "dim_location": small_generator.dim_location,
        "dim_weather": small_generator.dim_weather,
        "dim_customer": small_generator.dim_customer,
        "fact_sales": small_generator.fact_sales,
        "fact_inventory": small_generator.fact_inventory,
        "fact_social": small_generator.fact_social,
        "fact_fulfillment": small_generator.fact_fulfillment,
        "fact_reviews": small_generator.fact_reviews,
    }


@pytest.fixture
def sample_Y_df(rng):
    """Create a sample Nixtla-format Y DataFrame."""
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    records = []
    for uid in [f"SKU_{i}__FC_Phoenix" for i in range(1000, 1005)]:
        for d in dates:
            records.append({
                "unique_id": uid,
                "ds": d,
                "y": float(max(0, rng.poisson(10) + 3 * np.sin(d.day_of_year / 365 * 2 * np.pi))),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_predictions(sample_Y_df, rng):
    """Create sample predictions matching Y_df."""
    preds = sample_Y_df.copy()
    preds["y_hat"] = preds["y"] + rng.normal(0, 2, len(preds))
    preds["y_hat"] = preds["y_hat"].clip(lower=0)
    return preds


@pytest.fixture
def binary_treatment_data(rng):
    """Create sample data for causal/uplift testing."""
    n = 1000
    X = rng.standard_normal((n, 5))
    treatment = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    # True effect: features 0 and 1 drive heterogeneous treatment effect
    true_cate = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    outcome = X[:, 0] + 0.5 * X[:, 2] + treatment * true_cate + rng.normal(0, 0.5, n)
    return X, treatment, outcome, true_cate
