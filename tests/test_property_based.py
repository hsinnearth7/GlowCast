"""Property-based tests using Hypothesis.

Tests structural invariants that must hold for all valid inputs,
not just specific examples.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from app.data.segment_genes import (
    CATEGORIES,
    COST_TIERS,
    PLANT_WEIGHTS,
    SEGMENT_GENES,
)


class TestSegmentGeneProperties:
    @given(category=st.sampled_from(CATEGORIES),
           cost_tier=st.sampled_from(COST_TIERS))
    def test_all_segments_have_positive_base_cost(self, category, cost_tier):
        genes = SEGMENT_GENES[(category, cost_tier)]
        assert genes["base_unit_cost"] > 0

    @given(category=st.sampled_from(CATEGORIES),
           cost_tier=st.sampled_from(COST_TIERS))
    def test_all_segments_have_valid_volatility(self, category, cost_tier):
        genes = SEGMENT_GENES[(category, cost_tier)]
        assert 0 <= genes["cost_volatility"] <= 1

    @given(category=st.sampled_from(CATEGORIES),
           cost_tier=st.sampled_from(COST_TIERS))
    def test_commodity_sensitivity_in_range(self, category, cost_tier):
        genes = SEGMENT_GENES[(category, cost_tier)]
        assert 0 <= genes["commodity_sensitivity"] <= 1

    @given(category=st.sampled_from(CATEGORIES),
           cost_tier=st.sampled_from(COST_TIERS))
    def test_labor_intensity_in_range(self, category, cost_tier):
        genes = SEGMENT_GENES[(category, cost_tier)]
        assert 0 <= genes["labor_intensity"] <= 1


class TestPlantWeightProperties:
    def test_weights_sum_to_one(self):
        total = sum(PLANT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_weights_positive(self):
        for plant, w in PLANT_WEIGHTS.items():
            assert w > 0, f"{plant} has non-positive weight"

    def test_asia_has_significant_weight(self):
        asia_total = sum(w for p, w in PLANT_WEIGHTS.items()
                         if "Shenzhen" in p or "Shanghai" in p or "Taipei" in p
                         or "Pune" in p or "Chennai" in p or "Tokyo" in p or "Osaka" in p)
        assert asia_total > 0.50  # Asia should be majority


class TestDataGeneratorProperties:
    @given(n_skus=st.integers(min_value=10, max_value=50))
    @settings(max_examples=3, deadline=60000)
    def test_generator_produces_correct_sku_count(self, n_skus):
        from app.data.data_generator import CostDataGenerator
        gen = CostDataGenerator(n_skus=n_skus, n_days=30, seed=42)
        gen._generate_dim_product()
        # Proportional allocation across 10 segments may round differently;
        # with small n_skus the rounding error can be up to n_segments
        assert abs(len(gen.dim_product) - n_skus) <= 10

    def test_negative_binomial_non_negative(self):
        """NegBin draws should always be >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = max(1, int(rng.integers(1, 20)))
            p = rng.uniform(0.01, 0.99)
            value = rng.negative_binomial(n, p)
            assert value >= 0
