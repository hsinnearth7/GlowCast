"""Property-based tests using Hypothesis.

Tests structural invariants that must hold for all valid inputs,
not just specific examples.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from app.data.segment_genes import (
    FC_WEIGHTS,
    PRICE_TIER_PARAMS,
    SEGMENT_GENES,
    compute_thi,
)


class TestSegmentGeneProperties:
    @given(concern=st.sampled_from(["AntiAging", "Acne", "Hydrating", "Brightening", "SunProtection"]),
           texture=st.sampled_from(["Lightweight", "Rich"]))
    def test_all_segments_have_positive_lambda(self, concern, texture):
        genes = SEGMENT_GENES[(concern, texture)]
        assert genes["base_demand_lambda"] > 0

    @given(concern=st.sampled_from(["AntiAging", "Acne", "Hydrating", "Brightening", "SunProtection"]),
           texture=st.sampled_from(["Lightweight", "Rich"]))
    def test_all_segments_have_valid_shelf_life(self, concern, texture):
        genes = SEGMENT_GENES[(concern, texture)]
        assert 365 <= genes["shelf_life_days"] <= 1095  # 1-3 years

    @given(concern=st.sampled_from(["AntiAging", "Acne", "Hydrating", "Brightening", "SunProtection"]),
           texture=st.sampled_from(["Lightweight", "Rich"]))
    def test_replenish_cycle_ordered(self, concern, texture):
        genes = SEGMENT_GENES[(concern, texture)]
        lo, hi = genes["replenish_cycle"]
        assert lo < hi


class TestPriceTierProperties:
    @given(tier=st.sampled_from(["Mass", "Prestige", "Luxury"]))
    def test_price_range_ordered(self, tier):
        lo, hi = PRICE_TIER_PARAMS[tier]["range"]
        assert lo < hi

    @given(tier=st.sampled_from(["Mass", "Prestige", "Luxury"]))
    def test_margin_positive(self, tier):
        lo, hi = PRICE_TIER_PARAMS[tier]["margin"]
        assert lo > 0
        assert hi <= 1


class TestTHIProperties:
    @given(temp=st.floats(min_value=-20, max_value=50),
           humidity=st.floats(min_value=10, max_value=100))
    @settings(max_examples=50)
    def test_thi_is_finite(self, temp, humidity):
        thi = compute_thi(temp, humidity)
        assert np.isfinite(thi)

    @given(humidity=st.floats(min_value=10, max_value=100))
    @settings(max_examples=20)
    def test_thi_increases_with_temperature(self, humidity):
        thi_cold = compute_thi(10, humidity)
        thi_hot = compute_thi(40, humidity)
        assert thi_hot > thi_cold


class TestFCWeightProperties:
    def test_weights_sum_to_one(self):
        total = sum(FC_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_weights_positive(self):
        for fc, w in FC_WEIGHTS.items():
            assert w > 0, f"{fc} has non-positive weight"

    def test_us_has_highest_total_weight(self):
        us_total = sum(w for fc, w in FC_WEIGHTS.items() if "Phoenix" in fc or "Miami" in fc or "Seattle" in fc or "Dallas" in fc)
        assert us_total > 0.35  # US should be ~40%


class TestDataGeneratorProperties:
    @given(n_skus=st.integers(min_value=10, max_value=50))
    @settings(max_examples=3, deadline=60000)
    def test_generator_produces_correct_sku_count(self, n_skus):
        from app.data.data_generator import GlowCastDataGenerator
        gen = GlowCastDataGenerator(n_skus=n_skus, n_days=30, seed=42)
        gen._generate_dim_product()
        # Proportional allocation may round differently; allow small tolerance
        assert abs(len(gen.dim_product) - n_skus) <= 5

    def test_negative_binomial_non_negative(self):
        """NegBin draws should always be >= 0."""
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = max(1, int(rng.integers(1, 20)))
            p = rng.uniform(0.01, 0.99)
            value = rng.negative_binomial(n, p)
            assert value >= 0
