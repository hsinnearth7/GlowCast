"""Tests for product taxonomy and segment gene definitions."""


from app.data.segment_genes import (
    BRAND_CONCERN_WEIGHTS,
    BRANDS,
    CONCERN_BASE_VOL,
    CONCERNS,
    FC_CLIMATE_PARAMS,
    FC_DEFINITIONS,
    FC_WEIGHTS,
    GLOBAL_SOURCES,
    PRICE_TIER_PARAMS,
    PRICE_TIERS,
    SEGMENT_GENES,
    SOURCE_WEIGHTS,
    TEMPERATURE_SWITCH_POINT,
    TEXTURES,
    compute_thi,
)


class TestTaxonomy:
    def test_concerns_count(self):
        assert len(CONCERNS) == 5

    def test_textures_count(self):
        assert len(TEXTURES) == 2

    def test_price_tiers_count(self):
        assert len(PRICE_TIERS) == 3

    def test_brands_count(self):
        assert len(BRANDS) == 5


class TestSegmentGenes:
    def test_segment_count(self):
        assert len(SEGMENT_GENES) == 10

    def test_all_concern_texture_pairs(self):
        for concern in CONCERNS:
            for texture in TEXTURES:
                assert (concern, texture) in SEGMENT_GENES

    def test_sku_counts_sum(self):
        total = sum(g["sku_count"] for g in SEGMENT_GENES.values())
        assert total == 175  # base allocation

    def test_all_genes_have_required_keys(self):
        required = [
            "sku_count", "subcategories", "base_demand_lambda",
            "price_elasticity", "seasonal_amplitude", "seasonal_direction",
            "social_sensitivity", "return_rate", "shelf_life_days",
            "replenish_cycle", "trend_annual_pct",
        ]
        for key, genes in SEGMENT_GENES.items():
            for field in required:
                assert field in genes, f"Missing {field} in {key}"

    def test_price_elasticity_negative(self):
        for key, genes in SEGMENT_GENES.items():
            assert genes["price_elasticity"] < 0, f"Elasticity should be negative for {key}"

    def test_seasonal_direction_valid(self):
        valid = {"summer", "winter", "spring", "neutral"}
        for key, genes in SEGMENT_GENES.items():
            assert genes["seasonal_direction"] in valid

    def test_sunprotection_highest_seasonality(self):
        sp_light = SEGMENT_GENES[("SunProtection", "Lightweight")]
        assert sp_light["seasonal_amplitude"] == 0.90

    def test_hydrating_inverse_seasonal_hedge(self):
        light = SEGMENT_GENES[("Hydrating", "Lightweight")]
        rich = SEGMENT_GENES[("Hydrating", "Rich")]
        assert light["seasonal_direction"] == "summer"
        assert rich["seasonal_direction"] == "winter"

    def test_brightening_shortest_shelf_life(self):
        bl = SEGMENT_GENES[("Brightening", "Lightweight")]
        assert bl["shelf_life_days"] == 450
        for key, genes in SEGMENT_GENES.items():
            assert genes["shelf_life_days"] >= 450

    def test_social_sensitivity_range(self):
        for key, genes in SEGMENT_GENES.items():
            assert 0 <= genes["social_sensitivity"] <= 1


class TestPriceTiers:
    def test_mass_range(self):
        assert PRICE_TIER_PARAMS["Mass"]["range"] == (5, 25)

    def test_luxury_highest_margin(self):
        luxury_min = PRICE_TIER_PARAMS["Luxury"]["margin"][0]
        prestige_max = PRICE_TIER_PARAMS["Prestige"]["margin"][1]
        assert luxury_min >= prestige_max - 0.02  # approximate

    def test_cogs_decreases_with_tier(self):
        mass_max = PRICE_TIER_PARAMS["Mass"]["cogs_pct"][1]
        luxury_max = PRICE_TIER_PARAMS["Luxury"]["cogs_pct"][1]
        assert luxury_max < mass_max


class TestFCDefinitions:
    def test_fc_count(self):
        assert len(FC_DEFINITIONS) == 12

    def test_countries(self):
        countries = {fc["country"] for fc in FC_DEFINITIONS.values()}
        assert countries == {"US", "DE", "UK", "JP", "IN"}

    def test_us_has_4_fcs(self):
        us_fcs = [fc for fc in FC_DEFINITIONS.values() if fc["country"] == "US"]
        assert len(us_fcs) == 4

    def test_fc_weights_sum_to_one(self):
        total = sum(FC_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_fcs_have_weights(self):
        for fc_id in FC_DEFINITIONS:
            assert fc_id in FC_WEIGHTS


class TestClimateParams:
    def test_climate_param_count(self):
        assert len(FC_CLIMATE_PARAMS) == 12

    def test_delhi_extreme_swing(self):
        delhi = FC_CLIMATE_PARAMS["Delhi"]
        assert delhi[1] == 18  # highest temp amplitude

    def test_temperature_switch_point(self):
        assert TEMPERATURE_SWITCH_POINT == 23.5

    def test_thi_computation(self):
        thi = compute_thi(30, 70)
        assert isinstance(thi, float)
        assert thi > 20


class TestBrandConcernWeights:
    def test_all_brands_have_weights(self):
        for brand in BRANDS:
            assert brand in BRAND_CONCERN_WEIGHTS

    def test_weights_sum_to_one(self):
        for brand, weights in BRAND_CONCERN_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{brand} weights sum to {total}"


class TestSocialParams:
    def test_source_weights_sum(self):
        total = sum(SOURCE_WEIGHTS)
        assert abs(total - 1.0) < 0.01

    def test_source_count_matches(self):
        assert len(GLOBAL_SOURCES) == len(SOURCE_WEIGHTS)

    def test_concern_base_vol(self):
        assert len(CONCERN_BASE_VOL) == 5
        for vol in CONCERN_BASE_VOL.values():
            assert vol > 0
