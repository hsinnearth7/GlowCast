"""Tests for segment gene definitions."""

from __future__ import annotations

from app.data.segment_genes import (
    CATEGORIES,
    COMMODITIES,
    COMMODITY_BASE_PRICES,
    COST_REDUCTION_ACTIONS,
    COST_TIERS,
    DIRECTION_PHASE,
    PLANT_DEFINITIONS,
    PLANT_WEIGHTS,
    SEGMENT_GENES,
    SUPPLIER_PROFILES,
    SUPPLIERS,
)


class TestSegmentGenes:

    def test_ten_segments(self):
        assert len(SEGMENT_GENES) == 10

    def test_segment_keys_match_axes(self):
        for cat, tier in SEGMENT_GENES:
            assert cat in CATEGORIES
            assert tier in COST_TIERS

    def test_all_combinations_present(self):
        for cat in CATEGORIES:
            for tier in COST_TIERS:
                assert (cat, tier) in SEGMENT_GENES

    def test_segment_gene_fields(self):
        required_fields = {
            "sku_count", "subcategories", "base_unit_cost", "cost_volatility",
            "commodity_sensitivity", "labor_intensity", "overhead_allocation",
            "seasonal_amplitude", "seasonal_direction", "tariff_exposure",
            "quality_rejection_rate", "moq",
        }
        for key, genes in SEGMENT_GENES.items():
            missing = required_fields - set(genes.keys())
            assert not missing, f"Segment {key} missing fields: {missing}"

    def test_plant_definitions_count(self):
        assert len(PLANT_DEFINITIONS) == 12

    def test_plant_weights_sum_to_one(self):
        total = sum(PLANT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_supplier_count(self):
        assert len(SUPPLIERS) == 5
        assert len(SUPPLIER_PROFILES) == 5

    def test_commodity_count(self):
        assert len(COMMODITIES) == 5
        assert len(COMMODITY_BASE_PRICES) == 5

    def test_direction_phase_keys(self):
        for genes in SEGMENT_GENES.values():
            assert genes["seasonal_direction"] in DIRECTION_PHASE

    def test_cost_reduction_actions(self):
        assert len(COST_REDUCTION_ACTIONS) == 8
        assert "supplier_switch" in COST_REDUCTION_ACTIONS
