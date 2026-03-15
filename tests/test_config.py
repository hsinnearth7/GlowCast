"""Tests for configuration loading and validation."""


import pytest

from app.settings import (
    DEFAULT_CONFIG,
    get_causal_config,
    get_cost_config,
    get_data_config,
    get_experimentation_config,
    get_pricing_config,
    get_sql_config,
    load_config,
)


class TestLoadConfig:
    def setup_method(self):
        load_config.cache_clear()

    def test_config_file_exists(self):
        assert DEFAULT_CONFIG.exists(), f"Config not found: {DEFAULT_CONFIG}"

    def test_load_config_returns_dict(self):
        config = load_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_load_config_has_all_sections(self):
        config = load_config()
        expected = ["data", "segments", "cost", "pricing", "experimentation", "causal", "monitoring", "sql"]
        for section in expected:
            assert section in config, f"Missing config section: {section}"

    def test_load_config_caching(self):
        c1 = load_config()
        c2 = load_config()
        assert c1 is c2  # same object due to lru_cache

    def test_load_config_missing_file_raises(self):
        load_config.cache_clear()
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_data_config_values(self):
        dc = get_data_config()
        assert dc["n_skus"] == 500
        assert dc["n_plants"] == 12
        assert dc["n_suppliers"] == 5

    def test_cost_config_returns_dict(self):
        cc = get_cost_config()
        assert isinstance(cc, dict)

    def test_cost_config_section(self):
        section = get_cost_config("should_cost")
        assert isinstance(section, dict)

    def test_pricing_config(self):
        pc = get_pricing_config()
        assert isinstance(pc, dict)

    def test_experimentation_config(self):
        ec = get_experimentation_config()
        assert ec["cuped"]["target_rho"] == 0.74
        assert ec["cuped"]["variance_reduction_target"] == 0.55

    def test_causal_config(self):
        cc = get_causal_config()
        assert cc["uplift"]["treatment_ratio"] == 0.20
        assert "x_learner" in cc["uplift"]["meta_learners"]

    def test_sql_config(self):
        sc = get_sql_config()
        assert "cost_variance" in sc
        assert "should_cost_gap" in sc
