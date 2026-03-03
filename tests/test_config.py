"""Tests for configuration loading and validation."""


import pytest

from app.settings import (
    DEFAULT_CONFIG,
    get_causal_config,
    get_data_config,
    get_eval_config,
    get_experimentation_config,
    get_model_config,
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
        expected = ["data", "segments", "model", "evaluation", "experimentation", "causal", "monitoring", "sql"]
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
        assert dc["n_skus"] == 5000
        assert dc["n_fcs"] == 12
        assert dc["n_countries"] == 5
        assert dc["n_years"] == 3
        assert dc["global_seed"] == 42

    def test_model_config_has_all_models(self):
        mc = get_model_config()
        expected_models = ["naive_ma30", "sarimax", "xgboost", "lightgbm", "chronos2_zs", "routing_ensemble"]
        for model in expected_models:
            assert model in mc, f"Missing model config: {model}"

    def test_model_config_specific(self):
        xgb = get_model_config("xgboost")
        assert xgb["n_estimators"] == 500
        assert xgb["learning_rate"] == 0.05
        assert xgb["max_depth"] == 5

    def test_eval_config(self):
        ec = get_eval_config()
        assert ec["n_windows"] == 12
        assert ec["conformal_coverage"] == 0.90
        assert ec["mape_targets"]["overall"] == 12.0

    def test_experimentation_config(self):
        ec = get_experimentation_config()
        assert ec["cuped"]["target_rho"] == 0.74
        assert ec["cuped"]["target_variance_reduction"] == 0.55

    def test_causal_config(self):
        cc = get_causal_config()
        assert cc["uplift"]["treatment_ratio"] == 0.20
        assert "x_learner" in cc["uplift"]["learners"]

    def test_sql_config(self):
        sc = get_sql_config()
        assert sc["engine"] == "sqlite"
        assert sc["cross_zone"]["penalty_per_unit"] == 50
