# CLAUDE.md — GlowCast

## Project Overview
GlowCast is a beauty/skincare supply chain AI platform for demand forecasting, causal inference, A/B testing, and MLOps. It simulates 5,000 SKUs across 12 fulfillment centers in 5 countries.

## Tech Stack
- Python 3.11+, pandas, numpy, scipy, scikit-learn
- LightGBM, XGBoost, statsforecast, hierarchicalforecast
- Pandera (data contracts), structlog (logging), PyYAML (config)
- MLflow (experiment tracking), Evidently (drift monitoring)
- DoWhy, CausalML, EconML (causal inference)
- SHAP, LIME (explainability)

## Key Commands
```bash
pip install -e ".[dev]"                          # Install
python -m app.data.data_generator --validate-only # Validate data schemas
pytest tests/ -v                                  # Run 120+ tests
ruff check app/ tests/                           # Lint
docker compose build && docker compose up -d     # Docker
```

## Architecture Patterns
- **Config:** `app/settings.py` — `load_config()` with `@lru_cache`, YAML-based
- **Logging:** `app/logging.py` — structlog with JSON/console modes
- **Seed:** `app/seed.py` — `GLOBAL_SEED=42`, `set_global_seed()`
- **Models:** ABC + Strategy + Factory pattern in `app/forecasting/models.py`
- **Data contracts:** Pandera schemas in `app/data/star_schema.py`
- **Feature store:** AP > CP eventual consistency, `.shift(1)` leakage prevention

## Domain Knowledge
- 10 segments: 5 concerns × 2 textures (Concern × Texture)
- Temperature switch point: 23.5°C (Lightweight vs Rich demand)
- Social lag: T-3 (social signals lead sales by 3 days)
- Shelf life: 450 days (Brightening/Light) to 910 days (Hydrating/Rich)
- Treatment ratio: 20/80 for uplift (why X-Learner wins)

## Test Organization
- Data tests: test_config, test_segment_genes, test_data_generator, test_contracts
- Model tests: test_forecasting_models, test_hierarchy, test_evaluation, test_ml_leakage
- Infrastructure tests: test_sql_pipelines, test_feature_store, test_drift_monitor, test_retrain_trigger
- Experimentation tests: test_cuped, test_sequential, test_interleaving, test_power, test_bucketing
- Causal tests: test_dowhy, test_uplift
- Explain tests: test_shap_lime, test_fairness, test_property_based, test_mlflow_tracker
