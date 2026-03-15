# CLAUDE.md — GlowCast

## Project Overview
GlowCast is a Cost & Commercial Analytics platform for should-cost modeling, OCOGS tracking, causal inference (DoWhy), A/B testing (CUPED), make-vs-buy analysis, and price elasticity. It simulates 500 SKUs across 12 manufacturing plants with 5 suppliers in 7 countries.

## Tech Stack
- Python 3.11+, pandas, numpy, scipy, scikit-learn
- Pandera (data contracts), structlog (logging), PyYAML (config)
- DoWhy, CausalML, EconML (causal inference)
- SHAP, LIME (explainability)
- MLflow (experiment tracking), Evidently (drift monitoring)
- FastAPI (REST API), Streamlit + Plotly (dashboard)

## Key Commands
```bash
pip install -e ".[dev]"                          # Install
python -m app.data.data_generator --validate-only # Validate data schemas
pytest tests/ -v                                  # Run tests
ruff check app/ tests/                           # Lint
docker compose build && docker compose up -d     # Docker
```

## Architecture Patterns
- **Config:** `app/settings.py` — `load_config()` with `@lru_cache`, YAML-based
- **Logging:** `app/logging.py` — structlog with JSON/console modes
- **Seed:** `app/seed.py` — `GLOBAL_SEED=42`, `set_global_seed()`
- **Cost Modules:** ABC + dataclass pattern in `app/cost/`
- **Data contracts:** Pandera schemas in `app/data/star_schema.py`

## Domain Knowledge
- 5 categories: RawMaterials, Components, Packaging, Labor, Overhead
- 2 cost tiers: Direct, Indirect (10 segments total)
- 5 commodities: Steel, Copper, Resin, Aluminum, Silicon
- 5 suppliers with distinct quality/cost/delivery profiles
- 12 plants across 7 countries (CN, TW, DE, US, MX, IN, JP)
- Treatment ratio: 20/80 for uplift (why X-Learner wins)
- CUPED target rho: 0.74 (55% variance reduction)

## Module Overview
- `app/cost/should_cost.py` — Should-Cost decomposition (BOM → elements → gap)
- `app/cost/ocogs_tracker.py` — OCOGS tracking (actual vs budget variance)
- `app/cost/cost_reduction.py` — Cost reduction recommendations (8 action types)
- `app/cost/make_vs_buy.py` — Make-vs-Buy multi-criteria analysis
- `app/cost/price_elasticity.py` — Log-log elasticity estimation
- `app/causal/dowhy_pipeline.py` — DoWhy 4-step causal inference
- `app/causal/uplift.py` — 4 meta-learner uplift models (S/T/X/CF)
- `app/experimentation/cuped.py` — CUPED variance reduction
- `app/data/data_generator.py` — CostDataGenerator (9-table star schema)

## Development Guidelines
- After editing Python files, verify syntax with `python -m py_compile <file>`.
- Run `pytest tests/ -v` after code changes, not as a final step.
- Check for scalar vs array type mismatches in numpy/pandas operations.
- Always JSON-serialize numpy types before any API response.

## Test Organization
- Data tests: test_config, test_segment_genes, test_data_generator, test_contracts
- Cost module tests: test_should_cost, test_ocogs, test_cost_reduction, test_make_vs_buy, test_price_elasticity
- Infrastructure tests: test_sql_pipelines, test_feature_store, test_drift_monitor, test_retrain_trigger
- Experimentation tests: test_cuped, test_sequential, test_interleaving, test_power, test_bucketing
- Causal tests: test_dowhy, test_uplift
- Explain tests: test_shap_lime, test_fairness, test_property_based, test_mlflow_tracker
