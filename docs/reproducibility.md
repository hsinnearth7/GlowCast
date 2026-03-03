# Reproducibility Protocol

> Reference: Pineau et al., "ML Reproducibility Checklist", NeurIPS 2019

## 1. Environment Lockfile

```bash
pip install -e ".[dev]"
# All dependency versions pinned via >= min, < next_major in pyproject.toml
# Docker image: python:3.11-slim (deterministic base)
```

## 2. Randomness Control

| Source | Seed Mechanism | Notes |
|--------|---------------|-------|
| Python `random` | `random.seed(42)` | Via `set_global_seed()` |
| NumPy | `np.random.seed(42)` + `default_rng(42)` | Generator-based RNG preferred |
| Scikit-learn | `random_state=42` in all estimators | Enforced by factory pattern |
| LightGBM | `seed=42` param | Passed via YAML config |
| XGBoost | `random_state=42` param | Passed via YAML config |
| PYTHONHASHSEED | `os.environ["PYTHONHASHSEED"] = "42"` | Set in `seed.py` |
| Data generation | `np.random.default_rng(42)` | Single RNG instance per generator |

## 3. Data Version Control

```bash
# Verify data reproducibility
python -m app.data.data_generator --validate-only
# SHA-256 hash printed to stdout; compare with expected value
```

The `GlowCastDataGenerator.compute_data_hash()` method produces a deterministic SHA-256 hash of all 9 star schema tables. Any change in data generation logic or parameters will change the hash.

## 4. Experiment Tracking

Each run logs:
- **Parameters:** model hyperparameters, training window, features used
- **Metrics:** MAPE, RMSE, WMAPE, coverage, Wilcoxon p-value, Cohen's d
- **Artifacts:** trained model, feature importance, conformal calibration quantile
- **Environment:** Python version, package versions, platform

Tracked via MLflow (or local JSON fallback). Run ID format: `YYYY-MM-DD_HH-MM-SS_{model_name}`.

## 5. One-Command Reproduce

```bash
git clone https://github.com/username/glowcast.git
cd glowcast
docker compose build && docker compose up -d

# Or without Docker:
pip install -e ".[dev]"
python -m app.data.data_generator --validate-only
pytest tests/ -v
```

## 6. Known Sources of Non-Determinism

| Source | Mitigation |
|--------|-----------|
| Float precision across platforms | Validation tolerance: MAPE within ±0.1pp |
| LightGBM parallel training | `num_threads=1` for exact reproducibility; default multi-thread for speed |
| Dictionary ordering | Python 3.7+ guarantees insertion order |
| Pandera validation order | Schema checks are order-independent |
| SQLite query optimizer | EXPLAIN ANALYZE documented; results are deterministic |

## 7. Validation Tolerance

- **MAPE:** Reproduced results must be within ±0.1 percentage points
- **P-values:** Wilcoxon/KS tests must agree at α = 0.05 significance level
- **Data hash:** Must match exactly (byte-level) on same platform
- **Coverage:** Conformal prediction coverage within ±1pp of target

## References

- Pineau, J., et al. (2019). "The Machine Learning Reproducibility Checklist." NeurIPS.
