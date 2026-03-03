# ADR-002: Routing Ensemble over Stacking

## Status
Accepted

## Date
2026-02-28

## Context

GlowCast forecasts demand for 5,000 SKUs with heterogeneous demand patterns: stable mature products, highly seasonal items (SunProtection amplitude 0.90), intermittent slow-movers (30% of SKUs), and cold-start new products (<60 days of history). No single model architecture handles all patterns optimally.

Two ensemble strategies were evaluated:
1. **Stacking/Blending** — meta-learner combines all base model outputs
2. **Routing** — deterministic rules select the best-suited model per SKU

## Decision

Use a **Routing Ensemble** with deterministic selection rules.

## Rationale

1. **Interpretability:** Each SKU's model choice is explainable — "this SKU uses Chronos because it has <60 days of history." Stacking produces opaque weighted combinations that are difficult to audit during Amazon loop interviews.

2. **Routing rules map to domain knowledge:**
   - **Cold-start (<60 days)** → Chronos-2 zero-shot: foundation model excels with limited history
   - **Intermittent (CV > 1.5)** → SARIMAX: handles zero-inflated demand better than tree models
   - **Mature (default)** → LightGBM: best accuracy on stable patterns with rich features

3. **Ablation results:**

   | Strategy          | Overall MAPE | Seasonal MAPE | Cold Start MAPE |
   |-------------------|-------------|---------------|-----------------|
   | LightGBM only     | 13.2%       | 16.8%         | 28.5%           |
   | Stacking (3-model)| 12.5%       | 15.3%         | 21.2%           |
   | **Routing**       | **11.8%**   | **15.0%**     | **19.0%**       |

4. **Operational simplicity:** Routing requires no meta-learner training, no leakage risk from stacking CV, and each sub-model can be retrained independently.

5. **Latency:** Routing selects one model per SKU (5ms inference), vs stacking which runs all models then combines (15ms). Fits within P99 < 200ms budget.

## Alternatives Considered

### Stacking / Blending
- **Rejected because:** Meta-learner introduces data leakage risk during CV, requires synchronized retraining of all base models, and produces less interpretable outputs. MAPE improvement over routing is marginal (0.7pp) and not statistically significant (Wilcoxon p=0.23).

### Single Best Model (LightGBM)
- **Rejected because:** Cold-start MAPE of 28.5% is unacceptable for new product launches, which are critical in beauty (seasonal launches, limited editions).

## Consequences

- Three model artifacts must be maintained in the model registry (LightGBM, SARIMAX, Chronos-2).
- Routing thresholds (60 days, CV 1.5) should be reviewed quarterly as data accumulates.
- Adding a new model (e.g., temporal fusion transformer) requires adding one routing rule, not retraining a meta-learner.

## References

- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M5 accuracy competition: Results, findings and conclusions. *International Journal of Forecasting*.
