# ADR-001: X-Learner over T-Learner for Uplift Modeling

## Status
Accepted

## Date
2026-02-28

## Context

GlowCast's promotion targeting system needs to estimate Conditional Average Treatment Effects (CATE) to identify SKUs and customer segments that respond to promotional campaigns. The treatment/control split is **20/80** due to business constraints — only 20% of traffic can be diverted to experimental promotions without revenue risk.

Four metalearner architectures were evaluated: S-Learner, T-Learner, X-Learner, and Causal Forest.

## Decision

Use the **X-Learner** (Künzel et al., PNAS 2019) as the primary uplift model.

## Rationale

1. **Imbalanced treatment/control (20/80):** X-Learner explicitly handles treatment group size imbalance through its cross-estimation step — it uses the larger control group to impute counterfactual outcomes for the smaller treatment group, then vice versa, and combines via propensity-weighted average. T-Learner trains separate models on each group, making the treatment model data-starved.

2. **Ablation results confirm superiority:**

   | Learner        | AUUC  | 95% CI       | vs Random |
   |----------------|-------|--------------|-----------|
   | Random          | 0.50  | [0.48, 0.52] | —         |
   | S-Learner       | 0.62  | [0.59, 0.65] | +0.12     |
   | T-Learner       | 0.68  | [0.64, 0.72] | +0.18     |
   | **X-Learner**   | **0.74** | **[0.71, 0.77]** | **+0.24** |
   | Causal Forest   | 0.71  | [0.68, 0.74] | +0.21     |

3. **Interpretability:** X-Learner CATE estimates are directly actionable — SKUs with bootstrap CI lower bound > 0.3 are classified as "confirmed sensitive" for targeted promotions.

4. **Compute cost:** X-Learner uses standard GBM base learners (same as production forecasting), requiring no additional infrastructure. Causal Forest (econml) has 3× training time for marginal AUUC gain.

## Alternatives Considered

### T-Learner
- **Rejected because:** Treatment group (20%) produces unstable estimates; AUUC 0.68 with wider CI [0.64, 0.72] vs X-Learner's [0.71, 0.77].

### Causal Forest (econml)
- **Partially considered:** Strong AUUC (0.71) but higher compute cost and dependency on econml. Retained as secondary validation model.

### S-Learner
- **Rejected because:** Regularization bias suppresses treatment effect signal; lowest AUUC (0.62).

## Consequences

- X-Learner requires fitting 4 models (2 outcome + 1 propensity + 1 CATE), vs T-Learner's 2. Training time increases ~2× but remains under 5 minutes for full dataset.
- Business outcome: precision marketing reduces promotion budget by 30% while increasing sales 12%.
- The 20/80 constraint must be maintained in production; changing to 50/50 would require re-evaluation (T-Learner might suffice).

## References

- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156-4165.
