# ADR-003: Feature Store AP > CP (Eventual Consistency)

## Status
Accepted

## Date
2026-02-28

## Context

GlowCast's feature store serves two paths:
1. **Offline (batch):** Training pipeline reads historical features for model fitting (nightly)
2. **Online (real-time):** Inference API reads latest features for per-request predictions (P99 < 200ms)

The CAP theorem forces a trade-off between Consistency (CP) and Availability (AP) when the offline store is being refreshed while the online store is serving predictions.

## Decision

Choose **AP over CP** — the feature store prioritizes availability with eventual consistency (1-day TTL).

## Rationale

1. **Stale features are acceptable in demand forecasting:** A feature computed from yesterday's data (e.g., `rolling_mean_28`) changes by <0.5% day-over-day for stable SKUs. A 24-hour staleness window has negligible impact on forecast accuracy.

2. **Availability is critical:** The inference API must always return a prediction. A CP system that blocks during offline refresh (10-20 minutes for 5,000 SKUs × 12 FCs) would violate the P99 < 200ms SLA.

3. **Training-serving skew is eliminated by design:** Both offline and online paths use the same `_compute_demand_features()` method. The only difference is data recency, not computation logic.

4. **Quantified impact:** A/B test comparing 1-hour-fresh vs 24-hour-fresh features showed no significant difference in MAPE (p=0.82, Wilcoxon signed-rank).

## Alternatives Considered

### CP (Strong Consistency)
- **Rejected because:** Requires read-lock during offline refresh, causing 10-20 minute availability gaps. Unacceptable for production serving.

### Lambda Architecture (dual pipeline)
- **Rejected because:** Maintaining separate batch and streaming compute paths doubles engineering complexity and introduces training-serving skew risk. Overkill for daily-cadence demand forecasting.

## Consequences

- Online features may be up to 24 hours stale. TTL is configurable via `monitoring.feature_store.offline_ttl_hours`.
- Feature freshness is monitored: alert fires if offline refresh fails for >48 hours.
- For social signal features (viral bursts), the T-3 lag already provides a natural buffer — a viral spike at T=0 doesn't affect features until T+3.

## References

- Brewer, E. A. (2000). Towards Robust Distributed Systems. *ACM PODC*.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly.
