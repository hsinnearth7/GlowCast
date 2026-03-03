# Latency Budget

> Target: P99 < 200ms end-to-end for single-SKU forecast request

## Budget Breakdown

| Component | P50 (ms) | P99 (ms) | Notes |
|-----------|----------|----------|-------|
| Feature lookup (online store) | 5 | 20 | In-memory dict lookup; cache hit |
| Model inference (LightGBM) | 2 | 5 | Single-row prediction; pre-loaded model |
| Routing logic | 1 | 2 | Deterministic rule evaluation |
| Hierarchical reconciliation | 10 | 30 | MinTrace for single SKU in hierarchy |
| Serialization + network | 15 | 40 | JSON response + HTTP overhead |
| Buffer | — | 103 | Headroom for GC pauses, cold cache |
| **Total** | **33** | **200** | — |

## Component Details

### Feature Lookup (20ms P99)
- Online feature store is an in-memory Python dict
- Cache miss triggers offline store read (~50ms) — counted as degraded (L1)
- Features pre-computed: 12 demand features + 3 social + 2 climate = 17 features

### Model Inference (5ms P99)
- LightGBM `predict()` on single row: ~2ms
- SARIMAX `forecast(1)`: ~8ms (fits within routing to LightGBM for mature SKUs)
- Chronos-2 (cold start): ~50ms — acceptable because cold-start SKUs have lower SLA

### Routing Logic (2ms P99)
- Two threshold checks: `history_days < 60` and `cv > 1.5`
- No ML inference; pure conditional logic

### Hierarchical Reconciliation (30ms P99)
- MinTrace reconciliation for a single SKU within its hierarchy path
- Pre-computed summing matrix S cached at startup
- Full reconciliation (all 5,000 SKUs): ~2s batch, not on critical path

### Serialization + Network (40ms P99)
- JSON serialization of forecast response: ~5ms
- Network round-trip (same datacenter): ~35ms

## Scaling Considerations

| Scenario | Expected Latency | Strategy |
|----------|-----------------|----------|
| Single SKU | < 200ms | Direct inference |
| Batch (100 SKUs) | < 500ms | Vectorized prediction |
| Full reforecast (5,000 SKUs) | < 30s | Parallel batch job |
| Real-time dashboard | < 1s | Pre-computed + cache |

## Monitoring

- Latency percentiles (P50, P95, P99) tracked per component
- Alert threshold: P99 > 180ms (90% of budget)
- Degradation action: if P99 > 200ms for 5 minutes, disable hierarchical reconciliation (saves 30ms)
