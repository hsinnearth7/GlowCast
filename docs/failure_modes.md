# Failure Modes & Graceful Degradation

> 5 components × 5 degradation levels (L0–L4)

## Degradation Levels

| Level | Name | Description |
|-------|------|-------------|
| L0 | Normal | All systems operational |
| L1 | Warning | Monitoring alert, no user impact |
| L2 | Degraded | Reduced accuracy, fallback active |
| L3 | Partial Outage | Some functionality unavailable |
| L4 | Full Outage | Component completely unavailable |

---

## Component 1: Feature Store

| Level | Trigger | Behavior | Recovery |
|-------|---------|----------|----------|
| L0 | — | Fresh features (< 1h old) served | — |
| L1 | Offline refresh latency > 30min | Alert; online store serves stale features | Monitor |
| L2 | Offline refresh fails | Online store serves 24h-stale features; MAPE degrades ~0.3pp | Auto-retry in 1h |
| L3 | Online store unreachable | Fall back to last-known feature cache (in-memory) | Restart service |
| L4 | Both stores down | Use hardcoded population-mean features | Page on-call; manual restart |

## Component 2: Forecasting Models

| Level | Trigger | Behavior | Recovery |
|-------|---------|----------|----------|
| L0 | — | Routing ensemble (LightGBM/SARIMAX/Chronos) | — |
| L1 | Single sub-model slow (>100ms) | Log warning; continue with remaining models | Auto-heal |
| L2 | LightGBM fails | Route all mature SKUs to SARIMAX; MAPE +3pp | Retrain LightGBM |
| L3 | All ML models fail | Fall back to NaiveMA(30); MAPE ~25% | Investigate + redeploy |
| L4 | Inference API down | Return last cached forecast (stale) | Restart + page on-call |

## Component 3: Social Signal Pipeline

| Level | Trigger | Behavior | Recovery |
|-------|---------|----------|----------|
| L0 | — | Real-time social ingestion, T-3 lag features | — |
| L1 | Single source down (e.g., TikTok API) | Reweight remaining sources | Monitor; auto-retry |
| L2 | 3+ sources down | Social features frozen at last known values | Alert; manual check |
| L3 | All sources down > 24h | Social sensitivity set to 0; pure historical demand | Escalate to data team |
| L4 | Social pipeline corrupted | Quarantine social features; models use non-social features only | Rebuild pipeline |

## Component 4: Drift Monitor

| Level | Trigger | Behavior | Recovery |
|-------|---------|----------|----------|
| L0 | — | All drift checks pass | — |
| L1 | Data drift on 1-2 features (KS p < 0.05) | Log + alert; no action | Investigate feature |
| L2 | Prediction drift (PSI > 0.1) | Alert; schedule early retrain | Retrain within 24h |
| L3 | Concept drift (MAPE > 20% for 7 days) | Auto-trigger retrain | Verify retrain success |
| L4 | Monitor itself fails | Silent degradation — no drift detection | Page on-call; restart monitor |

## Component 5: SQL Analytics Pipeline

| Level | Trigger | Behavior | Recovery |
|-------|---------|----------|----------|
| L0 | — | All 5 pipelines execute < 30s | — |
| L1 | Single pipeline slow (> 60s) | Log warning; results still valid | Optimize query |
| L2 | Single pipeline fails | Skip failed pipeline; dashboard shows stale data | Fix SQL + rerun |
| L3 | SQLite database corrupted | Regenerate from Parquet/CSV source files | Run data generator |
| L4 | All pipelines fail | Dashboard unavailable; forecasting unaffected | Rebuild database |

---

## Cross-Component Cascade

| Root Failure | Cascade Impact | Mitigation |
|-------------|----------------|------------|
| Feature store L4 | Forecasting degrades to L2-L3 | Population-mean fallback features |
| Social pipeline L3 | Feature store missing social features → Forecasting L1 | Social weight → 0 |
| Drift monitor L4 | No retrain triggers → potential concept drift accumulation | Scheduled weekly retrain as safety net |
| Data generator failure | No new training data → model staleness | Cache last successful dataset |
