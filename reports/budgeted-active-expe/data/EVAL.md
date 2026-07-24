# EVAL.md — reproduction verdicts

Paper: arXiv 2602.22021 (Budgeted Active Experimentation).  Config: **full**.  Total runtime: 152s.

All theory claims (2-5) use **adaptive covariate selection** (the least-sampled segment is queried each round, F_{t-1}-measurable) — the condition the prior toy reproduction omitted.

| Claim | Verdict | Runtime |
|---|---|---|
| claim1_algorithm1 | **VERIFIED** | 41.5s |
| claim2_unbiased | **VERIFIED** | 5.0s |
| claim3_concentration | **VERIFIED** | 5.9s |
| claim4_normality | **VERIFIED** | 43.8s |
| claim5_minimax | **VERIFIED** | 15.3s |
| claim6_ridehailing | **BLOCKED** | 40.0s |
