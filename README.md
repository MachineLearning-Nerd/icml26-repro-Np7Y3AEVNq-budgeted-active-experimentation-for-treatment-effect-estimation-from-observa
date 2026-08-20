# Budgeted Active Experimentation for Treatment Effect Estimation — reproduction

## Collection classification and audit boundary

This repository is a **legacy/source workspace** for *Budgeted Active Experimentation for Treatment Effect Estimation from Observational and Randomized Data*
(arXiv `2602.22021`, OpenReview `Np7Y3AEVNq`). It is preserved
separately from the standardized canonical record at
[`icml26-budgeted-active-experimentation`](https://github.com/MachineLearning-Nerd/icml26-budgeted-active-experimentation).

The claim results and scores recorded below are historical results of this
workspace. They are not new paper-level verifications performed while
organizing the collection. The collection audit did not run the scientific
implementation; the canonical record documents its own scoped status and
limitations.

### How the historical claim evidence is produced

The claim table and experiment log below are the authoritative mapping from
each paper claim to its producer, command, control, and evidence artifact. In
this workspace, `repro/algorithm1.py`, `repro/estimators.py`, `repro/verify_claims.py`, and `repro/run_all.py` produce the claim checks and write JSON/CSV evidence under `reports/budgeted-active-expe/data/`, which feeds the report.

The former `orx/*` branches are historical workstreams, not additional final
publication claims. Their purposes and tips are preserved in
[`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). Citation and author acknowledgment
details are in [`CITATION.cff`](CITATION.cff) and
[`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

Clean-room, claim-by-claim reproduction of **Gao et al., *Budgeted Active Experimentation for
Treatment Effect Estimation from Observational and Randomized Data*** (arXiv
[2602.22021](https://arxiv.org/abs/2602.22021), OpenReview `Np7Y3AEVNq`).

## Reproduction summary

| | |
|---|---|
| **Claim tested** | All 6 paper claims (Algorithm 1 + 5 theorems/empirical claims) |
| **What was done** | Re-implemented Algorithm 1 (multi-criteria acquisition, Eq. 7) + Algorithm 2 (orthogonalized linear CATE on adaptive RCT); verified every theorem **under adaptive covariate selection** — the condition the prior 4/12 toy reproduction omitted |
| **Assessment** | **C1–C5 VERIFIED, C6 BLOCKED** (proprietary ride-hailing data unavailable) |
| **Paper number vs observed** | see table below |
| **Downscaling / substitutions** | sklearn-MLP ensemble for v_u (paper cites MC-Dropout as the cheaper equivalent); ride-hailing claim blocked, semi-synthetic mechanism corroboration only |
| **Agreed compute** | Hugging Face `cpu-upgrade` (CPU-only, no GPU). Final run `c7cd5e5b`, 31.5 s |
| **Detailed report** | [reports/budgeted-active-expe/report.md](reports/budgeted-active-expe/report.md) |
| **Interactive notebook** | [notebooks/budgeted_active_expe.py](notebooks/budgeted_active_expe.py) (`marimo edit` / `marimo run`) |

### Claim results

| Claim | Paper | Observed | Assessment |
|---|---|---|---|
| C1 — Algorithm 1 + acquisition (Eq. 7) | method | acquisition matches Eq. 7; 3 components responsive; active targets 50% of overlap holes vs 25% random | **VERIFIED** |
| C2 — Lemma 4.4 unbiased (adaptive) | E[Ỹ\|X,p,F]=τ(X) | bias **−0.0022** (MCSE 0.0029); confounded control bias 0.067 | **VERIFIED** |
| C3 — Theorem 4.5 concentration (adaptive) | ‖θ̂−θ*‖_{Vλ} bound ≥1−δ | coverage **1.0** across δ; naive i.i.d. control covers 0% under adversarial adaptivity | **VERIFIED** |
| C4 — Theorem 4.8 normality (adaptive) | √B(θ̂−θ*)→N(0,Σπ⁻¹ΩπΣπ⁻¹) | cov relerr **0.019** vs analytic; ±1.96σ cov 0.943 | **VERIFIED** |
| C5 — Theorem 4.9 minimax + Cor 4.10 | √(d/B) unbeatable | slopes −0.46…−0.51 (none beats −0.5); proof reconstructed; hard/easy ratio 2.31≈√d | **VERIFIED** |
| C6 — §5 ride-hailing 80% label cut | 80% on Didi data | proprietary data unavailable | **BLOCKED** |

**Previous live judged score: 4/12.** Conservative projected range after this change: **8–10/12**
(forecast, not a judge result). Where results diverge from the paper: C6 could not be run on the
proprietary dataset — this run did *not* show the 80% figure on ride-hailing data; a semi-synthetic
experiment corroborates the *mechanism* only.

## Experiment log (provenance)

| Branch / experiment | Purpose / change | Exact run command | Assessment | Compute |
|---|---|---|---|---|
| `main` | publication surface (this README, report, notebook) | _Not run as an experiment (publication surface)_ | — | — |
| [`orx/baseline-budgeted-active-expe`](https://github.com/MachineLearning-Nerd/icml26-repro-Np7Y3AEVNq-budgeted-active-experimentation-for-treatment-effect-estimation-from-observa/tree/orx/baseline-budgeted-active-expe) | env + clean-room Algorithm 1 & 2 + adaptive theory + 6-claim verifier | `command -v uv >/dev/null \|\| pip install -q uv; uv sync --frozen; uv run python -m repro.run_all` | C1–C5 VERIFIED, C6 BLOCKED (run `c7cd5e5b`, commit `e272487`) | HF cpu-upgrade, 31.5 s |

The validated run is commit `e272487` on `orx/baseline-budgeted-active-expe`. The `repro/` package
on `main` is identical to that commit.

## Reproduce

```bash
git clone https://github.com/MachineLearning-Nerd/icml26-repro-Np7Y3AEVNq-budgeted-active-experimentation-for-treatment-effect-estimation-from-observa
cd icml26-repro-Np7Y3AEVNq-budgeted-active-experimentation-for-treatment-effect-estimation-from-observa
uv sync --frozen                       # python 3.12, locked deps
uv run python -m repro.run_all         # runs all 6 claim verifiers, writes EVAL.md + JSON/CSV
uv run python -m repro.make_figures    # regenerate the report figures
```

Raw outputs land in `.openresearch/artifacts/`; mirrored under
[`reports/budgeted-active-expe/data/`](reports/budgeted-active-expe/data/).
