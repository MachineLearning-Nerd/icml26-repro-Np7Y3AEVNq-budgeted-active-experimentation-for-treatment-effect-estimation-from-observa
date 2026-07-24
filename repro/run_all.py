"""Entrypoint: run every claim verifier, dump raw artifacts, write EVAL.md.

Usage:  uv run python -m repro.run_all [--smoke]
Exit code is nonzero if any claim whose verdict is VERIFIED/FALSIFIED fails its
internal assertion (BLOCKED claims never fail the run).
"""
from __future__ import annotations
import os, sys, json, time, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")  # keep logs clean (sklearn MLP ConvergenceWarning etc.)

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
ART = os.path.join(REPO, ".openresearch", "artifacts")
os.makedirs(ART, exist_ok=True)

from . import verify_claims as V

CLAIMS = [
    ("claim1_algorithm1", V.claim1),
    ("claim2_unbiased", V.claim2),
    ("claim3_concentration", V.claim3),
    ("claim4_normality", V.claim4),
    ("claim5_minimax", V.claim5),
    ("claim6_ridehailing", V.claim6),
]

SMOKE = {
    "seed": 0,
    "claim1_B": 160,
    "claim2_trials": 120, "claim2_B": 120, "claim2_d": 4,
    "claim3_trials": 150, "claim3_B": 160, "claim3_d": 3, "claim3_lam": 1.0,
    "claim4_trials": 300, "claim4_B": 300, "claim4_Bsmall": 24, "claim4_d": 2,
    "claim5_trials": 8, "claim5_d": 3, "claim5_Bs": [50, 100, 200, 400],
    "claim6_Bs": [400, 800], "claim6_seeds": 1,
}

FULL = {
    "seed": 0,
    "claim1_B": 480,
    "claim2_trials": 1000, "claim2_B": 240, "claim2_d": 5,
    "claim3_trials": 800, "claim3_B": 360, "claim3_d": 4, "claim3_delta": 0.1, "claim3_lam": 1.0,
    "claim4_trials": 3000, "claim4_B": 700, "claim4_d": 2,
    "claim5_trials": 60, "claim5_d": 4, "claim5_Bs": [50, 100, 200, 400, 800],
    "claim6_Bs": [1000, 2000, 4000, 8000], "claim6_seeds": 3,
}


def _jsonable(o):
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(x) for x in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else FULL

    t0 = time.time()
    summary = {}
    fail = False
    for name, fn in CLAIMS:
        tc = time.time()
        print(f"=== {name} ===", flush=True)
        try:
            r = fn(cfg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"verdict": "BLOCKED", "error": str(e), "contract": name}
            fail = True
        r["runtime_s"] = round(time.time() - tc, 1)
        r["config"] = {k: v for k, v in cfg.items() if k.startswith(name.split("_")[0]) or k == "seed"}
        with open(os.path.join(ART, f"{name}.json"), "w") as f:
            json.dump(_jsonable(r), f, indent=2, default=str)
        summary[name] = {"verdict": r.get("verdict"), "runtime_s": r["runtime_s"]}
        # dump claim5/6 raw rows to CSV
        if "raw_rows" in r:
            import pandas as pd
            pd.DataFrame(r["raw_rows"]).to_csv(os.path.join(ART, f"{name}_raw.csv"), index=False)
        if "corroboration" in r:
            import pandas as pd
            pd.DataFrame(r["corroboration"]).to_csv(os.path.join(ART, f"{name}_corroboration.csv"), index=False)
        print(f"  -> {r.get('verdict')}  ({r['runtime_s']}s)", flush=True)
        # print the headline metrics so they are captured in the run log
        headline_keys = {
            "claim1_algorithm1": ["structural_match_eq7", "component_responsive",
                                  "end_to_end_finite_estimate", "ablation_active_targets_overlap",
                                  "overlap_corr_o_vs_abs", "domain_corr_d_vs_dist",
                                  "queried_high_overlap_deficit_frac", "theta_hat"],
            "claim2_unbiased": ["grand_bias", "mc_se", "per_segment_bias", "unbiased_under_adaptive",
                                "control_confounded_bias"],
            "claim3_concentration": ["main_coverage", "target_coverage", "calibration_by_delta",
                                     "calibrated_rhs_over_lhs_ratio", "median_lhs",
                                     "control_adversarial_selfnorm_pointwise_cov",
                                     "control_adversarial_naive_iid_pointwise_cov"],
            "claim4_normality": ["empirical_mean_largeB", "excess_kurtosis_largeB", "skew_largeB",
                                 "cov_relerr_largeB_vs_analytic", "cov_relerr_largeB_vs_sandwich",
                                 "sweep_relerr_vs_analytic_by_B", "sweep_maxabs_kurtosis_by_B"],
            "claim5_minimax": ["proof_reconstruction_ok", "slopes", "efficient_estimator_slope",
                               "no_estimator_beats_rate", "estimator_matches_rate",
                               "control_easy_family_slope", "delta_choice"],
            "claim6_ridehailing": ["verdict", "blocker", "corroboration", "active_smallB_auuc"],
        }.get(name, [])
        print("   " + json.dumps(_jsonable({k: r.get(k) for k in headline_keys if k in r}),
                                 default=str), flush=True)
        if r.get("verdict") in ("VERIFIED", "FALSIFIED"):
            # explicit re-check: the verdict must be backed by a truthy internal flag
            for k in ("unbiased_under_adaptive", "bound_holds_at_advertised_prob",
                      "proof_reconstruction_ok", "end_to_end_finite_estimate",
                      "structural_match_eq7", "mean_near_zero"):
                if k in r and r[k] is False:
                    fail = True; print(f"  !! internal check {k}=False", flush=True)

    total = time.time() - t0
    out = {"project": "icml26-repro-Np7Y3AEVNq", "paper": "arXiv 2602.22021",
           "total_runtime_s": round(total, 1), "config": "smoke" if args.smoke else "full",
           "claims": summary}
    with open(os.path.join(ART, "verdict.json"), "w") as f:
        json.dump(_jsonable(out), f, indent=2)

    _write_eval(summary, total, args.smoke)
    print(json.dumps(summary, indent=2))
    print(f"\nTOTAL {total:.1f}s  artifacts -> {ART}")
    sys.exit(1 if fail else 0)


def _write_eval(summary, total, smoke):
    lines = ["# EVAL.md — reproduction verdicts\n",
             f"Paper: arXiv 2602.22021 (Budgeted Active Experimentation).  "
             f"Config: **{'smoke' if smoke else 'full'}**.  Total runtime: {total:.0f}s.\n",
             "All theory claims (2-5) use **adaptive covariate selection** (the least-sampled "
             "segment is queried each round, F_{t-1}-measurable) — the condition the prior toy "
             "reproduction omitted.\n",
             "| Claim | Verdict | Runtime |\n|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | **{s['verdict']}** | {s['runtime_s']}s |")
    with open(os.path.join(ART, "EVAL.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
