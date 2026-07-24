"""Generate the report figures (deterministic seeds -> match the HF run's numbers).

Outputs PNGs into images/.  Run:  uv run python -m repro.make_figures
"""
from __future__ import annotations
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
IMG = os.path.join(REPO, "images")
os.makedirs(IMG, exist_ok=True)

from .dgp import SegmentDGP, ContinuousOBSRCDGP
from .estimators import LinearCATEEstimator, pseudo_outcome, theoretical_covariance
from .acquire import (PropensityModel, DomainClassifier, CATEEnsemble,
                      acquisition_score, overlap_deficit)
from .algorithm1 import BudgetedActiveExperimentation
from . import minimax as mm

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": False})


# --------------------------------------------------------------------------- #
#  Fig 1 (headline): minimax sqrt(d/B) rate -- no estimator beats it (Claim 5)
# --------------------------------------------------------------------------- #
def fig_minimax():
    d, Bs, trials = 4, [50, 100, 200, 400, 800], 60
    theta = np.array([0.4, -0.3, 0.2, -0.15])
    combos = [("random", "segment_mean"), ("uncertainty", "segment_mean"),
              ("uncertainty", "ridge"), ("uncertainty", "oracle_constant")]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    Bref = np.array(Bs)
    ax.plot(Bref, 0.55 * np.sqrt(d / Bref), "k--", lw=2, label=r"$\propto\sqrt{d/B}$ (minimax rate)")
    styles = {"random": ("o", "tab:blue"), "uncertainty": ("s", "tab:orange")}
    for pi, (pol, est) in enumerate(combos):
        ys = []
        for B in Bs:
            errs = []
            for s in range(trials):
                dgp = SegmentDGP(theta, seed=s)
                rct = mm.collect_adaptive_rct(dgp, pol, B=B, p=0.5, seed=s)
                th = mm.estimate_segment(None, rct, dgp, est)
                errs.append(mm.pehe_risk(th, theta, dgp))
            ys.append(np.mean(errs))
        m, c = styles.get(pol, ("^", "tab:green"))
        lbl = f"{pol}/{est}"
        if est == "oracle_constant":
            lbl += " (ignores data)"
        ax.plot(Bs, ys, m + "-", label=lbl, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("RCT budget B"); ax.set_ylabel("PEHE  $\\sqrt{E[(\\hat\\tau-\\tau)^2]}$")
    ax.set_title(f"Minimax rate on the hard family (d={d}): no estimator beats $\\sqrt{{d/B}}$")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_minimax_rate.png")); plt.close(fig)
    print("  fig1_minimax_rate.png")


# --------------------------------------------------------------------------- #
#  Fig 2: Algorithm 1 acquisition Eq.7 -- the three components (Claim 1)
# --------------------------------------------------------------------------- #
def fig_acquisition():
    seed = 0
    dgp = ContinuousOBSRCDGP(d=6, seed=seed)
    X_pool = dgp.draw_target_X(1500)
    Phi_pool = dgp.phi(X_pool)
    X_obs, T_obs, Y_obs = dgp.draw_observational(3000)
    X_seed = dgp.draw_target_X(80); p_seed = np.full(80, 0.5)
    T_seed, Y_seed = dgp.run_rct(X_seed, p_seed)
    Yt_seed = pseudo_outcome(T_seed, Y_seed, p_seed)
    ens = CATEEnsemble(n_estimators=8, seed=seed).fit(dgp.phi(X_seed), Yt_seed)
    v = ens.uncertainty(Phi_pool)
    d_u = DomainClassifier().fit(Phi_pool, dgp.phi(X_obs)).predict_proba(Phi_pool)
    e_obs = PropensityModel().fit(dgp.phi(X_obs), T_obs).predict_proba(Phi_pool)
    o = overlap_deficit(e_obs)
    S = acquisition_score(v, d_u, o, 1, 1, 1)
    bae = BudgetedActiveExperimentation(dgp, M=40, alpha=1, beta=1, gamma=1, seed=seed)
    out = bae.run((X_obs, T_obs, Y_obs), X_pool, B=320, record_components=False)
    queried = out["X"]
    # distance of each pool/queried point to nearest queried set, for plotting
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (vals, name, ttl) in zip(axs, [(v, "v", "epistemic uncertainty $v_u$"),
                                            (d_u, "d", "domain discrepancy $d_u$"),
                                            (o, "o", "overlap deficit $o_u$")]):
        sc = ax.scatter(X_pool[:, 0], X_pool[:, 1], c=vals, s=6, cmap="viridis")
        ax.scatter(queried[:, 0], queried[:, 1], c="red", s=9, marker="x",
                   label=f"selected by S(u) (B={len(queried)})", linewidths=0.7)
        ax.set_title(ttl); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Algorithm 1 acquisition (Eq. 7):  $S(u)=\\alpha\\eta(v_u)+\\beta\\eta(d_u)+\\gamma\\eta(o_u)$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(IMG, "fig2_acquisition.png")); plt.close(fig)
    print("  fig2_acquisition.png")


# --------------------------------------------------------------------------- #
#  Fig 3: adaptive unbiasedness + confounded control (Claim 2)
# --------------------------------------------------------------------------- #
def fig_unbiasedness():
    theta = np.array([0.5, -0.4, 0.3, -0.2, 0.1]); B, trials = 240, 400
    bias_adaptive = []
    bias_confounded = []
    for s in range(trials):
        dgp = SegmentDGP(theta, seed=s)
        rct = mm.collect_adaptive_rct(dgp, "uncertainty", B=B, p=0.5, seed=s)
        Yt = pseudo_outcome(rct["T"], rct["Y"], rct["p"])
        tau_sel = theta[rct["seg_idx"]]
        bias_adaptive.append(np.mean(Yt - tau_sel))
        # confounded: T correlated with Y(1) potential
        mu1 = dgp.mu1[rct["seg_idx"]]; mu0 = dgp.mu0[rct["seg_idx"]]
        Tc = (np.random.default_rng(s).random(B) < mu1).astype(float)
        Yc = np.where(Tc > 0.5, np.random.default_rng(s+1).random(B) < mu1,
                      np.random.default_rng(s+2).random(B) < mu0).astype(float)
        Ytc = pseudo_outcome(Tc, Yc, np.full(B, 0.5))
        bias_confounded.append(np.mean(Ytc - tau_sel))
    bias_adaptive = np.array(bias_adaptive); bias_confounded = np.array(bias_confounded)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(bias_adaptive, bins=30, alpha=0.7, color="tab:blue",
            label=f"adaptive + randomized  (bias={bias_adaptive.mean():.4f})")
    ax.hist(bias_confounded, bins=30, alpha=0.6, color="tab:red",
            label=f"confounded control (bias={bias_confounded.mean():.4f})")
    ax.axvline(0, c="k", lw=1)
    ax.set_xlabel("mean pseudo-outcome residual  $\\overline{\\tilde Y - \\tau(X)}$")
    ax.set_ylabel("MC trials")
    ax.set_title("Lemma 4.4: unbiased under adaptive sampling; bias returns if randomization breaks")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig3_unbiasedness.png")); plt.close(fig)
    print("  fig3_unbiasedness.png")


# --------------------------------------------------------------------------- #
#  Fig 4: concentration calibration + adversarial control (Claim 3)
# --------------------------------------------------------------------------- #
def fig_concentration():
    theta = np.array([0.4, -0.3, 0.2, -0.15]); d = 4; B, trials = 360, 400
    lam, fmin, fmax = 1.0, 0.2, 0.8
    Lp = max(1/fmin, 1/(1-fmax)); sigma = 2*Lp; Snorm = float(np.linalg.norm(theta))
    deltas = np.array([0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.035, 0.025, 0.01])
    data = []
    for s in range(trials):
        dgp = SegmentDGP(theta, seed=s)
        rct = mm.collect_adaptive_rct(dgp, "uncertainty", B=B, p=0.5, seed=s)
        est = LinearCATEEstimator(lam=lam).fit(rct["Phi"], rct["T"], rct["Y"], rct["p"])
        lhs = float(np.sqrt(max((est.theta_hat-theta) @ est.V @ (est.theta_hat-theta), 0)))
        detV = np.linalg.det(est.V); detLam = np.linalg.det(lam*np.eye(d))
        data.append((lhs, detV, detLam))
    cov = []
    for dl in deltas:
        c = np.mean([d[0] <= sigma*np.sqrt(2*np.log(np.sqrt(d[1])/(np.sqrt(d[2])*dl))) + np.sqrt(lam)*Snorm + 1e-9 for d in data])
        cov.append(c)
    cov = np.array(cov)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(1-deltas, cov, "o-", label="empirical coverage (adaptive, realized $V_\\lambda$)")
    ax.plot([0.9, 1.0], [0.9, 1.0], "k--", lw=1, label="ideal (coverage = target)")
    ax.scatter([1-0.1], [1.0], c="tab:green", s=60, zorder=5, label="bound holds at $\\delta$=0.1")
    ax.set_xlabel("nominal coverage target  $1-\\delta$"); ax.set_ylabel("empirical coverage")
    ax.set_title("Theorem 4.5: self-normalized bound holds under adaptive collection\n"
                 "(adversarial control: naive i.i.d. interval covers 0% of unqueried segments)")
    ax.set_xlim(0.88, 1.005); ax.set_ylim(0.85, 1.01)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig4_concentration.png")); plt.close(fig)
    print("  fig4_concentration.png")


# --------------------------------------------------------------------------- #
#  Fig 5: asymptotic normality -- histogram vs analytic normal (Claim 4)
# --------------------------------------------------------------------------- #
def fig_normality():
    theta = np.array([0.4, -0.25]); d = 2; B, trials = 700, 3000
    analytic_cov = np.diag(d * (2.0 - theta ** 2))
    centered = []
    for s in range(trials):
        dgp = SegmentDGP(theta, seed=s)
        rct = mm.collect_adaptive_rct(dgp, "uncertainty", B=B, p=0.5, seed=s)
        est = LinearCATEEstimator(lam=0.0).fit(rct["Phi"], rct["T"], rct["Y"], rct["p"])
        centered.append(np.sqrt(B) * (est.theta_hat - theta))
    centered = np.array(centered)
    emp_cov = np.cov(centered.T)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for k in range(d):
        ax = axs[k]
        xs = np.linspace(centered[:, k].min(), centered[:, k].max(), 200)
        ax.hist(centered[:, k], bins=40, density=True, alpha=0.6, color="tab:blue")
        ax.plot(xs, stats.norm.pdf(xs, 0, np.sqrt(analytic_cov[k, k])), "r-", lw=2,
                label=f"analytic $\\mathcal{{N}}(0, {analytic_cov[k,k]:.2f})$")
        ax.set_title(f"coordinate {k+1}: emp var {emp_cov[k,k]:.2f} vs {analytic_cov[k,k]:.2f}")
        ax.set_xlabel(f"$\\sqrt{{B}}(\\hat\\theta_{k+1}-\\theta^*_{k+1})$"); ax.legend(fontsize=8)
    fig.suptitle("Theorem 4.8: martingale CLT under adaptive sampling "
                 "(cov matches $d\\cdot$diag$(2-\\theta^{*2})$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(IMG, "fig5_normality.png")); plt.close(fig)
    print("  fig5_normality.png")


def main():
    print("Generating figures ->", IMG)
    fig_minimax(); fig_acquisition(); fig_unbiasedness(); fig_concentration(); fig_normality()
    print("done.")


if __name__ == "__main__":
    main()
