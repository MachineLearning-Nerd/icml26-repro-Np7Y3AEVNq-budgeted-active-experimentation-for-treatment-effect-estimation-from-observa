"""Algorithm 1 -- Active Sampling for OBS-RCT Fusion (full multi-criteria).

Given an observational log D_obs, an unlabeled pool D_pool, a budget B and a
batch size M, this loop iteratively:
  1. fits the three scoring components (v_u, d_u, o_u),
  2. ranks the pool by S(u) = alpha*eta(v) + beta*eta(d) + gamma*eta(o) (Eq. 7),
  3. selects the top-m_k candidates, runs bounded randomization p in [fmin,fmax],
  4. appends the RCT quadruples and removes the queried units from the pool.

The estimator is identified purely from the randomized samples (Algorithm 2);
observational data only shape the design (the paper's central separation).
"""
from __future__ import annotations
import numpy as np
from .dgp import ContinuousOBSRCDGP
from .estimators import LinearCATEEstimator, pseudo_outcome
from .acquire import (PropensityModel, DomainClassifier, CATEEnsemble,
                      acquisition_score, overlap_deficit)


class BudgetedActiveExperimentation:
    """Faithful implementation of Algorithm 1."""

    def __init__(self, dgp: ContinuousOBSRCDGP, fmin: float = 0.2, fmax: float = 0.8,
                 M: int = 40, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 n_estimators: int = 8, seed: int = 0, leverage_uncertainty: bool = False):
        self.dgp = dgp
        self.fmin, self.fmax = fmin, fmax
        self.M = M
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.n_estimators = n_estimators
        self.leverage_uncertainty = leverage_uncertainty
        self.seed = seed

    def _design_probs(self, X_sel: np.ndarray, theta_hat: np.ndarray | None) -> np.ndarray:
        """Covariate-dependent randomization clipped into [fmin, fmax] (Sec. 3.1).

        p_i = Clip(0.5 + 0.3 * tau_hat(X_i), fmin, fmax) -- a mild, known,
        bounded policy that adapts to the current CATE estimate."""
        if theta_hat is None:
            base = np.full(X_sel.shape[0], 0.5)
        else:
            base = 0.5 + 0.3 * (X_sel @ theta_hat)
        return np.clip(base, self.fmin, self.fmax)

    def run(self, D_obs: tuple[np.ndarray, np.ndarray, np.ndarray],
            X_pool: np.ndarray, B: int,
            record_components: bool = True) -> dict:
        X_obs, T_obs, Y_obs = D_obs
        Phi_obs = self.dgp.phi(X_obs)
        pool = X_pool.copy()
        Phi_pool = self.dgp.phi(pool)
        # collected RCT data
        Xr, Tr, Yr, pr = [], [], [], []
        comp_log = []
        queried = 0
        theta_hat = None
        rnd = np.random.default_rng(self.seed)
        rnd_prop = PropensityModel().fit(Phi_obs, T_obs)

        while queried < B:
            m = min(self.M, B - queried)
            Phi_current = np.vstack([Phi_obs] + ([self.dgp.phi(np.array(Xr))] if Xr else []))

            # --- 1. update scoring components ---
            Yt_rct = pseudo_outcome(np.array(Tr), np.array(Yr), np.array(pr)) if Xr else None
            if Yt_rct is not None and len(Yt_rct) >= self.n_estimators:
                Phi_rct = self.dgp.phi(np.array(Xr))
                if self.leverage_uncertainty:
                    # v_u = phi^T V^{-1} phi (Corollary 4.6: linear predictive variance)
                    V = Phi_rct.T @ Phi_rct + 1e-3 * np.eye(Phi_rct.shape[1])
                    Vinv = np.linalg.inv(V)
                    v = np.einsum("ij,jk,ik->i", Phi_pool, Vinv, Phi_pool)
                    theta_hat = np.linalg.solve(Phi_rct.T @ Phi_rct + 1e-3 * np.eye(Phi_rct.shape[1]),
                                                Phi_rct.T @ Yt_rct)
                else:
                    ens = CATEEnsemble(n_estimators=self.n_estimators, seed=self.seed + queried).fit(
                        Phi_rct, Yt_rct)
                    v = ens.uncertainty(Phi_pool)
                    theta_hat = np.linalg.lstsq(Phi_rct, Yt_rct, rcond=None)[0]
            else:
                # cold-start: uniform uncertainty, no RCT yet
                v = np.zeros(Phi_pool.shape[0])

            dom = DomainClassifier().fit(Phi_pool, Phi_current)
            d = dom.predict_proba(Phi_pool)
            e_obs_pool = rnd_prop.predict_proba(Phi_pool)
            o = overlap_deficit(e_obs_pool)

            S = acquisition_score(v, d, o, self.alpha, self.beta, self.gamma)

            if record_components:
                comp_log.append({"queried": queried, "v": v.copy(), "d": d.copy(),
                                 "o": o.copy(), "S": S.copy()})

            # --- 3. select top-m & experiment ---
            top = np.argsort(-S)[:m]
            X_sel = pool[top]
            p_sel = self._design_probs(X_sel, theta_hat)
            T_sel, Y_sel = self.dgp.run_rct(X_sel, p_sel)
            Xr.extend(X_sel.tolist()); Tr.extend(T_sel.tolist())
            Yr.extend(Y_sel.tolist()); pr.extend(p_sel.tolist())
            queried += m
            # remove queried from pool
            mask = np.ones(pool.shape[0], dtype=bool)
            mask[top] = False
            pool = pool[mask]
            Phi_pool = self.dgp.phi(pool)

        Xr = np.array(Xr); Tr = np.array(Tr); Yr = np.array(Yr); pr = np.array(pr)
        return {
            "X": Xr, "T": Tr, "Y": Yr, "p": pr,
            "Phi": self.dgp.phi(Xr),
            "components": comp_log,
            "remaining_pool": pool,
        }
