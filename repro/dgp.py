"""Data-generating processes (DGPs) for the budgeted active experimentation repro.

Two families are provided:

1. ``SegmentDGP`` -- the discrete "d segment" construction used both by the
   theory (Assumptions 4.1-4.3 hold exactly) and by the minimax hard instance
   (Eqs. 40-41).  d covariate types x^(j) with phi(x^(j)) = e_j, P_X = 1/d,
   binary potential outcomes Y(t)|x^(j) ~ Bern(1/2 +- theta_j/2).  This is the
   setting in which adaptive *covariate selection* is most transparent: the
   learner chooses WHICH segment to run an RCT on based on past outcomes.

2. ``ContinuousOBSRCDGP`` -- an industrial-style continuous-covariate OBS+pool
   setting with a (near-)deterministic historical propensity e_obs(x), used to
   exercise the full Algorithm 1 multi-criteria acquisition (v_u / d_u / o_u).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Discrete segment DGP (theory + minimax hard instance, Eqs. 40-41)
# --------------------------------------------------------------------------- #
class SegmentDGP:
    """d segments, phi(x^(j)) = e_j, tau(x^(j)) = theta_j.

    Potential outcomes Y(1)|x^(j) ~ Bern(1/2 + theta_j/2),
                         Y(0)|x^(j) ~ Bern(1/2 - theta_j/2),
    exactly the hard-instance family of Theorem 4.9 / Eq. (41).  |theta_j| < 1
    keeps the Bernoulli parameters in (0,1).
    """

    def __init__(self, theta_star: np.ndarray, seed: int = 0):
        self.theta_star = np.asarray(theta_star, dtype=float)
        self.d = self.theta_star.size
        assert np.all(np.abs(self.theta_star) < 1.0), "need |theta_j| < 1 for valid Bernoulli"
        self.rng = np.random.default_rng(seed)

    @property
    def mu1(self) -> np.ndarray:  # P(Y(1)=1 | x^(j))
        return 0.5 + self.theta_star / 2.0

    @property
    def mu0(self) -> np.ndarray:  # P(Y(0)=1 | x^(j))
        return 0.5 - self.theta_star / 2.0

    def feature(self, j: int) -> np.ndarray:
        e = np.zeros(self.d)
        e[j] = 1.0
        return e

    def all_features(self) -> np.ndarray:  # (d, d) identity matrix
        return np.eye(self.d)

    def draw_pool(self, n_pool: int) -> np.ndarray:
        """Draw n_pool segment indices ~ uniform (target marginal P_X = 1/d)."""
        return self.rng.integers(0, self.d, size=n_pool)

    def sample_rct(self, seg_idx: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run randomized experiments on the given (already-selected) segments.

        Parameters
        ----------
        seg_idx : array of segment indices in {0,...,d-1} (the adaptive choice).
        p       : array of known randomization probabilities in (0,1).

        Returns (T, Y, p) where Y is the observed binary outcome.
        """
        T = (self.rng.random(seg_idx.size) < p).astype(float)
        # draw potential outcomes for the selected segments
        mu_t = np.where(T > 0.5, self.mu1[seg_idx], self.mu0[seg_idx])
        Y = (self.rng.random(seg_idx.size) < mu_t).astype(float)
        return T, Y, p

    def tau(self, seg_idx: np.ndarray) -> np.ndarray:
        return self.theta_star[seg_idx]


# --------------------------------------------------------------------------- #
#  Continuous OBS+pool DGP for the full Algorithm 1 (Claim 1)
# --------------------------------------------------------------------------- #
class ContinuousOBSRCDGP:
    """Industrial-style OBS + unlabeled-pool setting.

    Covariates X ~ N(0, I_d).  The *target* CATE is linear-realizable in a
    feature map phi(x):

        tau(x) = <theta*, phi(x)>,   phi(x) = [x_1, ..., x_d]   (identity).

    Baseline response mu0(x) = sigmoid(-1 + 0.5 x_1) so outcomes are binary and
    bounded in [0,1]; mu1(x) = sigmoid(-1 + 0.5 x_1 + 2*tau(x)) is then shifted
    so that E[Y(1)-Y(0)|x] != tau(x) in general -- BUT the IPW pseudo-outcome
    remains unbiased for tau(x) under RCT randomization (Lemma 4.4), which is
    exactly the point of the paper (OBS shapes design; RCT identifies).

    The observational propensity is a (near-)deterministic historical policy
    e_obs(x) = sigmoid(3*(x_1 + x_2)) clipped into [eps, 1-eps]; this creates
    the overlap deficit the acquisition's o_u term targets.
    """

    def __init__(self, d: int = 6, theta_star: np.ndarray | None = None,
                 obs_bias: float = 3.0, seed: int = 0):
        self.d = d
        if theta_star is None:
            theta_star = np.zeros(d)
            theta_star[:3] = [0.4, -0.3, 0.2]
        self.theta_star = np.asarray(theta_star, dtype=float)
        self.obs_bias = obs_bias
        self.rng = np.random.default_rng(seed)

    # -- feature map phi(x) = x (identity); replaceable for richer encoders -- #
    def phi(self, X: np.ndarray) -> np.ndarray:
        return X

    def tau(self, X: np.ndarray) -> np.ndarray:
        return self.phi(X) @ self.theta_star

    def _mu(self, X: np.ndarray, t: int) -> np.ndarray:
        base = 1.0 / (1.0 + np.exp(-(1.0 + 0.5 * X[:, 0])))  # mu0 baseline, in (0,1)
        # shift treatment mean by tau but keep in [0,1] via a second sigmoid
        logit = np.log(base / (1.0 - base + 1e-12)) + t * 2.0 * self.tau(X)
        logit = np.clip(logit, -30, 30)
        return 1.0 / (1.0 + np.exp(-logit))

    def e_obs(self, X: np.ndarray, eps: float = 0.02) -> np.ndarray:
        """Historical (biased, near-deterministic) observational propensity."""
        raw = 1.0 / (1.0 + np.exp(-self.obs_bias * (X[:, 0] + X[:, 1])))
        return np.clip(raw, eps, 1.0 - eps)

    def draw_target_X(self, n: int) -> np.ndarray:
        return self.rng.standard_normal((n, self.d))

    def draw_observational(self, n_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw observational log under the biased historical policy.

        To induce selection bias, X_obs is sampled from a SHIFTED distribution
        (conditional on the policy region), mimicking support mismatch.
        """
        X = self.rng.standard_normal((n_obs, self.d))
        # shift the covariate marginal so supp(P_Xobs) != supp(P_X) in the tails
        X[:, 0] += 0.8
        X[:, 1] += 0.6
        e = self.e_obs(X)
        T = (self.rng.random(n_obs) < e).astype(float)
        mu_t = np.where(T > 0.5, self._mu(X, 1), self._mu(X, 0))
        Y = (self.rng.random(n_obs) < mu_t).astype(float)
        return X, T, Y

    def run_rct(self, X: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Randomized experiment on selected covariates X with known probs p."""
        T = (self.rng.random(X.shape[0]) < p).astype(float)
        mu_t = np.where(T > 0.5, self._mu(X, 1), self._mu(X, 0))
        Y = (self.rng.random(X.shape[0]) < mu_t).astype(float)
        return T, Y
