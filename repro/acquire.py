"""Multi-criteria acquisition function S(u) (Equation 7) and its three
components for the full Algorithm 1:

  v_u = Var over the CATE ensemble {tau_j(u)}    (epistemic uncertainty, Eq. 5)
  d_u = sigmoid(g_xi(phi(u)))                    (domain discrepancy, Eq. 6)
  o_u = 2 * |e_obs(phi(u)) - 0.5|                (overlap deficit)

Rank-normalized combination:
  S(u) = alpha * eta(v_u) + beta * eta(d_u) + gamma * eta(o_u),
  eta(a_u) = (1/|pool|) sum_{u'} 1{a_{u'} <= a_u}  (fractional rank in [0,1]).
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor


def fractional_rank(a: np.ndarray) -> np.ndarray:
    """eta(a) = fraction of pool with value <= a_u, in [0,1] (Eq. 7 rank map)."""
    order = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    return (order + 1.0) / a.size


# --------------------------------------------------------------------------- #
#  Component models
# --------------------------------------------------------------------------- #
class PropensityModel:
    """e_obs(phi(x)) ~ P(T_obs=1 | phi(x)), fit on the observational log."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=400, C=1.0)

    def fit(self, Phi_obs, T_obs):
        self.clf.fit(Phi_obs, T_obs)
        return self

    def predict_proba(self, Phi):
        return self.clf.predict_proba(Phi)[:, 1]


class DomainClassifier:
    """g_xi distinguishing pool (label 1) from current training set (label 0).

    d_u = sigmoid(g_xi) = P(domain=1 | phi(u)).  The logit approximates
    log(p_pool(phi)/p_current(phi)) -- a density-ratio estimator (Sec. 3)."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=400, C=1.0)

    def fit(self, Phi_pool, Phi_current):
        X = np.vstack([Phi_current, Phi_pool])
        y = np.concatenate([np.zeros(Phi_current.shape[0]), np.ones(Phi_pool.shape[0])])
        self.clf.fit(X, y)
        return self

    def predict_proba(self, Phi):
        return self.clf.predict_proba(Phi)[:, 1]


class CATEEnsemble:
    """Ensemble of E small regressors fit on bootstrap sub-samples of the RCT
    pseudo-outcomes.  Disagreement v_u = Var_j{tau_j(u)} (Eq. 5).

    (The paper uses MC-Dropout; Deep Ensembles are the cited alternative and
    are more reproducible on CPU.  Both target the same epistemic variance.)
    """

    def __init__(self, n_estimators: int = 8, hidden: tuple = (32,), seed: int = 0):
        self.n_estimators = n_estimators
        self.hidden = hidden
        self.seed = seed
        self.models: list = []

    def fit(self, Phi, Ytilde):
        rng = np.random.default_rng(self.seed)
        n = Phi.shape[0]
        self.models = []
        for j in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # bootstrap
            m = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu",
                             solver="adam", max_iter=200, random_state=int(self.seed + j))
            # guard: need >1 distinct target in bootstrap for MLP convergence
            if np.unique(Ytilde[idx]).size < 2:
                m.fit(Phi, Ytilde)  # fall back to full data
            else:
                m.fit(Phi[idx], Ytilde[idx])
            self.models.append(m)
        return self

    def predict_ensemble(self, Phi) -> np.ndarray:
        """Return (E, n) array of per-estimator CATE predictions."""
        return np.stack([m.predict(Phi) for m in self.models], axis=0)

    def uncertainty(self, Phi) -> np.ndarray:
        """v_u = Var over ensemble (Eq. 5)."""
        preds = self.predict_ensemble(Phi)
        return preds.var(axis=0)


# --------------------------------------------------------------------------- #
#  Acquisition S(u) (Equation 7)
# --------------------------------------------------------------------------- #
def acquisition_score(v: np.ndarray, d: np.ndarray, o: np.ndarray,
                      alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """S(u) = alpha*eta(v) + beta*eta(d) + gamma*eta(o)  (Eq. 7)."""
    return alpha * fractional_rank(v) + beta * fractional_rank(d) + gamma * fractional_rank(o)


def overlap_deficit(e_obs_pool: np.ndarray) -> np.ndarray:
    """o_u = 2*|e_obs(phi(u)) - 0.5|  (overlap deficit, targets positivity violations)."""
    return 2.0 * np.abs(e_obs_pool - 0.5)
