"""Algorithm 2 (orthogonalized linear CATE on adaptive RCT data) and the
theoretical objects: pseudo-outcome, information matrix V_lambda, leverage,
and the sandwich variance Sigma_pi^{-1} Omega_pi Sigma_pi^{-1}.

All routines accept *adaptively collected* data: the covariates X_t and
assignment probs p_t may be F_{t-1}-measurable.  No i.i.d. assumption is used.
"""
from __future__ import annotations
import numpy as np


def pseudo_outcome(T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """IPW pseudo-outcome (Eq. in Alg. 2):  Y~_t = T_t Y_t / p_t - (1-T_t) Y_t / (1-p_t).

    E[Y~_t | X_t, p_t, F_{t-1}] = tau(X_t) under Assumption 4.2 (randomization)."""
    T = np.asarray(T, float)
    Y = np.asarray(Y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return T * Y / p - (1 - T) * Y / (1 - p)


def information_matrix(Phi: np.ndarray, lam: float) -> np.ndarray:
    """V_lambda = lambda I_d + sum_t phi(X_t) phi(X_t)^T  (Eq. 8)."""
    d = Phi.shape[1]
    return lam * np.eye(d) + Phi.T @ Phi


def ridge_estimate(Phi: np.ndarray, Ytilde: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """theta_hat_lambda = V_lambda^{-1} b,  b = sum_t phi(X_t) Y~_t  (Algorithm 2)."""
    d = Phi.shape[1]
    V = information_matrix(Phi, lam)
    b = Phi.T @ Ytilde
    theta = np.linalg.solve(V, b)
    return theta, V


class LinearCATEEstimator:
    """Algorithm 2 estimator.  Stores V_lambda for downstream bound evaluation."""

    def __init__(self, lam: float = 0.0):
        self.lam = lam
        self.theta_hat: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def fit(self, Phi: np.ndarray, T: np.ndarray, Y: np.ndarray, p: np.ndarray):
        Yt = pseudo_outcome(T, Y, p)
        self.theta_hat, self.V = ridge_estimate(Phi, Yt, self.lam)
        return self

    def predict_tau(self, Phi: np.ndarray) -> np.ndarray:
        return Phi @ self.theta_hat

    def leverage(self, Phi: np.ndarray) -> np.ndarray:
        """phi(x)^T V_lambda^{-1} phi(x) per row -- the linear-model surrogate
        for epistemic uncertainty (Corollary 4.6 / v_u in Algorithm 1)."""
        Vinv = np.linalg.inv(self.V)
        return np.einsum("ij,jk,ik->i", Phi, Vinv, Phi)


def vnorm_error(theta_hat: np.ndarray, theta_star: np.ndarray, V: np.ndarray) -> float:
    """||theta_hat - theta*||_{V_lambda} = sqrt((th-th*)^T V (th-th*))."""
    diff = theta_hat - theta_star
    return float(np.sqrt(max(diff @ V @ diff, 0.0)))


def sandwich_matrices(Phi: np.ndarray, Ytilde: np.ndarray, theta: np.ndarray):
    """Estimate Sigma_pi and Omega_pi (Assumption 4.7) from adaptive data.

    Sigma_pi_hat = (1/B) sum phi_t phi_t^T
    Omega_pi_hat = (1/B) sum (Y~_t - tau_hat(X_t))^2 phi_t phi_t^T
    """
    B = Phi.shape[0]
    Sigma = Phi.T @ Phi / B
    resid = Ytilde - Phi @ theta
    Omega = (Phi * (resid ** 2)[:, None]).T @ Phi / B
    return Sigma, Omega


def theoretical_covariance(Phi: np.ndarray, Ytilde: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Sigma_pi^{-1} Omega_pi Sigma_pi^{-1} -- the asymptotic covariance of
    sqrt(B)(theta_hat - theta*) (Theorem 4.8)."""
    Sigma, Omega = sandwich_matrices(Phi, Ytilde, theta)
    Sinv = np.linalg.inv(Sigma)
    return Sinv @ Omega @ Sinv
