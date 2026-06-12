"""Portfolio optimizers with a common ``allocate(mu, cov)`` interface.

All optimizers return a weight vector that is long-only (unless configured
otherwise) and sums to 1. Expected returns ``mu`` may be ``None`` for
methods that do not use them (everything except MaxSharpe).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from portfolioengine.constraints import Constraints, apply_weight_cap
from portfolioengine.covariance import to_correlation


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility sqrt(w' cov w) in the units of ``cov``."""
    return float(np.sqrt(weights @ cov @ weights))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's contribution to portfolio variance, normalized to sum 1."""
    w = np.asarray(weights, dtype=float)
    contrib = w * (cov @ w)
    total = contrib.sum()
    if total <= 0.0:
        raise ValueError("portfolio variance must be positive")
    return contrib / total


def _validate(mu: np.ndarray | None, cov: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be square, got shape {sigma.shape}")
    if np.isnan(sigma).any():
        raise ValueError("cov contains NaN")
    if mu is not None:
        mu = np.asarray(mu, dtype=float).ravel()
        if mu.shape[0] != sigma.shape[0]:
            raise ValueError(
                f"dimension mismatch: {mu.shape[0]} returns vs {sigma.shape[0]} assets"
            )
        if np.isnan(mu).any():
            raise ValueError("mu contains NaN")
    return mu, sigma


class Optimizer(ABC):
    """Common interface: weights from expected returns and/or covariance."""

    name: str = "optimizer"

    def __init__(self, constraints: Constraints | None = None):
        self.constraints = constraints or Constraints()

    @abstractmethod
    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        """Return portfolio weights (sums to 1)."""

    def _prepare(self, mu: np.ndarray | None, cov: np.ndarray):
        mu, sigma = _validate(mu, cov)
        self.constraints.validate_dimension(sigma.shape[0])
        return mu, sigma


class EqualWeight(Optimizer):
    """1/N allocation."""

    name = "equal_weight"

    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        _, sigma = self._prepare(mu, cov)
        n = sigma.shape[0]
        return np.full(n, 1.0 / n)


class InverseVolatility(Optimizer):
    """Weights proportional to 1 / volatility."""

    name = "inverse_volatility"

    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        _, sigma = self._prepare(mu, cov)
        vols = np.sqrt(np.diag(sigma))
        if np.any(vols <= 0.0):
            raise ValueError("all asset variances must be positive")
        w = (1.0 / vols) / np.sum(1.0 / vols)
        return apply_weight_cap(w, self.constraints.max_weight)


class MinimumVariance(Optimizer):
    """Global minimum-variance portfolio via SLSQP (long-only, fully invested)."""

    name = "min_variance"

    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        _, sigma = self._prepare(mu, cov)
        n = sigma.shape[0]
        w0 = np.full(n, 1.0 / n)
        result = minimize(
            lambda w: float(w @ sigma @ w),
            w0,
            jac=lambda w: 2.0 * (sigma @ w),
            method="SLSQP",
            bounds=self.constraints.bounds(n),
            constraints=self.constraints.scipy_constraints(n),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"min-variance optimization failed: {result.message}")
        w = np.asarray(result.x)
        return w / w.sum()


class MaxSharpe(Optimizer):
    """Maximum Sharpe ratio (tangency) portfolio via SLSQP."""

    name = "max_sharpe"

    def __init__(
        self,
        constraints: Constraints | None = None,
        risk_free_rate: float = 0.0,
    ):
        super().__init__(constraints)
        self.risk_free_rate = risk_free_rate

    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        mu, sigma = self._prepare(mu, cov)
        if mu is None:
            raise ValueError("MaxSharpe requires expected returns")
        n = sigma.shape[0]
        rf = self.risk_free_rate

        def neg_sharpe(w: np.ndarray) -> float:
            vol = np.sqrt(w @ sigma @ w)
            if vol <= 0.0:
                return 0.0
            return -float((w @ mu - rf) / vol)

        w0 = np.full(n, 1.0 / n)
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=self.constraints.bounds(n),
            constraints=self.constraints.scipy_constraints(n),
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"max-Sharpe optimization failed: {result.message}")
        w = np.asarray(result.x)
        return w / w.sum()


class RiskParity(Optimizer):
    """Equal risk contribution portfolio (long-only).

    Solves the Spinu formulation ``min 0.5 x' cov x - sum(b_i log x_i)``
    by cyclical coordinate descent; the KKT conditions give
    ``x_i (cov x)_i = b_i``, i.e. risk contributions exactly proportional
    to the budgets. Weights are the normalized solution.
    """

    name = "risk_parity"

    def __init__(
        self,
        constraints: Constraints | None = None,
        budgets: np.ndarray | None = None,
        tol: float = 1e-12,
        max_iter: int = 10_000,
    ):
        super().__init__(constraints)
        self.budgets = budgets
        self.tol = tol
        self.max_iter = max_iter

    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        _, sigma = self._prepare(mu, cov)
        n = sigma.shape[0]
        if self.budgets is None:
            budgets = np.full(n, 1.0 / n)
        else:
            budgets = np.asarray(self.budgets, dtype=float).ravel()
            if budgets.shape[0] != n:
                raise ValueError("budgets dimension mismatch")
            if np.any(budgets <= 0.0):
                raise ValueError("budgets must be positive")
            budgets = budgets / budgets.sum()

        diag = np.diag(sigma)
        if np.any(diag <= 0.0):
            raise ValueError("all asset variances must be positive")

        x = np.sqrt(budgets) / np.sqrt(diag)
        sigma_x = sigma @ x
        for _ in range(self.max_iter):
            x_prev = x.copy()
            for i in range(n):
                c_ii = diag[i]
                c_i = sigma_x[i] - c_ii * x[i]  # sum_{j != i} sigma_ij x_j
                x_new = (-c_i + np.sqrt(c_i * c_i + 4.0 * c_ii * budgets[i])) / (2.0 * c_ii)
                dx = x_new - x[i]
                if dx != 0.0:
                    sigma_x += sigma[:, i] * dx
                    x[i] = x_new
            if np.max(np.abs(x - x_prev)) < self.tol:
                break
        w = x / x.sum()
        return apply_weight_cap(w, self.constraints.max_weight)


class HierarchicalRiskParity(Optimizer):
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Steps: correlation-distance matrix -> single-linkage hierarchical
    clustering -> quasi-diagonalization (leaf order) -> recursive bisection
    allocating inverse-variance between sub-clusters.
    """

    name = "hrp"

    def __init__(
        self,
        constraints: Constraints | None = None,
        linkage_method: str = "single",
    ):
        super().__init__(constraints)
        self.linkage_method = linkage_method

    def allocate(self, mu: np.ndarray | None, cov: np.ndarray) -> np.ndarray:
        _, sigma = self._prepare(mu, cov)
        n = sigma.shape[0]
        if n == 1:
            return np.array([1.0])
        order = self._quasi_diagonal_order(sigma)
        w = self._recursive_bisection(sigma, order)
        w = w / w.sum()
        return apply_weight_cap(w, self.constraints.max_weight)

    def _quasi_diagonal_order(self, sigma: np.ndarray) -> np.ndarray:
        corr = to_correlation(sigma)
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)
        return np.asarray(leaves_list(link), dtype=int)

    @staticmethod
    def _cluster_variance(sigma: np.ndarray, indices: np.ndarray) -> float:
        sub = sigma[np.ix_(indices, indices)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    def _recursive_bisection(self, sigma: np.ndarray, order: np.ndarray) -> np.ndarray:
        weights = np.ones(sigma.shape[0])
        clusters: list[np.ndarray] = [order]
        while clusters:
            next_clusters: list[np.ndarray] = []
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]
                var_left = self._cluster_variance(sigma, left)
                var_right = self._cluster_variance(sigma, right)
                alpha = 1.0 - var_left / (var_left + var_right)
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
                next_clusters.extend([left, right])
            clusters = next_clusters
        return weights


DEFAULT_OPTIMIZERS: tuple[type[Optimizer], ...] = (
    EqualWeight,
    InverseVolatility,
    MinimumVariance,
    MaxSharpe,
    RiskParity,
    HierarchicalRiskParity,
)
