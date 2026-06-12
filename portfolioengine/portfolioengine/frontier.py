"""Efficient frontier: target-return sweep of minimum-variance problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from portfolioengine.constraints import Constraints
from portfolioengine.optimizers import MinimumVariance, portfolio_volatility


@dataclass(frozen=True)
class FrontierPoint:
    """One point on the efficient frontier."""

    expected_return: float
    volatility: float
    weights: np.ndarray

    @property
    def sharpe(self) -> float:
        return self.expected_return / self.volatility if self.volatility > 0 else 0.0


def _extreme_return(
    mu: np.ndarray,
    constraints: Constraints,
    maximize: bool = True,
) -> float:
    """Max (or min) achievable expected return under the constraint set."""
    n = mu.shape[0]
    sign = -1.0 if maximize else 1.0
    result = minimize(
        lambda w: sign * float(w @ mu),
        np.full(n, 1.0 / n),
        jac=lambda w: sign * mu,
        method="SLSQP",
        bounds=constraints.bounds(n),
        constraints=constraints.scipy_constraints(n),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"return bound optimization failed: {result.message}")
    return float(result.x @ mu)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 30,
    constraints: Constraints | None = None,
) -> list[FrontierPoint]:
    """Sweep target returns and solve min-variance at each level.

    The sweep runs from the minimum-variance portfolio's return up to the
    maximum achievable return under the constraints, so every point is on
    the *efficient* (upper) branch of the frontier.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    constraints = constraints or Constraints()
    n = mu.shape[0]
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    constraints.validate_dimension(n)

    mv = MinimumVariance(constraints)
    w_minvar = mv.allocate(None, sigma)
    ret_low = float(w_minvar @ mu)
    ret_high = _extreme_return(mu, constraints, maximize=True)
    if ret_high <= ret_low:
        return [FrontierPoint(ret_low, portfolio_volatility(w_minvar, sigma), w_minvar)]

    points: list[FrontierPoint] = []
    base_cons = constraints.scipy_constraints(n)
    bounds = constraints.bounds(n)
    w0 = w_minvar
    for target in np.linspace(ret_low, ret_high, n_points):
        cons = [*base_cons, {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t}]
        result = minimize(
            lambda w: float(w @ sigma @ w),
            w0,
            jac=lambda w: 2.0 * (sigma @ w),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            continue
        w = np.asarray(result.x)
        w = w / w.sum()
        points.append(FrontierPoint(float(w @ mu), portfolio_volatility(w, sigma), w))
        w0 = w  # warm start the next solve
    return points
