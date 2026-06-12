"""Synthetic multi-asset universes with block correlation structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Universe:
    """Synthetic asset universe.

    - ``returns``: daily simple returns, T x N DataFrame.
    - ``sectors``: asset -> sector label.
    - ``annual_mu`` / ``annual_cov``: the true generating parameters
      (annualized), useful for validating estimators.
    """

    returns: pd.DataFrame
    sectors: dict[str, str]
    annual_mu: np.ndarray
    annual_cov: np.ndarray

    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def sector_labels(self) -> tuple[str, ...]:
        return tuple(self.sectors[a] for a in self.returns.columns)


def make_universe(
    n_assets: int = 20,
    n_days: int = 1512,
    n_sectors: int = 4,
    seed: int = 7,
    intra_corr: float = 0.55,
    inter_corr: float = 0.10,
    start: str = "2019-01-02",
) -> Universe:
    """Generate a synthetic universe with realistic sector block correlations.

    Assets within a sector share ``intra_corr``; cross-sector pairs share
    ``inter_corr``. Volatilities and expected returns vary per asset, with a
    mild risk-return relationship plus noise.
    """
    if n_assets < n_sectors:
        raise ValueError("need at least one asset per sector")
    if not -1.0 < inter_corr <= intra_corr < 1.0:
        raise ValueError("require -1 < inter_corr <= intra_corr < 1")

    rng = np.random.default_rng(seed)
    sector_names = [f"sector_{chr(65 + i)}" for i in range(n_sectors)]
    sector_idx = np.array([i % n_sectors for i in range(n_assets)])
    assets = [f"A{i:02d}" for i in range(n_assets)]
    sectors = {a: sector_names[sector_idx[i]] for i, a in enumerate(assets)}

    # Correlation: block structure.
    same = sector_idx[:, None] == sector_idx[None, :]
    corr = np.where(same, intra_corr, inter_corr)
    np.fill_diagonal(corr, 1.0)
    # Guarantee positive definiteness.
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-8:
        corr += (1e-8 - eigvals.min()) * np.eye(n_assets)

    annual_vol = rng.uniform(0.12, 0.32, size=n_assets)
    annual_mu = 0.01 + 0.35 * annual_vol + rng.normal(0.0, 0.02, size=n_assets)
    annual_cov = corr * np.outer(annual_vol, annual_vol)

    daily_cov = annual_cov / 252.0
    daily_mu = annual_mu / 252.0
    chol = np.linalg.cholesky(daily_cov)
    shocks = rng.standard_normal((n_days, n_assets))
    daily_returns = daily_mu + shocks @ chol.T

    dates = pd.bdate_range(start, periods=n_days)
    returns = pd.DataFrame(daily_returns, index=dates, columns=assets)
    return Universe(
        returns=returns,
        sectors=sectors,
        annual_mu=annual_mu,
        annual_cov=annual_cov,
    )
