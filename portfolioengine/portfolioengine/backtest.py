"""Backtest comparison harness: rolling re-estimation + periodic rebalancing.

A lightweight internal simulator: weights are re-estimated every
``rebalance_every`` periods from a trailing ``lookback`` window and held
constant (constant-mix) between rebalances. Optional proportional
transaction costs are charged on turnover.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolioengine.covariance import sample_cov
from portfolioengine.optimizers import Optimizer


@dataclass(frozen=True)
class BacktestResult:
    """Per-method backtest output."""

    name: str
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # one row per rebalance date
    avg_turnover: float  # mean one-way turnover per rebalance (sum |dw|)
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float

    def stats(self) -> dict[str, float]:
        return {
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "avg_turnover": self.avg_turnover,
        }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def run_backtest(
    returns: pd.DataFrame,
    optimizers: dict[str, Optimizer],
    lookback: int = 252,
    rebalance_every: int = 21,
    cov_estimator: Callable[[pd.DataFrame], np.ndarray] = sample_cov,
    cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, BacktestResult]:
    """Run every optimizer over the same universe and rebalance schedule.

    Parameters
    ----------
    returns:
        T x N DataFrame of per-period simple returns.
    optimizers:
        ``{name: optimizer}``; each is called as ``allocate(mu, cov)`` with
        the trailing-window mean and the estimated covariance (per-period).
    lookback:
        Estimation window length; the backtest starts after the first window.
    rebalance_every:
        Holding period between re-estimations, in periods.
    cov_estimator:
        Function mapping a returns window to a covariance matrix.
    cost_bps:
        One-way proportional transaction cost in basis points, charged on
        turnover at each rebalance.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every must be >= 1, got {rebalance_every}")
    if len(returns) <= lookback:
        raise ValueError(
            f"need more than lookback={lookback} rows, got {len(returns)}"
        )
    if returns.isna().any().any():
        raise ValueError("returns contain NaN values")

    arr = returns.to_numpy(dtype=float)
    index = returns.index
    n_periods = len(returns)
    rebalance_points = list(range(lookback, n_periods, rebalance_every))
    cost_rate = cost_bps / 10_000.0

    # Pre-compute estimates once per rebalance date (shared by all methods).
    estimates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for t in rebalance_points:
        window = returns.iloc[t - lookback : t]
        mu = window.to_numpy().mean(axis=0)
        cov = cov_estimator(window)
        estimates.append((t, mu, cov))

    results: dict[str, BacktestResult] = {}
    out_index = index[lookback:]
    for name, optimizer in optimizers.items():
        port_returns = np.empty(n_periods - lookback)
        weight_rows: list[np.ndarray] = []
        rebalance_dates: list = []
        turnovers: list[float] = []
        prev_w: np.ndarray | None = None

        for k, (t, mu, cov) in enumerate(estimates):
            w = np.asarray(optimizer.allocate(mu, cov), dtype=float)
            weight_rows.append(w)
            rebalance_dates.append(index[t])
            turnover = float(np.abs(w - prev_w).sum()) if prev_w is not None else 0.0
            turnovers.append(turnover)
            prev_w = w

            t_end = estimates[k + 1][0] if k + 1 < len(estimates) else n_periods
            segment = arr[t:t_end] @ w
            if cost_rate > 0.0 and turnover > 0.0:
                segment = segment.copy()
                segment[0] -= cost_rate * turnover
            port_returns[t - lookback : t_end - lookback] = segment

        rets = pd.Series(port_returns, index=out_index, name=name)
        equity = (1.0 + rets).cumprod()
        ann_ret = float(rets.mean() * periods_per_year)
        ann_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
        results[name] = BacktestResult(
            name=name,
            equity=equity,
            returns=rets,
            weights=pd.DataFrame(
                weight_rows, index=rebalance_dates, columns=returns.columns
            ),
            # First rebalance is initial deployment, not turnover.
            avg_turnover=float(np.mean(turnovers[1:])) if len(turnovers) > 1 else 0.0,
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            sharpe=ann_ret / ann_vol if ann_vol > 0 else 0.0,
            max_drawdown=_max_drawdown(equity.to_numpy()),
        )
    return results
