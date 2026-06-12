"""Out-of-sample evaluation: equity curves, risk metrics, cost sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deeptrader.agents import Agent
from deeptrader.env import TradingEnv

PERIODS_PER_YEAR = 252


@dataclass
class EvalResult:
    """Greedy-policy performance over one environment pass."""

    equity_curve: np.ndarray  # length n_steps + 1, starts at 1.0
    rewards: np.ndarray
    positions: np.ndarray
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    win_rate: float


def sharpe_ratio(step_returns: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    step_returns = np.asarray(step_returns, dtype=float)
    std = step_returns.std()
    if std < 1e-12:
        return 0.0
    return float(step_returns.mean() / std * np.sqrt(periods_per_year))

def max_drawdown(equity_curve: np.ndarray) -> float:
    equity_curve = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity_curve)
    return float(np.max(1.0 - equity_curve / peaks))


def _trade_stats(positions: np.ndarray, rewards: np.ndarray) -> tuple[int, float]:
    """Trade count = position changes; win rate over closed holding segments."""
    prev = 0
    n_trades = 0
    segment_pnl = 0.0
    wins = 0
    closed = 0
    for pos, reward in zip(positions, rewards, strict=True):
        if pos != prev:
            n_trades += 1
            if prev != 0:  # a holding segment just closed
                closed += 1
                wins += segment_pnl > 0
            segment_pnl = 0.0
        if pos != 0:
            segment_pnl += reward
        prev = pos
    if prev != 0:  # final open segment counts as a trade outcome
        closed += 1
        wins += segment_pnl > 0
    win_rate = wins / closed if closed else 0.0
    return n_trades, win_rate


def evaluate(env: TradingEnv, agent: Agent) -> EvalResult:
    """Run the greedy policy once over ``env`` and compute performance metrics."""
    agent.reset()
    obs = env.reset()
    rewards: list[float] = []
    positions: list[int] = []
    done = False
    while not done:
        action = agent.act(obs, explore=False)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        positions.append(info["position"])
    rewards_arr = np.asarray(rewards)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + rewards_arr)])
    n_trades, win_rate = _trade_stats(np.asarray(positions), rewards_arr)
    return EvalResult(
        equity_curve=equity,
        rewards=rewards_arr,
        positions=np.asarray(positions),
        total_return=float(equity[-1] - 1.0),
        sharpe=sharpe_ratio(rewards_arr),
        max_drawdown=max_drawdown(equity),
        n_trades=n_trades,
        win_rate=float(win_rate),
    )


def cost_sensitivity(
    agents: dict[str, Agent],
    prices: np.ndarray,
    cost_bps_levels: tuple[float, ...] = (0.0, 5.0, 10.0),
    train_frac: float = 0.7,
    ac_window: int = 20,
) -> list[dict[str, float | str]]:
    """Evaluate already-trained agents OOS at several transaction-cost levels."""
    from deeptrader.env import train_test_split_envs

    rows: list[dict[str, float | str]] = []
    for cost in cost_bps_levels:
        _, test_env = train_test_split_envs(
            prices, train_frac=train_frac, cost_bps=cost, ac_window=ac_window
        )
        for name, agent in agents.items():
            result = evaluate(test_env, agent)
            rows.append(
                {
                    "agent": name,
                    "cost_bps": cost,
                    "total_return": result.total_return,
                    "sharpe": result.sharpe,
                    "n_trades": result.n_trades,
                }
            )
    return rows
