"""DeepTrader demo: train Q-learning and REINFORCE on a regime-switching market.

Trains both agents on the in-sample segment, evaluates out-of-sample against
rule-based baselines, and writes hub results JSON. Run with::

    cd deeptrader/ && python3 -m deeptrader.demo
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from deeptrader.agents import (
    Agent,
    BuyAndHoldAgent,
    QLearningAgent,
    RandomAgent,
    REINFORCEAgent,
    SMACrossoverAgent,
)
from deeptrader.discretize import Discretizer
from deeptrader.env import train_test_split_envs
from deeptrader.evaluation import cost_sensitivity, evaluate
from deeptrader.market import regime_switching_series
from deeptrader.training import train

SEED = 42
N_POINTS = 2_600
TRAIN_FRAC = 0.7
COST_BPS = 5.0
EPISODES = 60
HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"


def _clean(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def _downsample(values: list, max_points: int = 480) -> list:
    if len(values) <= max_points:
        return values
    stride = math.ceil(len(values) / max_points)
    sampled = values[::stride]
    if (len(values) - 1) % stride != 0:
        sampled.append(values[-1])
    return sampled


def run_demo() -> dict:
    series = regime_switching_series(n=N_POINTS, seed=SEED)
    train_env, test_env = train_test_split_envs(
        series.prices, train_frac=TRAIN_FRAC, cost_bps=COST_BPS
    )

    # --- train the two RL agents (never on the test segment) ---------------
    discretizer = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
    q_agent = QLearningAgent(
        discretizer, alpha=0.1, gamma=0.5, epsilon=1.0, epsilon_decay=0.92, seed=1
    )
    q_history = train(q_agent, train_env, episodes=EPISODES, eval_env=test_env, eval_every=10)

    pg_agent = REINFORCEAgent(obs_dim=train_env.observation_dim, lr=0.1, gamma=0.99, seed=2)
    pg_agent.fit_scaler(train_env.observations)
    pg_history = train(pg_agent, train_env, episodes=EPISODES, eval_env=test_env, eval_every=10)

    # --- OOS evaluation ------------------------------------------------------
    agents: dict[str, Agent] = {
        "Q-learning": q_agent,
        "REINFORCE": pg_agent,
        "Buy & Hold": BuyAndHoldAgent(),
        "SMA crossover": SMACrossoverAgent(fast=5, slow=20),
        "Random": RandomAgent(seed=3),
    }
    results = {name: evaluate(test_env, agent) for name, agent in agents.items()}
    q_res = results["Q-learning"]
    pg_res = results["REINFORCE"]
    bh_res = results["Buy & Hold"]
    rnd_res = results["Random"]

    summary_metrics = [
        {"label": "Q-learning OOS return", "value": f"{q_res.total_return * 100:.1f}", "unit": "%"},
        {"label": "Q-learning OOS Sharpe", "value": f"{q_res.sharpe:.2f}"},
        {"label": "REINFORCE OOS Sharpe", "value": f"{pg_res.sharpe:.2f}"},
        {"label": "Buy & Hold OOS Sharpe", "value": f"{bh_res.sharpe:.2f}"},
        {
            "label": "Q edge vs random (return)",
            "value": f"{(q_res.total_return - rnd_res.total_return) * 100:.1f}",
            "unit": "pp",
        },
        {"label": "Episodes trained", "value": str(EPISODES)},
        {"label": "OOS test steps", "value": str(test_env.n_steps)},
        {"label": "Tabular states", "value": str(discretizer.n_states)},
    ]

    # --- chart: training curves ---------------------------------------------
    episodes_x = _downsample(list(range(1, EPISODES + 1)))
    training_chart = {
        "id": "training_curves",
        "title": "Training curve: episode reward (in-sample)",
        "type": "line",
        "x_label": "Episode",
        "y_label": "Episode reward (sum of step rewards)",
        "x": episodes_x,
        "series": [
            {"name": "Q-learning", "data": _downsample([_clean(r) for r in q_history.episode_rewards])},
            {"name": "REINFORCE", "data": _downsample([_clean(r) for r in pg_history.episode_rewards])},
        ],
    }

    # --- chart: OOS equity curves vs buy-and-hold ----------------------------
    steps = list(range(len(q_res.equity_curve)))
    equity_chart = {
        "id": "oos_equity",
        "title": "Out-of-sample equity curves (greedy policies, 5 bps costs)",
        "type": "line",
        "x_label": "Test step",
        "y_label": "Equity (start = 1.0)",
        "x": _downsample(steps),
        "series": [
            {
                "name": name,
                "data": _downsample([_clean(v) for v in results[name].equity_curve.tolist()]),
            }
            for name in ("Q-learning", "REINFORCE", "Buy & Hold", "SMA crossover")
        ],
    }

    # --- table: agent comparison ---------------------------------------------
    comparison_table = {
        "id": "agent_comparison",
        "title": f"OOS agent comparison ({COST_BPS:.0f} bps transaction cost)",
        "columns": ["Agent", "Total return (%)", "Sharpe", "Max drawdown (%)", "Trades", "Win rate (%)"],
        "rows": [
            [
                name,
                _clean(res.total_return * 100, 1),
                _clean(res.sharpe, 2),
                _clean(res.max_drawdown * 100, 1),
                res.n_trades,
                _clean(res.win_rate * 100, 1),
            ]
            for name, res in results.items()
        ],
    }

    # --- table: cost sensitivity ----------------------------------------------
    sensitivity_rows = cost_sensitivity(
        {name: agents[name] for name in ("Q-learning", "REINFORCE", "Buy & Hold")},
        series.prices,
        cost_bps_levels=(0.0, 5.0, 10.0),
        train_frac=TRAIN_FRAC,
    )
    sensitivity_table = {
        "id": "cost_sensitivity",
        "title": "OOS cost sensitivity (same trained policies, varying costs)",
        "columns": ["Agent", "Cost (bps)", "Total return (%)", "Sharpe", "Trades"],
        "rows": [
            [
                row["agent"],
                _clean(row["cost_bps"], 0),
                _clean(row["total_return"] * 100, 1),
                _clean(row["sharpe"], 2),
                row["n_trades"],
            ]
            for row in sensitivity_rows
        ],
    }

    return {
        "schema_version": 1,
        "product": "deeptrader",
        "title": "DeepTrader",
        "tagline": "RL trading agents: tabular Q-learning and REINFORCE with OOS evaluation",
        "version": "0.1.0",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": summary_metrics,
        "charts": [training_chart, equity_chart],
        "tables": [comparison_table, sensitivity_table],
        "notes": (
            "Offline MVP: PPO/SAC via FinRL/Stable-Baselines3 from the design doc are replaced "
            "by tabular Q-learning and a NumPy linear-softmax REINFORCE agent. Market is a "
            "seeded regime-switching AR(1) (momentum vs mean-reversion), so structure is "
            "genuinely learnable; agents train on the first 70% and are evaluated strictly "
            "out-of-sample. Returns are large because the synthetic signal is deliberately "
            "strong; the point is the learning dynamics, not realistic alpha."
        ),
    }


def main() -> None:
    payload = run_demo()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUB_DATA_DIR / "deeptrader.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"DeepTrader demo complete -> {out_path}")
    for metric in payload["summary_metrics"]:
        unit = metric.get("unit", "")
        print(f"  {metric['label']}: {metric['value']}{unit}")


if __name__ == "__main__":
    main()
