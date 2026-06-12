# DeepTrader

Reinforcement-learning trading agents for FIREKit: a self-contained gym-like
trading environment, tabular Q-learning and linear REINFORCE agents, a seeded
training harness, and strict out-of-sample evaluation with cost-sensitivity
analysis.

## What's inside

| Module | Contents |
|---|---|
| `deeptrader.market` | `regime_switching_series`: seeded AR(1) returns whose autocorrelation flips between momentum and mean-reversion regimes — genuinely learnable structure |
| `deeptrader.env` | `TradingEnv`: `reset()` / `step(action) -> (obs, reward, done, info)`, discrete actions short/flat/long, reward = `position * next_return - cost * |position change|`; `train_test_split_envs` (rewarded returns never shared) |
| `deeptrader.discretize` | `Discretizer`: per-dimension quantile bins -> mixed-radix tabular state ids |
| `deeptrader.agents` | Common `Agent` interface (act / update / save / load): `QLearningAgent` (tabular, epsilon-greedy with decay), `REINFORCEAgent` (linear softmax policy, NumPy gradient ascent, mean baseline), `BuyAndHoldAgent`, `SMACrossoverAgent`, `RandomAgent` |
| `deeptrader.training` | `train`: multi-episode loop with seed control, training curves, periodic greedy OOS evaluation |
| `deeptrader.evaluation` | `evaluate`: OOS equity curve, total return, Sharpe, max drawdown, trade count, win rate; `cost_sensitivity` at 0/5/10 bps |

## Quick start

```python
from deeptrader import (
    Discretizer, QLearningAgent, evaluate, regime_switching_series,
    train, train_test_split_envs,
)

series = regime_switching_series(n=2600, seed=42)
train_env, test_env = train_test_split_envs(series.prices, cost_bps=5.0)

disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, seed=1)
train(agent, train_env, episodes=60)

result = evaluate(test_env, agent)   # greedy policy, strictly out-of-sample
print(result.total_return, result.sharpe, result.max_drawdown)
```

## Demo

```bash
cd deeptrader/ && python3 -m deeptrader.demo
```

Writes `hub/data/deeptrader.json` (hub schema v1): training curves, OOS
equity curves vs buy-and-hold, an agent comparison table, and a transaction
cost sensitivity table.

## Tests

```bash
cd deeptrader/ && python3 -m pytest tests -q
```

Includes a learning guarantee test: the trained Q-agent must beat a
random-policy baseline out-of-sample on the seeded regime-switching series.

## Design-doc substitutions (offline MVP)

- PPO / SAC / A2C / TD3 via FinRL and Stable-Baselines3 (torch) are replaced
  by **tabular Q-learning** and a **NumPy linear-softmax REINFORCE** agent
  behind one common `Agent` interface.
- gymnasium is replaced by a dependency-free gym-like environment
  (`reset` / `step`).
- Live/paper trading layers are out of scope; evaluation is offline OOS
  backtesting on synthetic data with deterministic seeds.
