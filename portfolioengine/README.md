# PortfolioEngine

Asset allocation and portfolio optimization for the FIREKit ecosystem.

## Features

- **Optimizers** with a common `allocate(mu, cov)` interface, all long-only
  and fully invested by default:
  - `EqualWeight` — 1/N
  - `InverseVolatility` — weights proportional to 1/sigma
  - `MinimumVariance` — SLSQP global minimum variance
  - `MaxSharpe` — SLSQP tangency portfolio
  - `RiskParity` — exact equal risk contribution via cyclical coordinate
    descent (Spinu formulation); risk contributions equalize to ~1e-10
  - `HierarchicalRiskParity` — Lopez de Prado HRP: correlation distance,
    single-linkage clustering, quasi-diagonalization, recursive bisection
- **Covariance estimation** — sample, Ledoit-Wolf shrinkage (scikit-learn),
  EWMA (RiskMetrics-style)
- **Efficient frontier** — target-return sweep of min-variance problems with
  warm starts
- **Backtest harness** — `run_backtest` does rolling re-estimation and
  periodic rebalancing of every optimizer on the same universe, producing
  equity curves, Sharpe, max drawdown, and turnover per method
- **Constraints** — long-only, per-asset max weight, sector caps (exact in
  SLSQP optimizers; cap via clip-and-redistribute in heuristic ones)
- **Synthetic universes** — `make_universe` builds block-correlated
  multi-sector return panels for experimentation

## Quick start

```python
from portfolioengine import (
    Constraints, MaxSharpe, RiskParity, efficient_frontier,
    ledoit_wolf_cov, make_universe, run_backtest,
)

universe = make_universe(n_assets=20, n_sectors=4, seed=7)
cons = Constraints(long_only=True, max_weight=0.15)

results = run_backtest(
    universe.returns,
    {"MaxSharpe": MaxSharpe(cons), "RiskParity": RiskParity(cons)},
    lookback=252,
    rebalance_every=21,
    cov_estimator=ledoit_wolf_cov,
    cost_bps=5.0,
)
print({name: round(r.sharpe, 2) for name, r in results.items()})
```

## Demo

```bash
cd portfolioengine && python3 -m portfolioengine.demo  # writes ../hub/data/portfolioengine.json
```

## Development

```bash
cd portfolioengine
python3 -m pytest tests -q
ruff check .
```
