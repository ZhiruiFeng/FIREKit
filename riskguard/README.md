# RiskGuard

Position sizing and risk management toolkit for the FIREKit ecosystem.

## Features

- **Kelly criterion sizing** — binary-outcome Kelly (`kelly_binary`), the
  continuous moment approximation `mu / sigma^2` (`kelly_from_moments`),
  fractional Kelly scaling, and multi-asset Kelly via the inverse covariance
  matrix with sanity clipping (`kelly_multi_asset`).
- **Volatility targeting** — `VolatilityTargeter` scales exposure to hit an
  annualized vol target from lagged rolling realized vol (no look-ahead).
- **Drawdown protection** — `DrawdownCircuitBreaker`, a stateful filter that
  cuts exposure when max drawdown breaches a threshold and re-enters once the
  underlying strategy recovers.
- **Risk metrics** — historical, Gaussian, and Cornish-Fisher VaR and
  CVaR/Expected Shortfall at configurable confidence, plus rolling versions.
  All reported as positive loss fractions.
- **Exposure limits** — `LimitEngine` enforces per-position, sector, gross,
  and net caps; returns clipped/rescaled weights plus a violations report.
- **RiskReport** — `build_risk_report(asset_returns, weights)` assembles all
  of the above in one call.

## Quick start

```python
import numpy as np
import pandas as pd
from riskguard import (
    DrawdownCircuitBreaker, VolatilityTargeter,
    kelly_binary, value_at_risk, build_risk_report,
)

kelly_binary(0.6, 2.0)                       # 0.40

rng = np.random.default_rng(0)
r = pd.Series(0.0005 + 0.01 * rng.standard_normal(1000))

vt = VolatilityTargeter(target_vol=0.10).apply(r)
protected = DrawdownCircuitBreaker(max_drawdown=0.15).apply(vt.scaled_returns)
value_at_risk(r, 0.99, method="cornish_fisher")
```

## Demo

```bash
cd riskguard && python3 -m riskguard.demo   # writes ../hub/data/riskguard.json
```

## Development

```bash
cd riskguard
python3 -m pytest tests -q
ruff check .
```
