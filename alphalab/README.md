# AlphaLab

Factor mining workbench for the FIREKit ecosystem (offline MVP): a wide-panel
factor framework, a 20-factor built-in library (momentum, reversal,
volatility, volume/liquidity, technical, beta, and 5 Alpha101 formulaic
alphas), and a rigorous evaluation stack — IC / rank-IC time series, IR and
t-stats, quantile portfolios with top-minus-bottom spread, turnover, and a
factor correlation matrix.

## Install

```bash
cd alphalab && pip install -e ".[dev]"
```

## Usage

```python
from alphalab import FactorZoo, Momentum, make_synthetic_panel

# 1. Build a panel (date x symbol wide frames; Panel.from_long for long data)
panel = make_synthetic_panel(n_symbols=50, n_days=756, seed=7)

# 2. Evaluate the whole zoo vs 5-day forward returns
report = FactorZoo.default().evaluate(panel, horizon=5)
print(report.to_frame()[["ic_mean", "rank_ic_mean", "rank_ic_ir", "spread_mean", "turnover"]])
print("best:", report.best.name, "redundant pairs:", report.top_correlated_pairs(3))

# 3. Or evaluate a single factor
from alphalab import evaluate_factor, forward_returns
values = Momentum(63).compute(panel)
result = evaluate_factor("momentum_63d", values, forward_returns(panel.close, 5), horizon=5)
print(result.rank_ic)  # ICStats(mean, std, ir, t_stat, pct_positive, n_obs)
```

Custom factors subclass `Factor` and implement `_compute(panel) -> DataFrame`;
register them in a `FactorZoo` to include them in batch runs. Cross-sectional
and time-series operators (`cs_rank`, `ts_corr`, `delta`, ...) live in
`alphalab.ops`.

## Demo

```bash
cd alphalab && python3 -m alphalab.demo   # writes ../hub/data/alphalab.json
```

## Tests

```bash
cd alphalab && python3 -m pytest tests -q
```

Out of scope for this MVP (per design doc): the eval-based DSL factor
builder, purged K-fold / CPCV cross-validation, multiple-testing correction,
and factor-combination engines.
