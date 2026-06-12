# FIREKit User Manual

> 中文版本: [MANUAL.zh-CN.md](MANUAL.zh-CN.md)

FIREKit is an ecosystem of nine installable Python products for building
AI-powered quantitative trading systems, plus a self-contained dashboard
(the **Hub**) and two top-level entry points (`validate_all.py`,
`pipeline.py`) that tie everything together.

This manual covers: setup, the day-to-day workflow, a per-product guide
with working code, the end-to-end pipeline walkthrough, and troubleshooting.

---

## 1. The ecosystem at a glance

| Layer | Product | What it does |
|-------|---------|--------------|
| Data | **DataStream** | Data sources, Parquet store, quality engine, point-in-time universe |
| Research | **AlphaLab** | 20-factor zoo (incl. Alpha101), IC / quantile evaluation |
| Research | **SentimentPulse** | Financial lexicon scorer, pluggable LLM providers, shock detection |
| Signals | **SignalML** | Model zoo, purged walk-forward validation, ensembles, registry |
| Signals | **DeepTrader** | Gym-like trading env, Q-learning + REINFORCE agents, OOS evaluation |
| Allocation | **PortfolioEngine** | Min-var / max-Sharpe / risk-parity / HRP optimizers, efficient frontier |
| Risk | **RiskGuard** | Kelly sizing, vol targeting, circuit breaker, VaR/CVaR, exposure limits |
| Execution | **ExecutionCore** | Order lifecycle, paper broker, TWAP/VWAP, shortfall analytics |
| Backtesting | **VectorForge** | Hybrid vectorized / event-driven backtester with multi-asset portfolio mode |

Data flows bottom-up: DataStream feeds the research layer (AlphaLab,
SentimentPulse), which feeds the signal layer (SignalML, DeepTrader), which
feeds allocation (PortfolioEngine), risk (RiskGuard), and execution
(ExecutionCore). VectorForge validates any strategy independently.

Every product lives in its own directory with the same shape:

```
<product>/
  <product>/        # the Python package
    demo.py         # deterministic demo, writes hub/data/<product>.json
  tests/            # pytest suite
  pyproject.toml
  README.md         # full API documentation for that product
```

---

## 2. Installation

Requirements: **Python 3.11+**. All products share one dependency set
(no per-product installs needed):

```bash
git clone https://github.com/ZhiruiFeng/FIREKit.git
cd FIREKit
pip install numpy pandas polars scipy scikit-learn pydantic pyarrow pyyaml pytest
```

Nothing needs `pip install -e`; the top-level scripts handle paths, and
each product runs from its own directory.

---

## 3. The core workflow

### 3.1 Validate everything (one command)

```bash
python3 validate_all.py
```

This runs five layers and prints a PASS/FAIL line per check
(~2 minutes total, exit code 0 = all green):

1. **environment** — Python version + required packages
2. **smoke** — every product imports and reports its version
3. **tests** — every product's pytest suite (~1000 tests)
4. **demos** — `run_all.py`: all 9 demos + hub JSON schema validation + bundle
5. **pipeline** — `pipeline.py --fast`: the cross-product integration run

Useful flags:

```bash
python3 validate_all.py --skip-tests            # fast sanity check
python3 validate_all.py --only signalml,riskguard   # focus layers 2-3
python3 validate_all.py --skip-demos --skip-pipeline # tests only
```

Run this before and after any change, and before every commit.

### 3.2 Run the demos and open the dashboard

```bash
python3 run_all.py                     # all product demos + bundle
python3 run_all.py --only alphalab     # one product's demo
python3 run_all.py --bundle-only       # rebuild products.js from existing JSON
python3 -m http.server -d hub 8080     # open http://localhost:8080
```

Each demo is deterministic (seeded) and writes `hub/data/<product>.json`
following `hub/SCHEMA.md`. The Hub (`hub/index.html`) is dependency-free
vanilla JS and also works from `file://`.

### 3.3 Run the end-to-end pipeline

```bash
python3 pipeline.py          # full run (30 symbols, 3 years, ~15s)
python3 pipeline.py --fast   # smaller run (~7s)
```

This chains **all nine products** on one shared synthetic universe and adds
a tenth tile ("End-to-End Pipeline") to the Hub. See section 5 for the
stage-by-stage walkthrough.

### 3.4 Test a single product

```bash
cd signalml && python3 -m pytest tests -q
cd signalml && python3 -m signalml.demo
```

---

## 4. Per-product guide

Each subsection shows the canonical usage pattern. All snippets are runnable
from the product's own directory (or with the product directory on
`sys.path`, as `pipeline.py` does).

### 4.1 DataStream — data pipeline

```python
from datastream import SyntheticSource, ParquetStore, QualityEngine, PointInTimeUniverse

source = SyntheticSource(n_symbols=30, start="2021-01-04", end="2023-12-29",
                         seed=11, issue_rate=0.01)
raw = source.fetch()                      # long frame: timestamp, symbol, OHLCV

store = ParquetStore("./market_data")     # partitioned by symbol, zstd
store.write(raw)

clean, report = QualityEngine().run(store.read_many())
print(report.score, report.issue_counts())   # quality 0-100, issues by type

universe = PointInTimeUniverse()
universe.add("SYM001", "2021-01-04")          # joined date (optional left date)
survivorship_safe = universe.filter_frame(clean)
```

Key facts:
- Canonical format is a **long** DataFrame (`timestamp, symbol, open, high,
  low, close, volume`); `normalize_frame` coerces to it.
- The QualityEngine **repairs** duplicates, NaNs, non-positive prices and
  OHLC violations, but only **flags** return outliers and gaps — handle
  flagged outliers yourself (see `mask_return_outliers` in `pipeline.py`).
- `FileSource` loads CSV/Parquet directories for real data.

### 4.2 AlphaLab — factor research

```python
from alphalab import FactorZoo, Panel, Momentum, make_synthetic_panel

panel = Panel.from_long(clean_long_frame)       # or make_synthetic_panel(...)
report = FactorZoo.default().evaluate(panel, horizon=5, n_quantiles=5)

print(report.to_frame().head(10))     # ranked by |rank-IC IR|
best = report.best                    # FactorEvaluation
print(best.name, best.rank_ic.mean, best.rank_ic.ir, best.turnover)
print(report.top_correlated_pairs(3))  # redundancy candidates

mom = Momentum(lookback=126).compute(panel)     # one factor, dates x symbols
```

Key facts: a `Panel` is five aligned wide DataFrames (dates × symbols);
factor values are wide DataFrames you can feed into SignalML as features or
into VectorForge as signals.

### 4.3 SentimentPulse — news sentiment

```python
from sentimentpulse import (LexiconProvider, AliasMap, scores_frame,
                            daily_scores, sentiment_index, detect_shocks, event_study)

provider = LexiconProvider()                    # bundled financial lexicon
results = provider.score_batch(headlines)       # scores in [-1, 1]

frame = scores_frame(items, results)            # items: list[NewsItem]
daily = daily_scores(frame, dates=trading_days) # wide: date x symbol
index = sentiment_index(daily, halflife=5.0)    # decay-weighted index
shocks = detect_shocks(daily)                   # z-score spikes
study = event_study(shocks, close_prices)       # avg fwd returns after shocks
```

Key facts: providers are pluggable (`SentimentProvider` ABC) — implement
`score_batch` against a real LLM to upgrade from the lexicon. `AliasMap`
maps company names/aliases to tickers for tagging untagged news.

### 4.4 SignalML — ML signals

```python
from signalml import (build_dataset, WalkForwardEngine, RidgeSignalModel,
                      GradientBoostingSignalModel, ic_weights, EnsembleCombiner,
                      summarize, ModelRegistry)

X, y = build_dataset(close_wide, volume_wide, horizon=5)   # (date, symbol) index

engine = WalkForwardEngine(
    {"ridge": lambda: RidgeSignalModel(alpha=10.0),
     "gbdt": lambda: GradientBoostingSignalModel()},
    train_size=252, test_size=21, gap=5,        # purged gap kills label leakage
)
result = engine.run(X, y)                       # stitched OOS predictions

weights = ic_weights(result.predictions, result.actuals)
signal = EnsembleCombiner("zscore_mean", weights=weights).combine(result.predictions)
print(summarize(signal, result.actuals))        # ic, rank_ic, hit_rate, spread

ModelRegistry("./models").save(result.models["ridge"], metadata={"ic": 0.02})
```

Key facts: everything is indexed by `(date, symbol)` MultiIndex; the `gap`
parameter must be ≥ the label horizon to prevent leakage; extra features
(e.g. a sentiment index) are added as plain columns on `X`.

### 4.5 DeepTrader — RL agents

```python
from deeptrader import (regime_switching_series, train_test_split_envs,
                        Discretizer, QLearningAgent, train, evaluate, cost_sensitivity)

series = regime_switching_series(n=2600, seed=42)
train_env, test_env = train_test_split_envs(series.prices, cost_bps=5.0)

disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, seed=1)
train(agent, train_env, episodes=60)

result = evaluate(test_env, agent)     # equity curve, Sharpe, maxDD, win rate
table = cost_sensitivity(test_env, agent)   # robustness at 0/5/10 bps
```

Key facts: agents act on positions {-1, 0, +1}; always compare against the
included baselines (`BuyAndHoldAgent`, `SMACrossoverAgent`, `RandomAgent`)
on the out-of-sample segment — tabular RL on noisy prices often loses.

### 4.6 PortfolioEngine — allocation

```python
from portfolioengine import (RiskParity, MaxSharpe, Constraints,
                             ledoit_wolf_cov, efficient_frontier, run_backtest, make_universe)

cov = ledoit_wolf_cov(returns_window)          # shrinkage, np.ndarray out
w = RiskParity(Constraints(long_only=True, max_weight=0.25)).allocate(None, cov)
w2 = MaxSharpe(Constraints(long_only=True)).allocate(mu, cov)   # needs mu

frontier = efficient_frontier(mu, cov, n_points=30)
result = run_backtest(returns, optimizers=None, lookback=252,
                      rebalance_every=21, cost_bps=5.0)   # rolling comparison
```

Key facts: all optimizers share `allocate(mu, cov) -> weights`; analytic
optimizers ignore `mu`. SLSQP-based ones (MinVar, MaxSharpe) enforce sector
caps exactly; heuristics enforce `max_weight` by clip-and-redistribute.

### 4.7 RiskGuard — risk management

```python
from riskguard import (VolatilityTargeter, DrawdownCircuitBreaker,
                       kelly_from_moments, build_risk_report)

vt = VolatilityTargeter(target_vol=0.10, window=20, max_leverage=1.5)
scaled = vt.apply(strategy_returns)            # .scaled_returns, .exposure

breaker = DrawdownCircuitBreaker(max_drawdown=0.12, reentry_drawdown=0.04)
protected = breaker.apply(scaled.scaled_returns)   # .filtered_returns, .n_triggers

size = kelly_from_moments(mean=0.001, variance=0.0004, fraction=0.5)

report = build_risk_report(asset_returns, weights, target_vol=0.10)
print(report.to_dict()["var"]["historical"]["0.95"])
```

Key facts: both the vol targeter and the circuit breaker are lag-1
(no look-ahead); `build_risk_report` assembles VaR/CVaR (3 methods ×
2 confidences), Kelly suggestions, vol-target exposure and limit checks
in one call.

### 4.8 ExecutionCore — order execution

```python
from executioncore import (PaperBroker, OrderManager, Order, OrderSide, PriceFeed,
                           FixedBpsSlippage, PerShareCommission, MaxOrderSize,
                           MaxNotional, TWAP, AlgoExecutor, run_session,
                           fill_rate, slippage_bps, synthetic_intraday_feed)

feed = synthetic_intraday_feed("ACME", n_bars=390, start_price=100.0, seed=7)
broker = PaperBroker(feed, slippage=FixedBpsSlippage(2.0),
                     commission=PerShareCommission(0.005), participation_cap=0.25)
oms = OrderManager(broker, cash=1_000_000,
                   validators=(MaxOrderSize(10_000), MaxNotional(500_000)))

order = Order(symbol="ACME", side=OrderSide.BUY, quantity=5_000)
run_session(broker, submissions={0: [order]}, oms=oms)

print(fill_rate(oms.orders.values()), slippage_bps(order), oms.snapshot().equity)
```

Key facts: fills are strictly causal (an order can never fill on a bar known
at submission); `participation_cap` produces realistic partial fills; use
`AlgoExecutor(TWAP(...), ...)` for scheduled execution and `algo_comparison`
for TWAP-vs-VWAP analytics.

### 4.9 VectorForge — backtesting

```python
from vectorforge import (PortfolioData, MissingDataPolicy, VectorizedBacktester,
                         TargetWeights, CrossSectionalSignal, Rebalancer)

data = PortfolioData.from_dict({sym: ohlcv_df for sym, ohlcv_df in frames.items()})
data = data.align(policy=MissingDataPolicy.FORWARD_FILL)

# Option A: built-in cross-sectional signal
momentum = CrossSectionalSignal.momentum(lookback=126)
weights = momentum.generate(data).top_percentile(20)

# Option B: your own weight schedule (n_symbols x n_dates array)
weights = TargetWeights.from_array(w_matrix, list(data.symbols), data.dates)

result = VectorizedBacktester().run_portfolio(
    strategy=weights, data=data, initial_capital=1_000_000)
print(result.total_return, result.sharpe_ratio, result.max_drawdown)
print(result.equity_curve, result.turnover_history)
```

Key facts: VectorForge is the **independent referee** — whatever produced
your weights (ML, RL, optimizer), `run_portfolio` recomputes the equity
curve from raw prices. The pipeline uses exactly this to cross-check its
hand-computed returns.

---

## 5. The end-to-end pipeline, stage by stage

`pipeline.py` is the reference integration — read it when wiring products
together in your own code. What each stage does and hands to the next:

| # | Product | In | Out |
|---|---------|----|----|
| 1 | DataStream | seeded synthetic source (1% corrupted) | cleaned long frame + quality score |
| 2 | AlphaLab | `Panel.from_long(clean)` (outliers masked) | ranked 20-factor report |
| 3 | SentimentPulse | synthetic news for the same symbols | daily sentiment index (date × symbol) |
| 4 | SignalML | price/volume features **+ sentiment column**, walk-forward ridge + GBDT | OOS ensemble signal |
| 5 | PortfolioEngine | top-8 names by signal each 21 days, Ledoit-Wolf cov | risk-parity weight schedule |
| 6 | RiskGuard | daily strategy returns (weights lagged 1 day) | vol-targeted + circuit-broken returns, VaR/CVaR report |
| 7 | ExecutionCore | first rebalance as market orders into an intraday paper session | fill rate, slippage, commissions |
| 8 | DeepTrader | the strategy's own equity curve as the trading env | RL timing benchmark vs buy-and-hold |
| 9 | VectorForge | the same weight schedule + raw OHLCV | independent equity curve, Sharpe, maxDD |

Two details worth copying into real systems:

- **No look-ahead anywhere**: signals use a purged gap ≥ horizon, weights
  decided on day *t* earn returns from *t+1*, vol targeting and the breaker
  are lag-1, and the paper broker can't fill on the submission bar.
- **Independent verification**: stage 9 recomputes performance from prices
  and must agree with the hand-computed stage 6 numbers (it does, to within
  rounding) — if they diverge, you have a bug.

Expect *unimpressive* performance numbers: the universe is near-zero-drift
GBM, so an honest pipeline produces a near-flat strategy. The point is the
plumbing, not the alpha.

---

## 6. The Hub dashboard

- `hub/index.html` + `hub/app.js`: dependency-free renderer for
  `hub/data/products.js`.
- `hub/SCHEMA.md`: the JSON contract (schema v1) every demo emits —
  `summary_metrics`, `charts` (≤500 points, ≤5 series), `tables`, `notes`;
  no NaN/Infinity allowed.
- `run_all.py` validates every `hub/data/*.json` against the schema before
  bundling, and fails loudly on violations.

To add your own tile: write a JSON file following the schema into
`hub/data/`, then `python3 run_all.py --bundle-only`.

---

## 7. Development workflow

1. Branch, then make your change in the product's package.
2. Add/adjust tests in `<product>/tests/` (test-first per the project
   constitution; coverage gate is 70%).
3. `cd <product> && python3 -m pytest tests -q`
4. `python3 validate_all.py` — all five layers green.
5. If you changed a demo or the hub schema, eyeball the dashboard:
   `python3 run_all.py && python3 -m http.server -d hub 8080`.
6. Commit. Spec-driven feature work uses the `speckit.*` skills and
   `specs/<feature>/` documents; `VALIDATION.yaml` defines the CI contract
   for VectorForge.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: numpy` (etc.) | Install the dependency set from section 2. |
| `ModuleNotFoundError: <product>` in your own script | Product packages aren't pip-installed; run from the product directory or add product dirs to `sys.path` (see top of `pipeline.py`). |
| Demo fails with chart length errors | Hub schema caps charts at 500 x-points — downsample (see `downsample()` in `pipeline.py`). |
| `run_all.py` reports `non-finite literal` | Your JSON contains NaN/Infinity; replace with `null` or drop the points. |
| Absurd backtest returns on synthetic data | Unrepaired return outliers — QualityEngine only flags them; mask spikes (see `mask_return_outliers` in `pipeline.py`). |
| SignalML demo is slow (~35s) | Permutation importances on the GBDT model; pass smaller `max_iter` / `importance_sample` as in `pipeline.py`. |
| Hub page is blank | `hub/data/products.js` missing — run `python3 run_all.py` first. |
| Walk-forward IC looks too good | Check `gap >= horizon` in `WalkForwardEngine`; a smaller gap leaks labels. |

---

## 9. Where to go next

- Product deep-dives: each `<product>/README.md` and `docs/products/*.md`
- Architecture & roadmap: `docs/ECOSYSTEM_OVERVIEW.md`
- VectorForge ecosystem guide (EN/中文): `docs/guides/`
- Hub data contract: `hub/SCHEMA.md`
