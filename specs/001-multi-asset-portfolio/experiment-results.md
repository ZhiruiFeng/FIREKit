# Experiment Results: VectorForge v0.3.0 Multi-Asset Portfolio

**Date**: 2026-06-12
**Branch**: `claude/evaluate-research-design-gexq2p`
**Benchmark suite**: `vectorforge/tests/benchmark/` (run with `cd vectorforge && python3 -m pytest tests/benchmark -q -s`; total runtime ~40 s)

## Performance Experiments (SC-001..SC-008)

**Environment / hardware caveat**: all numbers were measured on a shared cloud container (Linux, Python 3.11.15, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1). JAX and Numba are **not installed**, so `VectorizedBacktester` ran on its pure-NumPy fallback backend (`backend_info: requested=jax, actual=numpy`). Timings are single-run wall-clock (`time.perf_counter`) after a small warm-up backtest; expect run-to-run variance of roughly ±10-20% on shared hardware. Data is synthetic geometric Brownian motion with fixed seeds (see `tests/benchmark/helpers.py`).

### SC-001 — 100-asset × 10-year backtest < 5 s

- **Target**: backtest a 100-asset portfolio over 10 years of daily data (2,520 business days) in under 5 seconds (vectorized mode).
- **Setup**: `VectorizedBacktester.run_portfolio` with equal-weight `TargetWeights`, monthly `Rebalancer.calendar(RebalanceFrequency.MONTHLY)` (116 rebalances executed).
- **Measured**: **0.386 s** (probe runs: 0.396–0.411 s).
- **Verdict**: **PASS** (~13× headroom, even without JAX/Numba acceleration).
- **Test**: `tests/benchmark/test_portfolio_performance.py::TestSC001BacktestSpeed`

### SC-002 — Portfolio metrics < 1 s after backtest

- **Target**: portfolio-level metrics (Sharpe, drawdown, sector exposure) computed within 1 second of backtest completion.
- **Setup**: `PortfolioMetrics.from_backtest_result(...)` on the SC-001 result (100 assets × 2,520 days, 10-sector map), timing `sharpe_ratio()` + `max_drawdown()` + `sector_exposure()`.
- **Measured**: **0.017 s**.
- **Verdict**: **PASS** (~60× headroom).
- **Test**: `tests/benchmark/test_portfolio_performance.py::TestSC002MetricsSpeed`

### SC-003 — 500-asset cross-sectional signal < 500 ms per period

- **Target**: cross-sectional signal generation for 500 assets in under 500 ms per period.
- **Setup / interpretation**: the API is batch, so we time the full `CrossSectionalSignal.momentum(lookback=252, skip_recent=21).generate(data)` (raw momentum + percentile ranking for every date) over 500 assets × 504 days, then divide by the 231 periods that produce a valid signal. This amortized per-period figure is conservative (it includes ranking work for all dates).
- **Measured**: 0.036 s total → **0.15 ms per period**. Downstream top-decile selection + weight construction (`top_percentile(10).to_weights()`) adds 0.02 ms per period.
- **Verdict**: **PASS** (~3,300× headroom).
- **Test**: `tests/benchmark/test_signals_performance.py::TestSC003SignalSpeed`

### SC-004 — Scaling to 1,000 assets no worse than linear

- **Target**: up to 1,000 assets without performance degradation greater than linear scaling (benchmark allows a generous 3× of linear extrapolation from the 100-asset baseline to avoid flaky failures).
- **Setup**: same monthly-rebalanced 10-year backtest at 100, 500, and 1,000 assets.
- **Initial measurement (bug found)**: the first run measured 0.41 s / 3.25 s / 12.5 s at 100/500/1,000 assets — ~30× time for 10× assets, clearly superlinear (≈quadratic) and over the 3×-linear limit at 1,000 assets (12.52 s vs 12.34 s). Root cause: in `VectorizedBacktester.run_portfolio` (`vectorforge/engine/vectorized.py`), the per-day rebalance check built `current_weights_dict` / `target_weights_dict` with comprehensions indexing `data.symbols[i]`, and the `PortfolioData.symbols` property returns a **copy of the full symbol list on every access** — O(n_symbols²) work per trading day, O(n² · T) overall.
- **Fix applied**: hoisted `symbols = data.symbols` and `dates = data.dates` out of the simulation loop (committed on this branch). Re-measured:

  | Assets | Backtest time (after fix) | 3×-linear limit |
  |-------:|--------------------------:|----------------:|
  | 100    | 0.211 s (baseline)        | —               |
  | 500    | 0.563 s                   | 3.17 s ✓        |
  | 1,000  | 1.055 s                   | 6.33 s ✓        |

  10× the assets now costs ~5× the time (sub-linear in this regime; fixed per-day Python-loop overhead dominates at small universes). The 1,000-asset 10-year backtest dropped from 12.5 s to ~1.1 s.
- **Verdict**: **PASS** (after fix; originally FAIL — the xfail marker has been removed from the test).
- **Test**: `tests/benchmark/test_memory_usage.py::TestSC004Scaling`

### SC-008 — Memory < 4 GB for 1,000 assets × 10 years

- **Target**: memory usage for 1,000 assets over 10 years remains under 4 GB.
- **Setup**: `tracemalloc` (tracks Python and NumPy allocations) around construction of the 1,000 × 2,520 × 5 `PortfolioData` plus a full monthly-rebalanced backtest; process-wide `ru_maxrss` reported as a cross-check.
- **Measured**: **0.172 GB tracemalloc peak**; process `ru_maxrss` 0.36 GB (includes interpreter + all loaded libraries). The float32 price cube itself is ~50 MB; result DataFrames (weights history, asset returns, contributions at 1,000 × 2,520 float64) dominate the rest.
- **Verdict**: **PASS** (~23× headroom vs 4 GB).
- **Test**: `tests/benchmark/test_memory_usage.py::TestSC008MemoryUsage`

### Summary

| Criterion | Target | Measured | Verdict |
|-----------|--------|----------|---------|
| SC-001 | 100 assets × 10 yr backtest < 5 s | 0.386 s | **PASS** |
| SC-002 | Metrics (Sharpe, drawdown, sector exposure) < 1 s | 0.017 s | **PASS** |
| SC-003 | 500-asset signal < 500 ms/period | 0.15 ms/period | **PASS** |
| SC-004 | ≤ linear scaling to 1,000 assets (3× tolerance) | ~5× time for 10× assets after O(n²) fix (1.06 s at 1,000 assets) | **PASS** (after fix) |
| SC-008 | < 4 GB for 1,000 assets × 10 yr | 0.17 GB traced peak (0.36 GB RSS) | **PASS** |

Note: SC-005/SC-006 (constraint and corporate-action correctness) are covered by the contract tests in `vectorforge/tests/contract/`, and SC-007 is a usability target; none of these are performance experiments, so they are out of scope for this report.
