# Changelog

All notable changes to VectorForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.3.0] - 2026-06-12

### Added

#### Multi-Asset Portfolio Support (`vectorforge/portfolio/`)
- `PortfolioData` container for multi-symbol OHLCV data: NumPy 3D array
  (symbols × dates × fields, float32) with `from_dict()`/`from_parquet()`
  constructors, calendar alignment (`align()` with `MissingDataPolicy`
  forward-fill/interpolate/drop/zero), validation (`validate()` with gap
  detection), tradeable masks, and per-symbol metadata (`SymbolMetadata`)
- Cross-sectional signals (`CrossSectionalSignal`): momentum, mean-reversion,
  volatility, relative-strength, and custom factories; percentile/fractional/
  ordinal ranking (`RankMethod`); sector-neutral ranking via `group_field`;
  `SignalResult.top_percentile()/bottom_percentile()` filtering and
  `to_weights()` with equal/market-cap weighting
- Rebalancing (`Rebalancer`): composable triggers (`CalendarTrigger`,
  `DriftTrigger`, `HybridTrigger`), calendar frequencies daily→annual,
  turnover constraints with largest-deviation prioritization,
  `compute_trades()` and full-simulation `run()`
- Portfolio metrics (`PortfolioMetrics`): HHI and top-N concentration,
  diversification ratio, correlation matrices (static and rolling), portfolio
  beta, sector exposure time series, per-asset return/risk attribution (MCTR),
  `from_backtest_result()` factory and `generate_report()`
- Corporate actions (`CorporateAction`): split adjustment for prices and
  positions, cash dividends with optional reinvestment (DRIP),
  `PortfolioData.apply_corporate_actions()`
- Engine integration: `VectorizedEngine.run_portfolio()` producing
  `PortfolioBacktestResult` with equity curve, per-asset contributions, and
  Sharpe/Sortino/drawdown; `PortfolioStrategy` base class with
  `generate_weights()`

### Changed
- `scipy` added as a runtime dependency (cross-sectional ranking)

### Fixed
- pandas 3.x compatibility (`fillna(method=...)` removal)

---

## [0.2.0] - 2026-01-07

### Added

#### Accelerated Computation Kernels (`vectorforge/engine/accelerated.py`)
- JAX JIT compilation for core backtest functions with `@jax.jit` decorator
- `vmap`-based vectorized parallel parameter testing for batch backtests
- Numba `@njit` optimization with `prange` for CPU parallelization
- Automatic backend selection: JAX (GPU) > Numba (CPU) > NumPy (fallback)
- Memory profiling utilities for optimization analysis
- `AcceleratedBacktester` class with unified interface for all backends

#### Advanced Order Types (`vectorforge/engine/event_driven.py`)
- Stop orders with configurable trigger logic
- Stop-limit orders combining stop triggers with limit execution
- Trailing stops (both absolute dollar and percentage-based)
- Trailing stop-limit orders
- Bracket orders with OCO (One-Cancels-Other) groups
- Time-in-force options: DAY, GTC, GTD, IOC, FOK, OPG, CLS

#### Exchange Calendar Integration (`vectorforge/calendar/exchange.py`)
- Support for 6 major exchanges:
  - NYSE (New York Stock Exchange)
  - NASDAQ
  - CME (Chicago Mercantile Exchange)
  - LSE (London Stock Exchange)
  - TSE (Tokyo Stock Exchange)
  - CRYPTO (24/7 cryptocurrency markets)
- Pre-market and after-hours session detection
- Holiday schedules with early close support (e.g., Christmas Eve)
- Trading days iteration and counting utilities
- Timezone-aware scheduling with proper DST handling

#### Hybrid Runner Intelligence (`vectorforge/engine/hybrid.py`)
- Strategy analysis via AST (Abstract Syntax Tree) inspection
- Detection of vectorizable operations (`generate_signals` method)
- Identification of event-driven requirements (`on_bar`, `on_fill` methods)
- Strategy complexity estimation and scoring (SIMPLE, MODERATE, COMPLEX)
- `run_adaptive()` method for automatic execution mode selection
- `get_discrepancy_report()` for comparing vectorized vs event-driven results
- Confidence-based mode recommendations

#### Vectorized Engine Enhancements (`vectorforge/engine/vectorized.py`)
- Memory-mapped arrays for datasets > 100k rows
- `run_batch_quick()` method for fast parameter sweeps
- Backend information reporting via `get_backend_info()`
- Automatic GPU scaling when JAX with CUDA is available

### Performance Improvements
- 10-year daily backtest: ~0.02s (5x faster than v0.1.0 target)
- 1000 parameter sweep: ~0.5s (4x faster than target)
- Memory usage: <1GB with streaming (50% better than target)
- Event processing: <1ms per bar

### Testing
- Added comprehensive test suite in `tests/test_v020_features.py`
- Tests for accelerated kernels, calendar integration, and advanced orders
- Benchmark tests for performance validation

---

## [0.1.0] - 2025-12-15

### Added

#### Core Engine Architecture
- `BaseEngine` abstract class defining the backtesting interface
- `VectorizedEngine` for fast historical backtests using NumPy/Pandas
- `EventDrivenEngine` for production-accurate simulation
- `HybridEngine` combining both modes with intelligent switching

#### Strategy Framework (`vectorforge/strategy/`)
- `BaseStrategy` abstract class with lifecycle hooks
- `generate_signals()` for vectorized signal generation
- `on_bar()` and `on_fill()` callbacks for event-driven logic
- `MomentumStrategy` example implementation
- `MovingAverageCrossover` example strategy
- `PositionManager` for tracking positions and P&L

#### Performance Analytics (`vectorforge/analysis/`)
- Sharpe ratio, Sortino ratio, Calmar ratio calculations
- Maximum drawdown and drawdown duration analysis
- Win rate, profit factor, average win/loss metrics
- Trade-by-trade analysis with entry/exit tracking
- Equity curve generation

#### Optimization Framework (`vectorforge/optimization/`)
- Grid search with parallel execution support
- Walk-forward optimization with rolling windows
- Purged K-Fold cross-validation for time series
- Parameter sensitivity analysis

#### Execution Models (`vectorforge/execution/`)
- Slippage models (fixed, percentage, volume-based)
- Commission models (per-trade, per-share, percentage)

#### Data Management (`vectorforge/data/`)
- Lookahead bias detection and prevention guards
- Survivorship bias guards
- Point-in-time universe tracking for historical index composition
- Data validation utilities

#### Configuration System
- YAML-based configuration via `BacktestConfig`
- Pydantic models for type-safe configuration
- Environment-aware settings

### Documentation
- Product README with quick start guide
- EXECUTION_PLAN.md with detailed roadmap
- Architecture documentation
- Integration guides (English and Chinese)

---

## Version History Summary

| Version | Release Date | Highlights |
|---------|--------------|------------|
| 0.2.0 | 2026-01-07 | Performance milestone: JAX/Numba acceleration, advanced orders, exchange calendars |
| 0.1.0 | 2025-12-15 | Foundation: dual-mode engine, strategy framework, optimization tools |

---

## Upgrade Guide

### From v0.1.0 to v0.2.0

No breaking changes. All existing code remains compatible.

**New features available:**

```python
# Use accelerated backtesting
from vectorforge.engine.accelerated import AcceleratedBacktester

backtester = AcceleratedBacktester(backend='jax')  # or 'numba', 'numpy'
results = backtester.run(data, strategy)

# Use exchange calendars
from vectorforge.calendar.exchange import ExchangeCalendar

nyse = ExchangeCalendar('NYSE')
if nyse.is_trading_day(date):
    # Execute trading logic
    pass

# Use adaptive hybrid runner
from vectorforge.engine.hybrid import HybridRunner

runner = HybridRunner(strategy, data)
results = runner.run_adaptive()  # Automatically selects best mode
```

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to VectorForge.

## License

VectorForge is released under the MIT License. See [LICENSE](../LICENSE) for details.
