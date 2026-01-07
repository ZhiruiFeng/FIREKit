# VectorForge Execution Plan

This document outlines the implementation roadmap for VectorForge, the high-performance backtesting engine for FIREKit.

## Overview

VectorForge is the foundational backtesting infrastructure that enables rapid strategy research through vectorized operations while maintaining production-ready simulation capabilities through event-driven execution.

## Current Status: Phase 2 - Multi-Asset Portfolio Support (v0.3.0)

> **See [PROGRESS.md](./PROGRESS.md) for detailed task tracking and [CHANGELOG.md](./CHANGELOG.md) for version history.**

### Active Development Focus
The next major milestone focuses on enabling multi-asset portfolio backtesting. This unlocks diversified strategy research and cross-sectional signal generation.

---

## Completed Phases

### Phase 0 (v0.1.0) - Foundation Complete ✓
- [x] Project structure and package configuration
- [x] Core engine architecture (base, vectorized, event-driven, hybrid)
- [x] Strategy base classes and example strategies
- [x] Position management system
- [x] Performance metrics and analysis tools
- [x] Drawdown and trade analytics
- [x] Optimization framework (grid search, walk-forward, cross-validation)
- [x] Execution models (slippage, commission)
- [x] Data guards for bias prevention
- [x] Point-in-time universe tracking
- [x] Configuration system with YAML support

### Phase 1 (v0.2.0) - Performance & Engine Refinement Complete ✓
- [x] JAX JIT compilation for core backtest functions
- [x] vmap-based parallel parameter testing
- [x] Numba-optimized inner loops for CPU path
- [x] Memory-mapped arrays for large datasets
- [x] Advanced order types (stop-limit, trailing stop, bracket, TIF)
- [x] Exchange calendar integration (NYSE, NASDAQ, LSE, TSE, CRYPTO)
- [x] Hybrid Runner strategy analysis and adaptive execution
- [x] Comprehensive test suite for v0.2.0 features

---

## Phase 1: Core Engine Refinement ✓ COMPLETE

### 1.1 Vectorized Engine Enhancements ✓

**Objective**: Achieve target performance of 1M+ trades/second

#### Completed Tasks
1. **JAX Integration** ✓ - GPU acceleration enabled
   - [x] JIT compilation for signal generation (`accelerated.py`)
   - [x] vmap for batch parameter testing (`run_batch_quick`)
   - [x] GPU memory management with device selection

2. **Numba Optimization** ✓ - CPU performance boost
   - [x] Numba-compiled inner loops with `@njit`
   - [x] Parallel execution with `prange`
   - [x] Cached compiled functions

3. **Memory Efficiency** ✓
   - [x] Memory-mapped arrays via `enable_memory_mapping()`
   - [x] Automatic fallback to mmap for datasets > 100k rows
   - [x] Backend info reporting

#### Success Criteria - ACHIEVED
- 10-year daily backtest: ~0.02s (exceeded target of 0.1s)
- 1000 parameter sweep: ~0.5s (exceeded target of 2s)
- Memory usage: <1GB with streaming (exceeded target of 2GB)

### 1.2 Event-Driven Engine Refinement ✓

**Objective**: Production-accurate simulation matching live trading behavior

#### Completed Tasks
1. **Order Book Simulation**
   - [x] Partial fill simulation ready

2. **Advanced Order Types** ✓
   - [x] Stop orders with trigger logic
   - [x] Stop-limit orders
   - [x] Trailing stops (absolute and percentage)
   - [x] Trailing stop-limit orders
   - [x] Bracket orders (OCO groups)
   - [x] Time-in-force options (DAY, GTC, GTD, IOC, FOK, OPG, CLS)

3. **Market Hours & Holidays** ✓
   - [x] Exchange calendar integration (NYSE, NASDAQ, CME, LSE, TSE, CRYPTO)
   - [x] Pre-market and after-hours session detection
   - [x] Holiday schedule with early close support
   - [x] Trading days iteration and counting

#### Success Criteria - ACHIEVED
- Event processing: <1ms per bar
- Support for all standard order types: ✓
- Exchange calendar coverage: 6 major exchanges

### 1.3 Hybrid Runner Intelligence ✓

**Objective**: Automatic mode selection based on strategy requirements

#### Completed Tasks
1. **Strategy Analysis** ✓
   - [x] Detect vectorizable operations (generate_signals)
   - [x] Identify event-driven requirements (on_bar, on_fill)
   - [x] Estimate computation complexity (AST analysis)
   - [x] State and order logic detection

2. **Adaptive Execution** ✓
   - [x] `run_adaptive()` for automatic mode selection
   - [x] Validation with event-driven after vectorized sweep
   - [x] `get_discrepancy_report()` for mode comparison
   - [x] Recommendations based on discrepancy analysis

---

## Phase 2: Multi-Asset Portfolio Support (v0.3.0) - IN PROGRESS

### 2.1 Multi-Asset Portfolio Support

**Objective**: Backtest strategies across multiple assets simultaneously

**Target Files**:
- `vectorforge/portfolio/` (new module)
- `vectorforge/portfolio/data.py` - Multi-symbol data management
- `vectorforge/portfolio/signals.py` - Cross-sectional signal generation
- `vectorforge/portfolio/rebalance.py` - Rebalancing logic
- `vectorforge/portfolio/metrics.py` - Portfolio-level analytics

#### Tasks

1. **Portfolio Data Management** (Priority: P0)
   - [ ] `PortfolioData` class for multi-symbol OHLCV storage
   - [ ] `align_data()` - Synchronize different trading calendars
   - [ ] `handle_missing()` - Forward-fill, interpolation options
   - [ ] `apply_split()` - Stock split adjustment factors
   - [ ] `apply_dividend()` - Cash dividend handling
   - [ ] `validate_consistency()` - Cross-symbol data validation

2. **Portfolio-Level Signals** (Priority: P0)
   - [ ] `CrossSectionalSignal` base class
   - [ ] `rank_signals()` - Rank-based signal generation
   - [ ] `relative_strength()` - Momentum across universe
   - [ ] `SectorGrouping` - Industry classification support
   - [ ] `market_cap_weight()` - Size-based allocation

3. **Rebalancing Logic** (Priority: P0)
   - [ ] `CalendarRebalancer` - Daily/weekly/monthly triggers
   - [ ] `ThresholdRebalancer` - Drift-based triggers
   - [ ] `optimize_turnover()` - Transaction cost minimization
   - [ ] `TurnoverConstraint` - Max daily turnover limits

4. **Portfolio Metrics** (Priority: P1)
   - [ ] `portfolio_sharpe()` - Aggregate risk-adjusted returns
   - [ ] `rolling_correlation()` - Correlation matrix tracking
   - [ ] `sector_exposure()` - Allocation breakdown by sector
   - [ ] `diversification_ratio()` - Portfolio diversification measure
   - [ ] `concentration_metrics()` - HHI, top-N holdings weight

#### Success Criteria
- Support 100+ symbols in single backtest
- Rebalancing options: daily, weekly, monthly, threshold-based
- Corporate actions: splits and dividends handled correctly
- Portfolio metrics: Sharpe, correlation, sector exposure
- Performance: <1s for 10-year 50-stock portfolio backtest

### 2.2 Intraday & High-Frequency

**Objective**: Support minute and tick-level backtesting

#### Tasks
1. **Time Resolution**
   - Minute bar support
   - Tick data handling
   - Aggregation utilities

2. **Intraday Features**
   - VWAP/TWAP calculations
   - Intraday seasonality
   - Market microstructure effects

3. **Performance at Scale**
   - Efficient tick processing
   - Incremental calculations
   - Memory-efficient streaming

### 2.3 Derivatives Support

**Objective**: Options and futures strategy backtesting

#### Tasks
1. **Options Framework**
   - Greeks calculation
   - Option pricing models
   - Multi-leg strategy support

2. **Futures Framework**
   - Roll handling (perpetual, calendar)
   - Margin calculations
   - Continuous contract construction

---

## Phase 3: Robustness & Validation

### 3.1 Enhanced Bias Prevention

**Objective**: Eliminate all sources of bias in backtests

#### Tasks
1. **Lookahead Detection**
   - Automated lookahead scanning
   - Strategy code analysis
   - Runtime detection hooks

2. **Survivorship Bias**
   - Historical index composition DB
   - Delisting handling
   - Point-in-time data validation

3. **Data Quality**
   - Outlier detection
   - Data gap handling
   - Adjustment factor validation

### 3.2 Statistical Validation

**Objective**: Ensure statistically rigorous results

#### Tasks
1. **Monte Carlo Simulation**
   - Return path randomization
   - Bootstrap confidence intervals
   - Stress testing scenarios

2. **Regime Analysis**
   - Market regime detection
   - Conditional performance metrics
   - Regime-specific optimization

3. **Overfitting Detection**
   - Multiple hypothesis correction
   - Probability of backtest overfitting
   - Walk-forward efficiency ratio

### 3.3 Bayesian Optimization

**Objective**: Intelligent parameter search

#### Tasks
1. **Gaussian Process Optimization**
   - Surrogate model for objective
   - Acquisition function (EI, UCB)
   - Early stopping criteria

2. **Hyperband Integration**
   - Successive halving
   - Resource allocation
   - Parallel evaluation

---

## Phase 4: Integration & Ecosystem

### 4.1 DataStream Integration

**Objective**: Seamless data pipeline connectivity

#### Tasks
1. **Data Loading Interface**
   - Unified data API
   - Automatic format detection
   - Incremental data updates

2. **Caching Layer**
   - Local cache management
   - Cache invalidation rules
   - Memory vs disk caching

### 4.2 SignalML Integration

**Objective**: ML model signals in backtests

#### Tasks
1. **Model Interface**
   - Prediction API integration
   - Feature computation hooks
   - Model versioning

2. **Online Learning**
   - Rolling model updates
   - Incremental training
   - Model decay handling

### 4.3 RiskGuard Integration

**Objective**: Risk management during simulation

#### Tasks
1. **Position Sizing**
   - Kelly criterion
   - Risk parity
   - Volatility targeting

2. **Risk Limits**
   - Position limits
   - Sector exposure limits
   - Drawdown triggers

### 4.4 ExecutionCore Alignment

**Objective**: Identical backtest and live code paths

#### Tasks
1. **Code Sharing**
   - Shared strategy interface
   - Order type compatibility
   - Event format alignment

2. **Validation Mode**
   - Paper trading comparison
   - Execution quality analysis
   - Slippage model calibration

---

## Phase 5: Enterprise Features

### 5.1 Distributed Backtesting

**Objective**: Scale to massive parameter searches

#### Tasks
1. **Task Distribution**
   - Ray/Dask integration
   - Work queue management
   - Result aggregation

2. **Cloud Deployment**
   - Containerization
   - Auto-scaling
   - Spot instance support

### 5.2 Visualization & Dashboards

**Objective**: Interactive result exploration

#### Tasks
1. **Performance Dashboard**
   - Equity curve visualization
   - Drawdown charts
   - Trade analysis plots

2. **Parameter Analysis**
   - Heatmaps
   - 3D surface plots
   - Sensitivity analysis

### 5.3 Strategy Versioning

**Objective**: Track strategy evolution

#### Tasks
1. **Version Control**
   - Strategy snapshots
   - Parameter history
   - Performance tracking

2. **Comparison Tools**
   - A/B testing framework
   - Diff visualization
   - Regression detection

---

## Implementation Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| JAX GPU acceleration | High | Medium | P0 |
| Multi-asset support | High | High | P1 |
| Walk-forward validation | High | Low | P0 (Done) |
| Intraday support | Medium | High | P2 |
| Options/Futures | Medium | High | P3 |
| Distributed backtest | Medium | Medium | P2 |
| Bayesian optimization | Medium | Medium | P1 |
| Visualization | Low | Low | P3 |

---

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external dependencies
- Cover edge cases (empty data, single bar, etc.)

### Integration Tests
- End-to-end backtest runs
- Strategy + Engine combinations
- Data pipeline integration

### Performance Tests
- Benchmark standard scenarios
- Memory profiling
- GPU utilization tests

### Validation Tests
- Known strategy results comparison
- Cross-validation with other frameworks
- Statistical significance tests

---

## Documentation Plan

1. **API Reference** - Auto-generated from docstrings
2. **User Guide** - Step-by-step tutorials
3. **Examples** - Common strategy patterns
4. **Best Practices** - Bias prevention, optimization tips
5. **Architecture** - Design decisions and rationale

---

## Dependencies

### Required
- numpy >= 1.26.0
- pandas >= 2.1.0
- pydantic >= 2.5.0
- pyyaml >= 6.0

### Optional (Performance)
- jax >= 0.4.20 (GPU acceleration)
- numba >= 0.58.0 (CPU optimization)
- polars >= 0.20.0 (Fast data processing)

### Optional (Enterprise)
- ray >= 2.9.0 (Distributed computing)
- plotly >= 5.18.0 (Visualization)
- mlflow >= 2.9.0 (Experiment tracking)

---

## Milestones

### v0.1.0 - Foundation ✓ COMPLETE
- Core engine architecture
- Basic strategies
- Performance metrics
- Walk-forward validation

### v0.2.0 - Performance ✓ COMPLETE
- JAX JIT compilation and vmap parallelization
- Numba-optimized inner loops
- Memory-mapped arrays for large datasets
- Advanced order types (stop-limit, trailing, bracket, TIF)
- Exchange calendar integration
- Hybrid Runner strategy analysis

### v0.3.0 - Multi-Asset (IN PROGRESS)
- [ ] Portfolio data management (multi-symbol OHLCV)
- [ ] Cross-sectional signal generation
- [ ] Calendar and threshold-based rebalancing
- [ ] Corporate actions handling (splits, dividends)
- [ ] Portfolio-level metrics (Sharpe, correlation, exposure)

### v0.4.0 - Validation
- Monte Carlo simulation
- Bayesian optimization
- Statistical tests

### v1.0.0 - Production Ready
- Full ecosystem integration
- Enterprise features
- Comprehensive documentation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| JAX compatibility | Fallback to NumPy/Numba |
| Memory limits | Streaming processing |
| Complexity creep | Modular architecture |
| Integration issues | Well-defined interfaces |
| Performance regression | Automated benchmarks |
