# Implementation Plan: VectorForge v0.3.0 Multi-Asset Portfolio Support

**Branch**: `001-multi-asset-portfolio` | **Date**: 2026-01-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-multi-asset-portfolio/spec.md`

## Summary

Enable VectorForge to backtest portfolio strategies across multiple assets simultaneously, with cross-sectional signal generation, rebalancing logic, and portfolio-level metrics. This extends the existing single-asset engine architecture to handle multi-symbol data containers, universe-wide ranking signals, calendar/drift-based rebalancing, and aggregate portfolio analytics.

## Technical Context

**Language/Version**: Python 3.11+ (consistent with existing VectorForge)
**Primary Dependencies**: NumPy, Pandas, Polars, JAX, Numba (existing); no new dependencies required
**Storage**: In-memory arrays/DataFrames; Parquet for persistence (existing patterns)
**Testing**: pytest with pytest-benchmark (existing)
**Target Platform**: Linux/macOS/Windows (Python)
**Project Type**: Single library package (vectorforge)
**Performance Goals**:
- 100-asset × 10-year daily backtest < 5 seconds (vectorized)
- 500-asset cross-sectional signal < 500ms per period
- 1,000-asset linear scaling
**Constraints**:
- Memory < 4GB for 1,000 assets × 10 years
- Maintain v0.2.0 API compatibility for single-asset strategies
**Scale/Scope**: Extend existing 7,600 LOC codebase; target ~2,000 new LOC

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Library-First Architecture** | ✅ Pass | New modules added within `vectorforge` package; no new packages or circular deps |
| **II. Performance-First Design** | ✅ Pass | Vectorized operations for signals/rebalancing; JAX/Numba for hot paths |
| **III. Test-First Development** | ✅ Pass | Contract tests define API before implementation |
| **IV. Production-Parity** | ✅ Pass | Same strategy code works in vectorized and event-driven modes |
| **V. Risk-First Execution** | ✅ Pass | Turnover constraints, drift thresholds built into rebalancing logic |

**Technology Standards Alignment**:
- ✅ Python 3.11+ (required by existing pyproject.toml)
- ✅ Polars/NumPy/JAX for computation (existing deps)
- ✅ pytest with benchmarks (existing test framework)
- ✅ Type hints for all public APIs

## Project Structure

### Documentation (this feature)

```text
specs/001-multi-asset-portfolio/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── portfolio_data.py
│   ├── signals.py
│   ├── rebalancing.py
│   └── metrics.py
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
vectorforge/vectorforge/
├── __init__.py                    # Add portfolio exports
├── config.py                      # Extend with portfolio config
│
├── portfolio/                     # NEW: Multi-asset support
│   ├── __init__.py
│   ├── data.py                    # PortfolioData container
│   ├── signals.py                 # CrossSectionalSignal
│   ├── rebalancer.py              # Rebalancing logic
│   ├── metrics.py                 # Portfolio-level metrics
│   └── corporate_actions.py       # Splits/dividends handling
│
├── engine/
│   ├── base.py                    # Extend BacktestResult for portfolio
│   ├── vectorized.py              # Add run_portfolio() method
│   ├── event_driven.py            # Add multi-asset position tracking
│   └── hybrid.py                  # Portfolio mode support
│
├── strategy/
│   └── base.py                    # Add PortfolioStrategy base class
│
└── analysis/
    └── metrics.py                 # Extend for portfolio analytics

tests/
├── contract/
│   ├── test_portfolio_data_contract.py
│   ├── test_signals_contract.py
│   ├── test_rebalancer_contract.py
│   └── test_metrics_contract.py
├── integration/
│   └── test_portfolio_backtest.py
└── unit/
    ├── test_portfolio_data.py
    ├── test_signals.py
    ├── test_rebalancer.py
    └── test_portfolio_metrics.py
```

**Structure Decision**: Extend existing single-package structure. New `portfolio/` subpackage contains multi-asset specific code. Engine classes gain new methods rather than parallel classes. This maintains backward compatibility and avoids code duplication.

## Complexity Tracking

> No constitution violations requiring justification.

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| Data alignment | Forward-fill with configurable options | Industry standard; simple to implement |
| Signal ranking | Percentile-based (0-100) | Consistent with academic literature |
| Rebalancing | Strategy pattern (Calendar, Drift, Hybrid) | Clean separation; extensible |
| Metrics | Extend existing PerformanceMetrics | Reuse code; consistent API |
