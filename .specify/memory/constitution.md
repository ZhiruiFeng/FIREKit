<!--
SYNC IMPACT REPORT
==================
Version change: N/A → 1.0.0 (Initial ratification)
Modified principles: N/A (Initial creation)
Added sections:
  - Core Principles (5 principles)
  - Technology Standards
  - Development Workflow
  - Governance
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ (Constitution Check section compatible)
  - .specify/templates/spec-template.md ✅ (User stories and testing aligned)
  - .specify/templates/tasks-template.md ✅ (Testing phases aligned)
Follow-up TODOs: None
-->

# FIREKit Constitution

## Core Principles

### I. Library-First Architecture

Every product in the FIREKit ecosystem MUST be designed as an independent, self-contained library.

- Each product (VectorForge, DataStream, AlphaLab, etc.) MUST function as a standalone package with its own namespace
- Libraries MUST be independently installable, testable, and documented
- Inter-library dependencies MUST be explicit and version-pinned
- No circular dependencies between products; shared utilities belong in a dedicated `firekit.core` module
- Each library MUST expose both programmatic API and CLI interface

**Rationale**: Modular architecture enables independent development cycles, parallel testing, and selective deployment. Users can adopt individual products without the full ecosystem.

### II. Performance-First Design

Speed and efficiency are non-negotiable; FIREKit targets institutional-grade performance metrics.

- Vectorized operations (NumPy/JAX/Polars) MUST be preferred over iterative loops
- Hot paths MUST achieve documented performance targets (e.g., VectorForge: 1M+ trades/sec)
- Memory allocation in critical paths MUST be minimized; prefer pre-allocation and buffer reuse
- Performance-critical components MAY use Rust via PyO3 when Python cannot meet targets
- All performance claims MUST be backed by reproducible benchmarks in `benchmarks/`

**Rationale**: Quantitative trading demands speed. Slow backtests limit research iterations; slow execution loses money.

### III. Test-First Development

Tests MUST precede implementation; the Red-Green-Refactor cycle is mandatory for all feature work.

- Contract tests MUST define API behavior before implementation begins
- Integration tests MUST validate cross-product interactions (DataStream → VectorForge → SignalML)
- Tests MUST fail before implementation (verified by CI)
- Financial calculations MUST include edge case coverage: zero quantities, negative prices, overflow scenarios
- Backtest results MUST be deterministic and reproducible given identical inputs

**Rationale**: Trading systems handle real money. Bugs are expensive. Test-first development catches errors before they become costly.

### IV. Production-Parity

Code MUST behave identically in backtest and live trading environments.

- The same strategy code MUST run without modification in backtest, paper trading, and live execution
- Time handling MUST be explicit: no implicit `datetime.now()` calls in strategy logic
- All data access MUST go through abstraction layers that can switch between historical and live feeds
- Event handling MUST be deterministic; random operations MUST accept seed parameters
- Latency simulation in backtests MUST be configurable to match production conditions

**Rationale**: Strategies that work in backtest but fail in production are useless. Eliminating code divergence prevents this class of failures.

### V. Risk-First Execution

Safety mechanisms are non-negotiable; risk management MUST be embedded at every execution layer.

- Position sizing MUST use validated algorithms (Kelly Criterion, fixed fractional)
- All strategies MUST define maximum drawdown thresholds that trigger automatic derisking
- Circuit breakers MUST halt trading on anomalous market conditions or system errors
- Order validation MUST prevent fat-finger errors: quantity limits, price deviation checks
- Risk metrics (VaR, Sharpe, drawdown) MUST be computed and logged for every backtest and live session

**Rationale**: A single catastrophic loss can wipe out years of gains. Defensive design is mandatory.

## Technology Standards

- **Language**: Python 3.11+ as primary; Rust (PyO3) for performance-critical extensions
- **Data Processing**: Polars preferred for DataFrames; NumPy/JAX for numerical computation
- **Storage**: Parquet for historical data; TimescaleDB for time-series; Redis for caching
- **ML Frameworks**: LightGBM/XGBoost for tabular; PyTorch for deep learning; Stable-Baselines3 for RL
- **Testing**: pytest with pytest-benchmark; contract tests in `tests/contract/`; integration tests in `tests/integration/`
- **Type Safety**: Type hints MUST be used for all public APIs; mypy validation in CI
- **Documentation**: Docstrings for all public functions; architecture decisions in `docs/adr/`

## Development Workflow

### Code Review Requirements

- All changes MUST be submitted via pull request
- PRs MUST pass CI checks: tests, type checking, linting, benchmarks (no regression)
- PRs touching financial calculations MUST include test cases demonstrating correctness
- PRs adding dependencies MUST justify the addition and document license compatibility

### Quality Gates

- Tests MUST pass before merge
- Code coverage for new code MUST NOT decrease overall project coverage
- Performance benchmarks MUST NOT regress beyond defined thresholds
- All MUST/SHOULD requirements in this constitution MUST be verifiable in code review

### Branch Strategy

- `main` branch MUST always be deployable
- Feature branches follow pattern: `<issue-number>-<short-description>`
- Releases tagged with semantic versioning: `vMAJOR.MINOR.PATCH`

## Governance

This constitution supersedes all other development practices within FIREKit. Compliance is mandatory.

### Amendment Procedure

1. Propose changes via pull request to `.specify/memory/constitution.md`
2. Changes MUST include rationale and impact analysis
3. Version bump determined by scope: MAJOR (principle removal/redefinition), MINOR (new principle/expansion), PATCH (clarification/typo)
4. All dependent templates MUST be updated in the same PR

### Compliance Review

- Code reviewers MUST verify PR compliance with relevant principles
- Constitution violations MUST be documented with justification if waived
- Quarterly review of constitution applicability recommended

### Versioning Policy

- MAJOR: Backward-incompatible governance changes (principle removal, fundamental redefinition)
- MINOR: New principles added, existing principles materially expanded
- PATCH: Clarifications, wording improvements, typo fixes

**Version**: 1.0.0 | **Ratified**: 2026-01-07 | **Last Amended**: 2026-01-07
