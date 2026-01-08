# Tasks: VectorForge v0.3.0 Multi-Asset Portfolio Support

**Input**: Design documents from `/specs/001-multi-asset-portfolio/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included per FIREKit Constitution (Test-First Development principle)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Library package**: `vectorforge/vectorforge/` for source
- **Tests**: `vectorforge/tests/` for test files

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create portfolio subpackage structure and shared types

- [X] T001 Create portfolio subpackage directory structure at vectorforge/vectorforge/portfolio/
- [X] T002 Create portfolio __init__.py with public exports at vectorforge/vectorforge/portfolio/__init__.py
- [X] T003 [P] Add MissingDataPolicy enum and SymbolMetadata dataclass to vectorforge/vectorforge/portfolio/data.py
- [X] T004 [P] Create test fixtures with sample multi-asset data at vectorforge/tests/fixtures/portfolio_data.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core PortfolioData container that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until PortfolioData is complete

### Contract Tests (Write First, Must Fail)

- [X] T005 [P] Contract test for PortfolioData basic construction in vectorforge/tests/contract/test_portfolio_data_contract.py
- [X] T006 [P] Contract test for PortfolioData alignment methods in vectorforge/tests/contract/test_portfolio_data_contract.py
- [X] T007 [P] Contract test for PortfolioData validation rules in vectorforge/tests/contract/test_portfolio_data_contract.py

### Implementation

- [X] T008 Implement PortfolioData class with from_dict() constructor in vectorforge/vectorforge/portfolio/data.py
- [X] T009 Implement PortfolioData.align() with forward-fill and configurable policies in vectorforge/vectorforge/portfolio/data.py
- [X] T010 Implement PortfolioData.validate() with gap detection and consistency checks in vectorforge/vectorforge/portfolio/data.py
- [X] T011 Implement PortfolioData property accessors (close, open, high, low, volume, returns) in vectorforge/vectorforge/portfolio/data.py
- [X] T012 [P] Implement PortfolioData.to_parquet() and from_parquet() in vectorforge/vectorforge/portfolio/data.py
- [X] T013 [P] Add PortfolioData to main package exports in vectorforge/vectorforge/__init__.py

**Checkpoint**: PortfolioData foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Backtest a Multi-Asset Strategy (Priority: P1) 🎯 MVP

**Goal**: Enable running backtests across multiple assets with portfolio-level results

**Independent Test**: Load price data for 10+ symbols, run equal-weight strategy, verify aggregate returns

### Contract Tests for User Story 1

- [X] T014 [P] [US1] Contract test for PortfolioBacktestResult in vectorforge/tests/contract/test_portfolio_backtest_contract.py
- [X] T015 [P] [US1] Contract test for VectorizedEngine.run_portfolio() in vectorforge/tests/contract/test_portfolio_backtest_contract.py

### Implementation for User Story 1

- [X] T016 [P] [US1] Create PortfolioBacktestResult dataclass extending BacktestResult in vectorforge/vectorforge/engine/base.py
- [X] T017 [P] [US1] Create PortfolioStrategy base class with generate_weights() in vectorforge/vectorforge/strategy/base.py
- [X] T018 [US1] Create TargetWeights dataclass with equal_weight() factory in vectorforge/vectorforge/portfolio/signals.py
- [X] T019 [US1] Implement VectorizedEngine.run_portfolio() method in vectorforge/vectorforge/engine/vectorized.py
- [X] T020 [US1] Implement portfolio equity curve calculation from weighted returns in vectorforge/vectorforge/engine/vectorized.py
- [X] T021 [US1] Implement per-asset contribution tracking in PortfolioBacktestResult in vectorforge/vectorforge/engine/vectorized.py
- [X] T022 [US1] Add basic portfolio Sharpe/Sortino/drawdown calculation in vectorforge/vectorforge/engine/vectorized.py
- [X] T023 [P] [US1] Unit tests for portfolio backtest in vectorforge/tests/unit/test_portfolio_backtest.py

### Integration Test for User Story 1

- [X] T024 [US1] Integration test: 50-symbol momentum backtest in vectorforge/tests/integration/test_portfolio_backtest.py

**Checkpoint**: User Story 1 complete - can backtest multi-asset portfolios with basic metrics

---

## Phase 4: User Story 2 - Cross-Sectional Signal Generation (Priority: P1)

**Goal**: Generate relative signals by ranking assets against each other

**Independent Test**: Generate momentum ranking for 20 assets, verify top 5 are correctly identified

### Contract Tests for User Story 2

- [X] T025 [P] [US2] Contract test for CrossSectionalSignal in vectorforge/tests/contract/test_signals_contract.py
- [X] T026 [P] [US2] Contract test for SignalResult filtering methods in vectorforge/tests/contract/test_signals_contract.py

### Implementation for User Story 2

- [X] T027 [P] [US2] Create RankMethod enum in vectorforge/vectorforge/portfolio/signals.py
- [X] T028 [P] [US2] Create SignalResult dataclass with top_percentile/bottom_percentile methods in vectorforge/vectorforge/portfolio/signals.py
- [X] T029 [US2] Implement CrossSectionalSignal base class with generate() method in vectorforge/vectorforge/portfolio/signals.py
- [X] T030 [US2] Implement CrossSectionalSignal.momentum() factory method in vectorforge/vectorforge/portfolio/signals.py
- [X] T031 [US2] Implement CrossSectionalSignal.mean_reversion() factory method in vectorforge/vectorforge/portfolio/signals.py
- [X] T032 [US2] Implement CrossSectionalSignal.volatility() factory method in vectorforge/vectorforge/portfolio/signals.py
- [X] T033 [US2] Implement sector-neutral ranking with group_field parameter in vectorforge/vectorforge/portfolio/signals.py
- [X] T034 [US2] Implement SignalResult.to_weights() with equal/market_cap methods in vectorforge/vectorforge/portfolio/signals.py
- [X] T035 [P] [US2] Unit tests for cross-sectional signals in vectorforge/tests/unit/test_signals.py

### Integration Test for User Story 2

- [X] T036 [US2] Integration test: momentum signal → top decile → equal weight portfolio in vectorforge/tests/integration/test_signals_integration.py

**Checkpoint**: User Story 2 complete - can generate ranked signals and convert to weights

---

## Phase 5: User Story 3 - Configure Portfolio Rebalancing (Priority: P1)

**Goal**: Support calendar-based and drift-based rebalancing with turnover constraints

**Independent Test**: Run monthly rebalancing, verify trades only occur at month boundaries

### Contract Tests for User Story 3

- [X] T037 [P] [US3] Contract test for RebalanceTrigger classes in vectorforge/tests/contract/test_rebalancer_contract.py
- [X] T038 [P] [US3] Contract test for Rebalancer.compute_trades() in vectorforge/tests/contract/test_rebalancer_contract.py
- [X] T039 [P] [US3] Contract test for turnover constraint enforcement in vectorforge/tests/contract/test_rebalancer_contract.py

### Implementation for User Story 3

- [X] T040 [P] [US3] Create RebalanceFrequency enum in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T041 [P] [US3] Create RebalanceOrders dataclass in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T042 [P] [US3] Create RebalanceResult dataclass in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T043 [US3] Implement RebalanceTrigger abstract base class in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T044 [US3] Implement CalendarTrigger with daily/weekly/monthly/quarterly/annual in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T045 [US3] Implement DriftTrigger with threshold-based triggering in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T046 [US3] Implement HybridTrigger combining calendar and drift in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T047 [US3] Implement Rebalancer.compute_trades() with turnover constraint in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T048 [US3] Implement turnover optimization (prioritize largest deviations) in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T049 [US3] Implement Rebalancer.run() for full simulation in vectorforge/vectorforge/portfolio/rebalancer.py
- [X] T050 [US3] Integrate Rebalancer with VectorizedEngine.run_portfolio() in vectorforge/vectorforge/engine/vectorized.py
- [X] T051 [P] [US3] Unit tests for rebalancer in vectorforge/tests/unit/test_rebalancer.py

### Integration Test for User Story 3

- [X] T052 [US3] Integration test: monthly rebalance with 20% turnover limit in vectorforge/tests/integration/test_rebalancer_integration.py

**Checkpoint**: User Story 3 complete - can run portfolio backtests with realistic rebalancing

---

## Phase 6: User Story 4 - Analyze Portfolio-Level Metrics (Priority: P2)

**Goal**: Provide concentration, diversification, and sector exposure analytics

**Independent Test**: Run backtest, request HHI and top-5 holdings, verify formulas

### Contract Tests for User Story 4

- [X] T053 [P] [US4] Contract test for PortfolioMetrics concentration methods in vectorforge/tests/contract/test_metrics_contract.py
- [X] T054 [P] [US4] Contract test for PortfolioMetrics diversification methods in vectorforge/tests/contract/test_metrics_contract.py
- [X] T055 [P] [US4] Contract test for PortfolioMetrics sector exposure in vectorforge/tests/contract/test_metrics_contract.py

### Implementation for User Story 4

- [X] T056 [P] [US4] Create ConcentrationMetrics and DiversificationMetrics dataclasses in vectorforge/vectorforge/portfolio/metrics.py
- [X] T057 [P] [US4] Create SectorExposure and ReturnAttribution dataclasses in vectorforge/vectorforge/portfolio/metrics.py
- [X] T058 [US4] Implement PortfolioMetrics class extending PerformanceMetrics in vectorforge/vectorforge/portfolio/metrics.py
- [X] T059 [US4] Implement herfindahl_index() and concentration_metrics() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T060 [US4] Implement diversification_ratio() and diversification_metrics() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T061 [US4] Implement sector_exposure() and sector_exposure_at() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T062 [US4] Implement portfolio_beta() and rolling_beta() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T063 [US4] Implement correlation_matrix() and rolling_correlation() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T064 [US4] Implement contribution_to_return() and return_attribution() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T065 [US4] Implement contribution_to_risk() using MCTR in vectorforge/vectorforge/portfolio/metrics.py
- [X] T066 [US4] Implement top_n_concentration() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T067 [US4] Implement generate_report() and to_dataframe() in vectorforge/vectorforge/portfolio/metrics.py
- [X] T068 [US4] Add PortfolioMetrics.from_backtest_result() factory in vectorforge/vectorforge/portfolio/metrics.py
- [X] T069 [P] [US4] Unit tests for portfolio metrics in vectorforge/tests/unit/test_portfolio_metrics.py

### Integration Test for User Story 4

- [X] T070 [US4] Integration test: full metrics analysis on completed backtest in vectorforge/tests/integration/test_metrics_integration.py

**Checkpoint**: User Story 4 complete - comprehensive portfolio analytics available

---

## Phase 7: User Story 5 - Handle Corporate Actions (Priority: P2)

**Goal**: Correctly process stock splits and dividends during backtests

**Independent Test**: Run backtest with 2:1 split, verify position quantity doubles and value unchanged

### Contract Tests for User Story 5

- [X] T071 [P] [US5] Contract test for CorporateAction dataclass in vectorforge/tests/contract/test_corporate_actions_contract.py
- [X] T072 [P] [US5] Contract test for PortfolioData.apply_corporate_actions() in vectorforge/tests/contract/test_corporate_actions_contract.py

### Implementation for User Story 5

- [X] T073 [P] [US5] Create CorporateAction dataclass in vectorforge/vectorforge/portfolio/corporate_actions.py
- [X] T074 [US5] Implement split adjustment logic for prices in vectorforge/vectorforge/portfolio/corporate_actions.py
- [X] T075 [US5] Implement split adjustment logic for positions in vectorforge/vectorforge/portfolio/corporate_actions.py
- [X] T076 [US5] Implement dividend cash payout logic in vectorforge/vectorforge/portfolio/corporate_actions.py
- [X] T077 [US5] Implement dividend reinvestment (DRIP) logic in vectorforge/vectorforge/portfolio/corporate_actions.py
- [X] T078 [US5] Implement PortfolioData.apply_corporate_actions() method in vectorforge/vectorforge/portfolio/data.py
- [X] T079 [US5] Integrate corporate actions with backtest execution in vectorforge/vectorforge/engine/vectorized.py
- [X] T080 [P] [US5] Unit tests for corporate actions in vectorforge/tests/unit/test_corporate_actions.py

### Integration Test for User Story 5

- [X] T081 [US5] Integration test: backtest with splits and dividends in vectorforge/tests/integration/test_corporate_actions_integration.py

**Checkpoint**: User Story 5 complete - corporate actions handled correctly

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T082 [P] Update portfolio exports in vectorforge/vectorforge/portfolio/__init__.py
- [X] T083 [P] Update main package exports in vectorforge/vectorforge/__init__.py
- [X] T084 [P] Add portfolio config options to vectorforge/vectorforge/config.py
- [X] T085 Performance benchmark: 100-asset 10-year backtest < 5s in vectorforge/tests/benchmark/test_portfolio_performance.py
- [X] T086 Performance benchmark: 500-asset signal generation < 500ms in vectorforge/tests/benchmark/test_signals_performance.py
- [X] T087 Performance benchmark: memory < 4GB for 1000 assets in vectorforge/tests/benchmark/test_memory_usage.py
- [X] T088 [P] Add type stubs for public APIs in vectorforge/vectorforge/portfolio/py.typed
- [X] T089 Run quickstart.md validation examples
- [X] T090 Update CHANGELOG.md with v0.3.0 changes at vectorforge/CHANGELOG.md
- [X] T091 Update PROGRESS.md to mark v0.3.0 complete at vectorforge/PROGRESS.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (P1): Can start after Phase 2
  - US2 (P1): Can start after Phase 2, integrates with US1
  - US3 (P1): Can start after Phase 2, integrates with US1+US2
  - US4 (P2): Can start after Phase 2, uses US1 results
  - US5 (P2): Can start after Phase 2, integrates with backtest flow
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

| Story | Depends On | Can Parallelize With |
|-------|------------|----------------------|
| US1 (Backtest) | Foundational | - |
| US2 (Signals) | Foundational | US1 (different files) |
| US3 (Rebalancing) | US1 (uses TargetWeights) | US4, US5 |
| US4 (Metrics) | US1 (uses BacktestResult) | US3, US5 |
| US5 (Corp Actions) | Foundational | US2, US3, US4 |

### Within Each User Story

1. Contract tests MUST be written and FAIL before implementation
2. Models/dataclasses before logic
3. Core implementation before integration
4. Unit tests validate implementation
5. Integration test confirms story works end-to-end

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004 can run in parallel

**Phase 2 (Foundational)**:
- T005, T006, T007 (contract tests) can run in parallel
- T012, T013 can run in parallel after core implementation

**Phase 3 (US1)**:
- T014, T015 (contract tests) can run in parallel
- T016, T017 can run in parallel (different files)
- T023 (unit tests) after implementation

**Phase 4 (US2)**:
- T025, T026 (contract tests) can run in parallel
- T027, T028 can run in parallel (same file but independent)
- T035 (unit tests) after implementation

**Phase 5 (US3)**:
- T037, T038, T039 (contract tests) can run in parallel
- T040, T041, T042 can run in parallel (dataclasses)
- T051 (unit tests) after implementation

**Phase 6 (US4)**:
- T053, T054, T055 (contract tests) can run in parallel
- T056, T057 can run in parallel (dataclasses)
- T069 (unit tests) after implementation

**Phase 7 (US5)**:
- T071, T072 (contract tests) can run in parallel
- T073 can start immediately
- T080 (unit tests) after implementation

**Phase 8 (Polish)**:
- T082, T083, T084, T088 can all run in parallel

---

## Parallel Example: User Story 2 (Signals)

```bash
# Launch contract tests together:
Task: "Contract test for CrossSectionalSignal in tests/contract/test_signals_contract.py"
Task: "Contract test for SignalResult filtering methods in tests/contract/test_signals_contract.py"

# Launch dataclasses together:
Task: "Create RankMethod enum in vectorforge/portfolio/signals.py"
Task: "Create SignalResult dataclass in vectorforge/portfolio/signals.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - PortfolioData)
3. Complete Phase 3: User Story 1 (Multi-Asset Backtest)
4. **STOP and VALIDATE**: Test 50-symbol backtest independently
5. Deploy/demo if ready - core portfolio backtesting works!

### Incremental Delivery

1. **Foundation** → PortfolioData ready
2. **+ US1** → Can backtest portfolios (MVP!)
3. **+ US2** → Can generate cross-sectional signals
4. **+ US3** → Can rebalance with constraints
5. **+ US4** → Full analytics suite
6. **+ US5** → Corporate actions support
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 + User Story 3 (backtest flow)
   - Developer B: User Story 2 + User Story 4 (signals + metrics)
   - Developer C: User Story 5 (corporate actions)
3. Stories integrate at checkpoints

---

## Summary

| Phase | Tasks | Parallel Tasks | Story Coverage |
|-------|-------|----------------|----------------|
| Setup | 4 | 2 | - |
| Foundational | 9 | 4 | PortfolioData (shared) |
| US1 | 11 | 4 | Multi-Asset Backtest |
| US2 | 12 | 4 | Cross-Sectional Signals |
| US3 | 16 | 7 | Rebalancing |
| US4 | 18 | 6 | Portfolio Metrics |
| US5 | 11 | 4 | Corporate Actions |
| Polish | 10 | 5 | Cross-cutting |
| **Total** | **91** | **36** | 5 User Stories |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify contract tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Follow FIREKit Constitution: Test-First Development, Performance-First Design
