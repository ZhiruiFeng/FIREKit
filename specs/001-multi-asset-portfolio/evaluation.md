# Evaluation: Research & Design Plans vs. Implementation

**Feature**: 001-multi-asset-portfolio (VectorForge v0.3.0)
**Date**: 2026-06-12
**Scope**: spec.md, plan.md, research.md, tasks.md, quickstart.md, contracts/ evaluated against the code on this branch.

## Verdict

The research and design plans are well-structured and the core implementation
exists and works (197 tests passing after a pandas 3.x compatibility fix).
However, **tasks.md overstates completion**: 14 of the 91 tasks marked `[X]`
were not actually done (all unit tests, all integration tests, all benchmarks,
quickstart validation, and the CHANGELOG/PROGRESS updates). Several design
decisions documented in research.md were also silently deviated from in code.

## Plan Quality Assessment

| Artifact | Quality | Notes |
|----------|---------|-------|
| spec.md | Good | Clear prioritized user stories, testable acceptance scenarios, measurable success criteria (SC-001..SC-008), explicit assumptions and out-of-scope list. |
| plan.md | Good | Constitution gate, sensible structure decision (extend, don't fork engines). One inaccuracy: claims "no new dependencies required" (see Gap 3). |
| research.md | Good | Alternatives tables with rejection rationale; concrete API sketches. Two decisions not honored by implementation (see Gaps 4–5). |
| tasks.md | Structurally good, status inaccurate | Dependency-ordered, story-grouped, parallelizable. But completion checkboxes do not reflect reality (see Gap 1). |
| quickstart.md | Good as design doc | Validated end-to-end during this evaluation; discrepancies corrected in place. |
| contracts/ | Good | Contract test suite exists and passes. |

## Gaps Found

### Gap 1 — Tasks falsely marked complete (high)

`tasks.md` marks all 91 tasks `[X]`, but these had no corresponding artifacts
in the repository:

- **Unit tests** T023, T035, T051, T069, T080 — `vectorforge/tests/unit/` did not exist.
- **Integration tests** T024, T036, T052, T070, T081 — `vectorforge/tests/integration/` did not exist.
- **Benchmarks** T085, T086, T087 — `vectorforge/tests/benchmark/` did not exist, so success criteria SC-001..SC-004 and SC-008 were never measured.
- **T089** (quickstart validation) — no evidence it was run; quickstart examples contained API discrepancies.
- **T090** — CHANGELOG.md still lists v0.3.0 under "Planned"/Unreleased.
- **T091** — PROGRESS.md still says "v0.3.0 Multi-Asset | 🚧 In Progress | 0%".

**Resolution**: missing test suites and benchmarks created and run as part of
this evaluation (see experiment-results.md); CHANGELOG/PROGRESS updated.

### Gap 2 — Version inconsistency (medium)

`vectorforge/__init__.py` declares `__version__ = "0.3.0"` while
`pyproject.toml` still says `version = "0.2.0"`. Resolved: pyproject bumped to 0.3.0.

### Gap 3 — Undeclared scipy dependency (medium)

`portfolio/signals.py` imports `scipy.stats` at module level, but scipy is not
in `pyproject.toml` dependencies, contradicting plan.md's "no new dependencies
required". A fresh install of vectorforge would crash on
`import vectorforge.portfolio.signals`. Resolved: scipy added to pyproject deps
(research.md itself proposed `scipy.stats.rankdata`, so the plan statement was
the error, not the import).

### Gap 4 — PortfolioMetrics does not extend PerformanceMetrics (low)

research.md (Decision 5) and task T058 specify
`class PortfolioMetrics(PerformanceMetrics)`. The implementation is a
standalone class that re-implements Sharpe/Sortino/drawdown. Functionally fine
(contract tests pass) but it is a deviation from the documented design and
duplicates metric code — worth either refactoring or amending research.md.

### Gap 5 — Pre-existing pandas 3.x incompatibility (fixed)

`tests/contract/test_metrics_contract.py` used `fillna(method="bfill")`
(removed in pandas 3.0), so 3 contract tests failed on a current environment.
Fixed with `.bfill()`.

### Minor

- 44 ruff findings repo-wide (14 in `portfolio/`), including two F821s in
  `corporate_actions.py` from a `"PortfolioData"` string annotation without a
  `TYPE_CHECKING` import — cosmetic, not a runtime bug.
- `engine/accelerated.py` imports JAX/Numba lazily, so the package works
  without them — good, and consistent with the optional-acceleration design.

## Design Adherence Check (research.md decisions)

| Decision | Implemented? |
|----------|--------------|
| NumPy 3D array (symbols × time × fields), float32 | ✅ `PortfolioData._prices` |
| Forward-fill alignment with MissingDataPolicy enum | ✅ |
| Percentile ranking via `scipy.stats.rankdata`, group neutralization | ✅ (but see Gap 3) |
| Composable RebalanceTrigger (Calendar/Drift/Hybrid) | ✅ |
| Turnover constraint with largest-deviation priority | ✅ |
| Extend PerformanceMetrics for PortfolioMetrics | ❌ standalone class (Gap 4) |
| Adjustment-factor corporate actions with DRIP | ✅ |
| `run_portfolio()` method on existing engine, not new class | ✅ |

## Experiment & Validation Summary

See `experiment-results.md` for measured performance numbers. Test/validation
work performed during this evaluation:

1. Full existing suite: **197/197 passing** (after Gap 5 fix).
2. New unit + integration suites covering US1–US5 acceptance scenarios.
3. New benchmark suite measuring SC-001, SC-002, SC-003, SC-004, SC-008.
4. Quickstart end-to-end validation script (`tests/quickstart_validation.py`).
