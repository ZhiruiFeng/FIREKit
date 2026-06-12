"""
Quickstart validation script for specs/001-multi-asset-portfolio/quickstart.md.

Reproduces every section of the quickstart against the real implementation,
using synthetic OHLCV data (geometric Brownian motion, fixed seed) instead of
the parquet files the doc references.

Run from /home/user/FIREKit/vectorforge:

    python3 tests/quickstart_validation.py

This is intentionally NOT named test_*.py so pytest does not collect it.

Discrepancies between the original quickstart.md and the real implementation
are recorded in DISCREPANCIES below; the code here uses the *correct* real API
so the script runs end to end.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from datetime import date

import numpy as np
import pandas as pd

# Make `vectorforge` importable when the package is not pip-installed and the
# script is invoked as `python3 tests/quickstart_validation.py` (sys.path[0]
# is then tests/, not the repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Quickstart-vs-implementation discrepancies found while writing this script.
# Each was confirmed against the library source and fixed in quickstart.md.
# ---------------------------------------------------------------------------
DISCREPANCIES = [
    "[Sections 5, 8] Doc imports/instantiates `VectorizedEngine` from "
    "vectorforge; no such class exists. Real class: `VectorizedBacktester` "
    "(vectorforge/engine/vectorized.py, exported from vectorforge/__init__.py).",
    "[Section 7] Doc constructs `CorporateAction(symbol=..., action_type='split', "
    "effective_date=..., adjustment_factor=...)` and `CorporateAction(..., "
    "action_type='dividend', cash_amount=..., reinvest=True)`. `CorporateAction` "
    "is an abstract frozen dataclass with only (symbol, ex_date) and cannot be "
    "instantiated; there are no `action_type`/`effective_date`/`adjustment_factor`/"
    "`cash_amount`/`reinvest` constructor kwargs. Real API: `Split(symbol, ex_date, "
    "ratio)` and `Dividend(symbol, ex_date, amount)` subclasses; dividend "
    "reinvestment is not a constructor flag (DRIP helper is "
    "`Dividend.reinvestment_shares()`); `apply_corporate_actions` only adjusts "
    "historical prices.",
    "[Section 2] Doc prints `len(weights.get_weights_at(weights.dates[0]))` as the "
    "number of *selected* assets. `SignalResult.to_weights()` returns a dense array "
    "with 0.0 (not NaN) for unselected assets, and `get_weights_at` only filters "
    "NaN, so len() always equals the full universe size. Also, at dates[0] the "
    "momentum signal is NaN (needs `lookback` history) so all weights are 0. The "
    "doc must count strictly positive weights at a date with valid signals.",
]


# ---------------------------------------------------------------------------
# Synthetic data: 30 symbols x 3 years of business days, GBM with fixed seed.
# ---------------------------------------------------------------------------
N_SYMBOLS = 30
START = "2021-01-01"
END = "2023-12-31"
SEED = 42

SECTORS = ["Technology", "Financials", "Healthcare", "Energy", "Consumer"]


def make_symbols(n: int) -> list[str]:
    return [f"SYM{i:02d}" for i in range(n)]


def make_synthetic_ohlcv(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Generate per-symbol OHLCV DataFrames via geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, END)
    n_dates = len(dates)
    symbols = make_symbols(N_SYMBOLS)

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        s0 = rng.uniform(20.0, 200.0)
        mu = rng.normal(0.08, 0.10) / 252.0  # daily drift
        sigma = rng.uniform(0.15, 0.40) / np.sqrt(252.0)  # daily vol
        z = rng.standard_normal(n_dates)
        log_ret = (mu - 0.5 * sigma**2) + sigma * z
        close = s0 * np.exp(np.cumsum(log_ret))

        open_ = np.empty_like(close)
        open_[0] = s0
        open_[1:] = close[:-1] * (1 + rng.normal(0, 0.002, n_dates - 1))
        spread = np.abs(rng.normal(0, 0.005, n_dates))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        volume = rng.integers(100_000, 5_000_000, n_dates).astype(float)

        data[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    # Drop a few random rows from two symbols so align()/forward-fill has work to do
    # (drop single weekdays only, so date gaps stay <= weekend size).
    drop_a = dates[[100, 101, 350]]
    drop_b = dates[[500, 501]]
    data["SYM05"] = data["SYM05"].drop(index=drop_a)
    data["SYM17"] = data["SYM17"].drop(index=drop_b)
    return data


def sector_for(symbol: str) -> str:
    return SECTORS[int(symbol[3:]) % len(SECTORS)]


# ---------------------------------------------------------------------------
# Section runners. Each mirrors a quickstart.md section.
# ---------------------------------------------------------------------------

def section_1(state: dict) -> None:
    """1. Loading Multi-Asset Data."""
    from vectorforge.portfolio import MissingDataPolicy, PortfolioData

    raw = make_synthetic_ohlcv()
    portfolio_data = PortfolioData.from_dict(raw)

    aligned = portfolio_data.align(
        policy=MissingDataPolicy.FORWARD_FILL,
        start_date=START,
        end_date=END,
    )

    validated = aligned.validate(min_dates=252, max_gap_days=5)
    print(f"  Universe: {validated.n_symbols} symbols x {validated.n_dates} dates")
    assert validated.n_symbols == N_SYMBOLS
    assert validated.n_dates >= 252
    state["validated"] = validated


def section_2(state: dict) -> None:
    """2. Generating Cross-Sectional Signals."""
    from vectorforge.portfolio import CrossSectionalSignal

    validated = state["validated"]

    momentum = CrossSectionalSignal.momentum(lookback=252, skip_recent=21)
    signals = momentum.generate(validated)
    top_decile = signals.top_percentile(10)
    weights = top_decile.to_weights(method="equal")

    # DISCREPANCY (recorded above): doc used len(get_weights_at(dates[0])).
    # to_weights() returns 0.0 (not NaN) for unselected assets and the momentum
    # signal is NaN at dates[0], so the doc's expression always prints the full
    # universe size. Count positive weights at a date with valid signals instead
    # (momentum is valid for indices in [lookback, n_dates - skip_recent)).
    holdings = weights.get_weights_at(weights.dates[-22])
    n_selected = sum(1 for w in holdings.values() if w > 0)
    print(f"  Selected {n_selected} assets")
    assert n_selected == 3, f"expected top 10% of 30 = 3 assets, got {n_selected}"
    state["momentum"] = momentum
    state["weights"] = weights


def section_3(state: dict) -> None:
    """3. Sector-Neutral Signal Generation."""
    from vectorforge.portfolio import SymbolMetadata

    validated = state["validated"]
    momentum = state["momentum"]

    # Doc shows AAPL/GOOG; here we tag the whole synthetic universe.
    for sym in validated.symbols:
        validated = validated.set_metadata(
            sym,
            SymbolMetadata(symbol=sym, sector=sector_for(sym), market_cap=1e9 + 1.0),
        )

    sector_neutral_signals = momentum.generate(validated, group_field="sector")
    sector_neutral_weights = sector_neutral_signals.top_percentile(20).to_weights(method="equal")

    holdings = sector_neutral_weights.get_weights_at(sector_neutral_weights.dates[-22])
    n_selected = sum(1 for w in holdings.values() if w > 0)
    print(f"  Sector-neutral selection: {n_selected} assets")
    assert n_selected > 0
    state["validated"] = validated  # keep metadata-enriched version


def section_4(state: dict) -> None:
    """4. Configuring Portfolio Rebalancing."""
    from vectorforge.portfolio import RebalanceFrequency, Rebalancer

    rebalancer = Rebalancer.calendar(
        frequency=RebalanceFrequency.MONTHLY,
        turnover_limit=0.20,
    )
    drift_rebalancer = Rebalancer.drift(threshold=0.05, turnover_limit=0.15)
    hybrid_rebalancer = Rebalancer.hybrid(
        calendar_frequency=RebalanceFrequency.MONTHLY,
        drift_threshold=0.10,
        turnover_limit=0.25,
    )
    print(
        f"  calendar={rebalancer.turnover_limit}, drift={drift_rebalancer.turnover_limit}, "
        f"hybrid={hybrid_rebalancer.turnover_limit}"
    )
    state["rebalancer"] = rebalancer


def section_5(state: dict) -> None:
    """5. Running a Portfolio Backtest."""
    # DISCREPANCY (recorded above): doc says `from vectorforge import
    # VectorizedEngine`; the real class is VectorizedBacktester.
    from vectorforge import VectorizedBacktester

    engine = VectorizedBacktester()
    result = engine.run_portfolio(
        strategy=state["weights"],
        data=state["validated"],
        rebalancer=state["rebalancer"],
        initial_capital=1_000_000,
    )

    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    assert np.isfinite(result.total_return)
    assert np.isfinite(result.sharpe_ratio)
    assert result.max_drawdown <= 0
    state["result"] = result


def section_6(state: dict) -> None:
    """6. Analyzing Portfolio Metrics."""
    from vectorforge.portfolio import PortfolioMetrics

    result = state["result"]
    sector_map = {sym: sector_for(sym) for sym in state["validated"].symbols}

    metrics = PortfolioMetrics.from_backtest_result(
        result,
        sector_map=sector_map,
        risk_free_rate=0.04,
    )

    concentration = metrics.concentration_metrics()
    print(f"  HHI: {concentration.herfindahl_index:.4f}")
    print(f"  Effective N: {concentration.effective_n:.1f}")
    print(f"  Top 5 Weight: {concentration.top_5_weight:.2%}")

    diversification = metrics.diversification_metrics()
    print(f"  Diversification Ratio: {diversification.diversification_ratio:.2f}")
    print(f"  Avg Correlation: {diversification.avg_correlation:.2f}")

    sector_exp = metrics.sector_exposure()
    assert isinstance(sector_exp, pd.DataFrame) and not sector_exp.empty
    print(f"  Sector exposure tail:\n{sector_exp.tail(2).round(3)}")

    attribution = metrics.return_attribution()
    for attr in attribution[:5]:
        print(f"  {attr.symbol}: {attr.contribution:.2%} contribution")
    assert len(attribution) == N_SYMBOLS


def section_7(state: dict) -> None:
    """7. Handling Corporate Actions."""
    # DISCREPANCY (recorded above): doc instantiated the abstract CorporateAction
    # base class with kwargs that don't exist. Real API: Split / Dividend
    # subclasses with (symbol, ex_date, ratio|amount).
    from vectorforge.portfolio import Dividend, Split

    validated = state["validated"]

    actions = [
        Split(symbol="SYM00", ex_date=date(2022, 6, 6), ratio=4.0),  # 4-for-1 split
        Dividend(symbol="SYM00", ex_date=date(2023, 2, 10), amount=0.23),
    ]

    adjusted = validated.apply_corporate_actions(actions)

    # Verify the split actually divided pre-ex-date historical prices by 4.
    idx = validated.symbols.index("SYM00")
    ex_idx = np.where(validated.dates >= pd.Timestamp(date(2022, 6, 6)))[0][0]
    orig_close = validated.close()[idx]
    adj_close = adjusted.close()[idx]
    ratio_before = orig_close[ex_idx - 1] / adj_close[ex_idx - 1]
    assert 3.5 < ratio_before < 4.5, f"split adjustment off: {ratio_before}"
    # Post-ex prices changed only by the (later) dividend back-adjustment.
    print(f"  Pre-split close adjusted by {ratio_before:.2f}x; "
          f"shape preserved: {adjusted.shape == validated.shape}")


def section_8(state: dict) -> None:
    """8. Complete Example: Momentum Strategy."""
    # DISCREPANCY (recorded above): doc uses VectorizedEngine; real class is
    # VectorizedBacktester.
    from vectorforge import VectorizedBacktester
    from vectorforge.portfolio import (
        CrossSectionalSignal,
        MissingDataPolicy,
        PortfolioData,
        PortfolioMetrics,
        RebalanceFrequency,
        Rebalancer,
    )

    # 1. Load and prepare data. The doc loads "sp500_universe.parquet"; we
    # round-trip our synthetic universe through to_parquet/from_parquet so the
    # documented from_parquet path is genuinely exercised.
    with tempfile.TemporaryDirectory() as tmp:
        pq_path = os.path.join(tmp, "universe.parquet")
        state["validated"].to_parquet(pq_path)
        data = PortfolioData.from_parquet(pq_path)

    aligned = data.align(policy=MissingDataPolicy.FORWARD_FILL).validate()

    # 2. Generate momentum signals
    momentum = CrossSectionalSignal.momentum(lookback=252, skip_recent=21)
    signals = momentum.generate(aligned)

    # 3. Top 10% momentum, equal weighted
    weights = signals.top_percentile(10).to_weights(method="equal")

    # 4. Monthly rebalancing, 25% turnover limit
    rebalancer = Rebalancer.calendar(
        frequency=RebalanceFrequency.MONTHLY,
        turnover_limit=0.25,
    )

    # 5. Run backtest
    engine = VectorizedBacktester()
    result = engine.run_portfolio(
        strategy=weights,
        data=aligned,
        rebalancer=rebalancer,
        initial_capital=1_000_000,
    )

    # 6. Analyze results
    metrics = PortfolioMetrics.from_backtest_result(result)
    report = metrics.generate_report()

    print("  === Portfolio Backtest Results ===")
    for key, value in report.items():
        print(f"  {key}: {value}")

    perf = report["performance"]
    print(
        f"  HEADLINE -> total_return={perf['total_return']:.4%} "
        f"sharpe={perf['sharpe_ratio']:.3f} max_drawdown={perf['max_drawdown']:.4%}"
    )
    assert np.isfinite(perf["total_return"])
    assert np.isfinite(perf["sharpe_ratio"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    sections = [
        (1, "Loading multi-asset data (from_dict/align/validate)", section_1),
        (2, "Cross-sectional signals (momentum/top_percentile/to_weights)", section_2),
        (3, "Sector-neutral signals (SymbolMetadata/group_field)", section_3),
        (4, "Rebalancer.calendar/.drift/.hybrid", section_4),
        (5, "engine.run_portfolio", section_5),
        (6, "PortfolioMetrics.from_backtest_result + analytics", section_6),
        (7, "Corporate actions (Split/Dividend/apply_corporate_actions)", section_7),
        (8, "Complete momentum strategy example", section_8),
    ]

    state: dict = {}
    results: list[tuple[int, str, bool]] = []

    for num, name, fn in sections:
        print(f"\n--- SECTION {num}: {name} ---")
        try:
            fn(state)
            print(f"SECTION {num}: OK")
            results.append((num, name, True))
        except Exception:
            traceback.print_exc()
            print(f"SECTION {num}: FAIL")
            results.append((num, name, False))

    print("\n================ SUMMARY ================")
    for num, name, ok in results:
        print(f"  Section {num}: {'OK  ' if ok else 'FAIL'} - {name}")

    print("\n========== DOC DISCREPANCIES FOUND ==========")
    for i, d in enumerate(DISCREPANCIES, 1):
        print(f"  {i}. {d}")

    return 0 if all(ok for _, _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
