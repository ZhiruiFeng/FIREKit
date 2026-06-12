"""
T086 — Cross-sectional signal generation benchmarks.

Success criterion (specs/001-multi-asset-portfolio/spec.md):

- SC-003: Cross-sectional signal generation for 500 assets completes in
  under 500 milliseconds *per period*.

Interpretation (documented, since the spec is per-period but the API is
batch): we call ``CrossSectionalSignal.momentum(...).generate(data)`` —
which computes raw momentum AND cross-sectional percentile ranks for every
date — over a 2-year (504-day) window for 500 assets, then divide total
wall-clock time by the number of periods that produced a valid signal
(n_dates - lookback - skip_recent = 231). This is conservative: the
amortized cost per period includes ranking work done for all dates.

Hardware caveat: shared cloud container, pure NumPy/SciPy backend — JAX and
Numba are not installed.
"""

from __future__ import annotations

import time

import numpy as np

from vectorforge.portfolio.signals import CrossSectionalSignal

from tests.benchmark.helpers import make_portfolio_data

N_ASSETS = 500
N_DAYS = 504  # 2 years of business days
LOOKBACK = 252
SKIP_RECENT = 21

SC003_TARGET_SECONDS_PER_PERIOD = 0.500


class TestSC003SignalSpeed:
    """SC-003: 500-asset cross-sectional momentum signal < 500 ms per period."""

    def test_sc003_momentum_500_assets_under_500ms_per_period(self):
        data = make_portfolio_data(N_ASSETS, N_DAYS, seed=7)
        signal = CrossSectionalSignal.momentum(lookback=LOOKBACK, skip_recent=SKIP_RECENT)

        # Warm-up on a tiny universe (amortize imports / first-call overhead).
        signal.generate(make_portfolio_data(5, 300, seed=1))

        start = time.perf_counter()
        result = signal.generate(data)
        elapsed = time.perf_counter() - start

        # Sanity: signal really covers the expected periods.
        valid_periods = int((~np.isnan(result.values)).any(axis=0).sum())
        assert valid_periods == N_DAYS - LOOKBACK - SKIP_RECENT
        assert result.values.shape == (N_ASSETS, N_DAYS)

        # Ranks are 0-100 percentiles on valid periods.
        valid_ranks = result.ranks[~np.isnan(result.values)]
        assert valid_ranks.min() >= 0.0
        assert valid_ranks.max() <= 100.0

        per_period = elapsed / valid_periods
        print(
            f"\nSC-003: momentum generate() for {N_ASSETS} assets x {N_DAYS} days: "
            f"{elapsed:.3f}s total, {valid_periods} valid periods -> "
            f"{per_period * 1000:.2f} ms/period (target < 500 ms/period)"
        )
        assert per_period < SC003_TARGET_SECONDS_PER_PERIOD, (
            f"SC-003 FAIL: {per_period * 1000:.1f} ms per period, target < 500 ms"
        )

    def test_sc003_top_decile_filter_and_weights_under_500ms_per_period(self):
        """Downstream selection (top 10% -> weights) stays within budget too."""
        data = make_portfolio_data(N_ASSETS, N_DAYS, seed=7)
        signal = CrossSectionalSignal.momentum(lookback=LOOKBACK, skip_recent=SKIP_RECENT)
        result = signal.generate(data)

        start = time.perf_counter()
        weights = result.top_percentile(10).to_weights(method="equal")
        elapsed = time.perf_counter() - start

        n_periods = N_DAYS - LOOKBACK - SKIP_RECENT
        per_period = elapsed / n_periods

        # Sanity: weight columns on valid dates sum to ~1.
        col_sums = weights.weights.sum(axis=0)
        active_cols = col_sums > 0
        assert active_cols.sum() >= n_periods
        np.testing.assert_allclose(col_sums[active_cols], 1.0, atol=1e-9)

        print(
            f"\nSC-003 (selection): top-10% filter + to_weights: {elapsed:.3f}s total, "
            f"{per_period * 1000:.2f} ms/period (target < 500 ms/period)"
        )
        assert per_period < SC003_TARGET_SECONDS_PER_PERIOD
