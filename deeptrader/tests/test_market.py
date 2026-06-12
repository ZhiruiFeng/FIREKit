"""Regime-switching market generator."""

import numpy as np
import pytest

from deeptrader.market import MEAN_REVERSION, MOMENTUM, regime_switching_series


class TestRegimeSeries:
    def test_deterministic_for_fixed_seed(self):
        a = regime_switching_series(n=500, seed=7)
        b = regime_switching_series(n=500, seed=7)
        np.testing.assert_array_equal(a.prices, b.prices)
        np.testing.assert_array_equal(a.regimes, b.regimes)

    def test_seed_changes_path(self):
        a = regime_switching_series(n=500, seed=7)
        b = regime_switching_series(n=500, seed=8)
        assert not np.array_equal(a.prices, b.prices)

    def test_shapes(self):
        s = regime_switching_series(n=300, seed=1)
        assert len(s.prices) == 301
        assert len(s.returns) == 300
        assert len(s.regimes) == 300

    def test_prices_consistent_with_returns(self):
        s = regime_switching_series(n=200, seed=3)
        rebuilt = s.prices[:-1] * (1.0 + s.returns)
        np.testing.assert_allclose(rebuilt, s.prices[1:], rtol=1e-10)

    def test_both_regimes_present(self):
        s = regime_switching_series(n=3000, seed=42)
        assert set(np.unique(s.regimes)) == {MOMENTUM, MEAN_REVERSION}

    def test_regime_autocorrelation_signs(self):
        # Lag-1 autocorrelation should be positive in momentum stretches and
        # negative in mean-reversion stretches.
        s = regime_switching_series(n=6000, seed=11)
        r0, r1 = s.returns[1:], s.returns[:-1]
        same_regime = s.regimes[1:] == s.regimes[:-1]
        mom = same_regime & (s.regimes[1:] == MOMENTUM)
        rev = same_regime & (s.regimes[1:] == MEAN_REVERSION)
        corr_mom = np.corrcoef(r0[mom], r1[mom])[0, 1]
        corr_rev = np.corrcoef(r0[rev], r1[rev])[0, 1]
        assert corr_mom > 0.3
        assert corr_rev < -0.3

    def test_prices_positive(self):
        s = regime_switching_series(n=3000, seed=42)
        assert (s.prices > 0).all()

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            regime_switching_series(n=1)
