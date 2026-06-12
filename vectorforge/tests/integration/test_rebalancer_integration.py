"""
Integration test: monthly rebalance with 20% turnover limit (T052, US3).

Verifies that trades occur only at month boundaries and that the
turnover constraint is respected within 0.1% tolerance (SC-005),
per FR-012, FR-014, FR-015.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorforge import VectorizedBacktester
from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.rebalancer import RebalanceFrequency, Rebalancer
from vectorforge.portfolio.signals import TargetWeights

N_SYMBOLS = 10
N_DAYS = 252  # ~12 months
TURNOVER_LIMIT = 0.20
SC005_TOLERANCE = 0.001  # 0.1%

SYMBOLS = [f"S{i}" for i in range(N_SYMBOLS)]
DATES = pd.date_range("2023-01-02", periods=N_DAYS, freq="B")


def make_ohlcv_df(close: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1e6),
        },
        index=dates,
    )


def first_trading_days(dates: pd.DatetimeIndex) -> set:
    """First trading day of each (year, month) in the index."""
    return {
        ts.date() for ts in dates.to_series().groupby([dates.year, dates.month]).min()
    }


@pytest.fixture(scope="module")
def portfolio_data() -> PortfolioData:
    rng = np.random.default_rng(42)
    data = {
        sym: make_ohlcv_df(
            100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, N_DAYS))), DATES
        )
        for sym in SYMBOLS
    }
    return PortfolioData.from_dict(data)


@pytest.fixture(scope="module")
def flipping_targets() -> TargetWeights:
    """Targets that alternate concentration each month, forcing 100%
    desired turnover at every monthly rebalance (>> 20% limit)."""
    weights = np.zeros((N_SYMBOLS, N_DAYS))
    for j, d in enumerate(DATES):
        if d.month % 2 == 0:
            weights[:5, j] = 0.2
        else:
            weights[5:, j] = 0.2
    return TargetWeights.from_array(weights, SYMBOLS, DATES)


@pytest.fixture(scope="module")
def backtest_result(portfolio_data, flipping_targets):
    engine = VectorizedBacktester()
    rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY, turnover_limit=TURNOVER_LIMIT)
    return engine.run_portfolio(
        strategy=flipping_targets,
        data=portfolio_data,
        initial_capital=1_000_000,
        rebalancer=rebalancer,
    )


def executed_turnover(result, rebalance_date) -> float:
    """
    Reconstruct the turnover actually executed at a rebalance date.

    weights_history[t] holds post-rebalance weights after one day of drift:
        w[t] = normalize(executed_weights * (1 + r_t))
    so executed_weights = normalize(w[t] / (1 + r_t)), and the executed
    turnover is |executed_weights - w[t-1]| / 2.
    """
    weights = result.weights_history
    ts = pd.Timestamp(rebalance_date)
    t = weights.index.get_loc(ts)
    r = result.asset_returns.loc[ts].values
    undrifted = weights.iloc[t].values / (1 + r)
    executed = undrifted / undrifted.sum()
    pre_trade = weights.iloc[t - 1].values
    return float(np.abs(executed - pre_trade).sum() / 2)


class TestMonthlyRebalanceTiming:
    """US3 scenario 1: position adjustments only at month boundaries."""

    def test_rebalances_occur_once_per_month(self, backtest_result):
        months = [(d.year, d.month) for d in backtest_result.rebalance_dates]
        assert len(months) == len(set(months)), "more than one rebalance in a month"
        # Roughly one rebalance per month of data
        n_months = len({(d.year, d.month) for d in DATES})
        assert len(months) == n_months

    def test_rebalances_fall_on_first_trading_day_of_month(self, backtest_result):
        firsts = first_trading_days(DATES)
        # The engine's initial allocation fires on the second trading day
        # (simulation starts at t=1); all later rebalances must land on the
        # first trading day of a month.
        first, *rest = backtest_result.rebalance_dates
        assert first in firsts or first == DATES[1].date()
        for d in rest:
            assert d in firsts, f"rebalance on {d} is not a month boundary"

    def test_no_trades_between_rebalance_dates(self, backtest_result):
        """Between rebalances, weight changes come only from price drift."""
        weights = backtest_result.weights_history
        asset_returns = backtest_result.asset_returns
        rebalance_ts = {pd.Timestamp(d) for d in backtest_result.rebalance_dates}

        for t in range(1, len(weights)):
            ts = weights.index[t]
            if ts in rebalance_ts:
                continue
            prev = weights.iloc[t - 1].values
            drifted = prev * (1 + asset_returns.loc[ts].values)
            if drifted.sum() > 0:
                drifted = drifted / drifted.sum()
            assert np.allclose(
                weights.iloc[t].values, drifted, atol=1e-9
            ), f"untriggered trade detected on {ts.date()}"


class TestTurnoverConstraint:
    """US3 scenario 3 / SC-005: executed turnover within 0.1% of the limit."""

    def test_desired_turnover_exceeds_limit(self, flipping_targets, backtest_result):
        """Sanity: the scenario genuinely demands more than 20% turnover."""
        # Month-flip rebalances request a full portfolio swap (100% turnover)
        flip_count = 0
        weights = backtest_result.weights_history
        for d in backtest_result.rebalance_dates[1:]:
            t = weights.index.get_loc(pd.Timestamp(d))
            target = flipping_targets.weights[:, t]
            pre = weights.iloc[t - 1].values
            if np.abs(target - pre).sum() / 2 > TURNOVER_LIMIT:
                flip_count += 1
        assert flip_count >= 3

    def test_executed_turnover_never_exceeds_limit(self, backtest_result):
        for d in backtest_result.rebalance_dates:
            turnover = executed_turnover(backtest_result, d)
            assert turnover <= TURNOVER_LIMIT + SC005_TOLERANCE, (
                f"executed turnover {turnover:.4f} on {d} exceeds "
                f"{TURNOVER_LIMIT:.0%} limit (+0.1% tolerance)"
            )

    def test_constrained_rebalances_use_full_budget(self, backtest_result, flipping_targets):
        """When desired turnover exceeds the limit, executed turnover should
        equal the limit within 0.1% (partial rebalancing, FR-014/FR-015)."""
        weights = backtest_result.weights_history
        checked = 0
        for d in backtest_result.rebalance_dates[1:]:
            t = weights.index.get_loc(pd.Timestamp(d))
            target = flipping_targets.weights[:, t]
            pre = weights.iloc[t - 1].values
            desired = np.abs(target - pre).sum() / 2
            if desired > TURNOVER_LIMIT + SC005_TOLERANCE:
                executed = executed_turnover(backtest_result, d)
                assert executed == pytest.approx(TURNOVER_LIMIT, abs=SC005_TOLERANCE)
                checked += 1
        assert checked >= 3

    def test_turnover_history_reports_executed_turnover(self, backtest_result):
        assert (
            backtest_result.turnover_history <= TURNOVER_LIMIT + SC005_TOLERANCE
        ).all()


class TestStandaloneRebalancerSimulation:
    """Rebalancer.run() honors the same monthly schedule and constraint."""

    @pytest.fixture(scope="class")
    def run_result(self, portfolio_data, flipping_targets):
        rebalancer = Rebalancer.calendar(
            RebalanceFrequency.MONTHLY, turnover_limit=TURNOVER_LIMIT
        )
        return rebalancer.run(flipping_targets, portfolio_data)

    def test_orders_only_at_month_starts(self, run_result):
        firsts = first_trading_days(DATES)
        for d in run_result.dates:
            assert d in firsts, f"order on {d} is not a first trading day of a month"

    def test_all_order_turnovers_within_limit(self, run_result):
        assert len(run_result.orders) > 3
        for order in run_result.orders:
            assert order.turnover <= TURNOVER_LIMIT + SC005_TOLERANCE

    def test_constrained_orders_report_exactly_the_limit(self, run_result):
        constrained = [o for o in run_result.orders if o.constrained]
        assert len(constrained) >= 3
        for order in constrained:
            assert order.turnover == pytest.approx(TURNOVER_LIMIT, abs=SC005_TOLERANCE)
            # Executed trades match the reported turnover (float32 prices)
            assert sum(abs(v) for v in order.trades.values()) / 2 == pytest.approx(
                order.turnover, abs=1e-6
            )
