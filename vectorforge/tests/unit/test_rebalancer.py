"""
Unit tests for portfolio rebalancing (T051, US3).

Covers RebalanceFrequency, Calendar/Drift/Hybrid triggers,
Rebalancer.compute_trades() with turnover constraints, and
Rebalancer.run() full simulation (FR-012, FR-013, FR-014, FR-015, FR-016).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.rebalancer import (
    CalendarTrigger,
    DriftTrigger,
    HybridTrigger,
    RebalanceFrequency,
    RebalanceOrders,
    Rebalancer,
    RebalanceResult,
)
from vectorforge.portfolio.signals import TargetWeights


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


EQUAL = {"A": 0.5, "B": 0.5}


class TestRebalanceFrequency:
    def test_enum_members(self):
        assert RebalanceFrequency.DAILY.value == "daily"
        assert RebalanceFrequency.WEEKLY.value == "weekly"
        assert RebalanceFrequency.MONTHLY.value == "monthly"
        assert RebalanceFrequency.QUARTERLY.value == "quarterly"
        assert RebalanceFrequency.ANNUAL.value == "annual"


class TestCalendarTrigger:
    """Calendar-based rebalancing (FR-012)."""

    def test_first_rebalance_always_fires(self):
        trigger = CalendarTrigger(RebalanceFrequency.MONTHLY)
        assert trigger.should_rebalance(date(2023, 1, 17), EQUAL, EQUAL, None) is True

    def test_monthly_does_not_fire_within_month(self):
        trigger = CalendarTrigger(RebalanceFrequency.MONTHLY)
        assert (
            trigger.should_rebalance(date(2023, 1, 25), EQUAL, EQUAL, date(2023, 1, 2)) is False
        )

    def test_monthly_fires_on_new_month(self):
        trigger = CalendarTrigger(RebalanceFrequency.MONTHLY)
        assert trigger.should_rebalance(date(2023, 2, 1), EQUAL, EQUAL, date(2023, 1, 2)) is True

    def test_daily_always_fires(self):
        trigger = CalendarTrigger(RebalanceFrequency.DAILY)
        assert trigger.should_rebalance(date(2023, 1, 2), EQUAL, EQUAL, date(2023, 1, 2)) is True

    def test_weekly_fires_only_on_new_iso_week(self):
        trigger = CalendarTrigger(RebalanceFrequency.WEEKLY)
        monday = date(2023, 1, 2)
        friday = date(2023, 1, 6)
        next_monday = date(2023, 1, 9)
        assert trigger.should_rebalance(friday, EQUAL, EQUAL, monday) is False
        assert trigger.should_rebalance(next_monday, EQUAL, EQUAL, monday) is True

    def test_quarterly_fires_only_on_new_quarter(self):
        trigger = CalendarTrigger(RebalanceFrequency.QUARTERLY)
        assert trigger.should_rebalance(date(2023, 3, 15), EQUAL, EQUAL, date(2023, 1, 2)) is False
        assert trigger.should_rebalance(date(2023, 4, 3), EQUAL, EQUAL, date(2023, 1, 2)) is True

    def test_annual_fires_only_on_new_year(self):
        trigger = CalendarTrigger(RebalanceFrequency.ANNUAL)
        assert trigger.should_rebalance(date(2023, 11, 1), EQUAL, EQUAL, date(2023, 1, 2)) is False
        assert trigger.should_rebalance(date(2024, 1, 2), EQUAL, EQUAL, date(2023, 1, 2)) is True


class TestDriftTrigger:
    """Drift-threshold rebalancing (FR-013, US3 scenario 2)."""

    def test_fires_when_drift_at_threshold(self):
        trigger = DriftTrigger(threshold=0.05)
        current = {"A": 0.55, "B": 0.45}
        target = {"A": 0.50, "B": 0.50}
        assert trigger.should_rebalance(date(2023, 1, 5), current, target, None) is True

    def test_does_not_fire_below_threshold(self):
        trigger = DriftTrigger(threshold=0.05)
        current = {"A": 0.53, "B": 0.47}
        target = {"A": 0.50, "B": 0.50}
        assert trigger.should_rebalance(date(2023, 1, 5), current, target, None) is False

    def test_get_drifts_absolute(self):
        trigger = DriftTrigger(threshold=0.05)
        drifts = trigger.get_drifts({"A": 0.55, "B": 0.45}, {"A": 0.50, "B": 0.50})
        assert drifts["A"] == pytest.approx(0.05)
        assert drifts["B"] == pytest.approx(-0.05)

    def test_get_drifts_relative(self):
        trigger = DriftTrigger(threshold=0.40, measure="relative")
        drifts = trigger.get_drifts({"A": 0.075}, {"A": 0.05})
        assert drifts["A"] == pytest.approx(0.5)
        assert trigger.should_rebalance(date(2023, 1, 5), {"A": 0.075}, {"A": 0.05}, None) is True

    def test_symbol_missing_from_current_counts_as_full_drift(self):
        trigger = DriftTrigger(threshold=0.05)
        drifts = trigger.get_drifts({"A": 1.0}, {"A": 0.9, "B": 0.1})
        assert drifts["B"] == pytest.approx(-0.1)


class TestHybridTrigger:
    """Combined calendar + drift triggers (FR-016, US3)."""

    def test_fires_on_calendar_even_without_drift(self):
        trigger = HybridTrigger(
            [CalendarTrigger(RebalanceFrequency.MONTHLY), DriftTrigger(threshold=0.10)]
        )
        assert trigger.should_rebalance(date(2023, 2, 1), EQUAL, EQUAL, date(2023, 1, 2)) is True

    def test_fires_on_drift_even_within_month(self):
        trigger = HybridTrigger(
            [CalendarTrigger(RebalanceFrequency.MONTHLY), DriftTrigger(threshold=0.10)]
        )
        current = {"A": 0.65, "B": 0.35}
        target = {"A": 0.50, "B": 0.50}
        assert (
            trigger.should_rebalance(date(2023, 1, 20), current, target, date(2023, 1, 2)) is True
        )

    def test_does_not_fire_when_no_trigger_fires(self):
        trigger = HybridTrigger(
            [CalendarTrigger(RebalanceFrequency.MONTHLY), DriftTrigger(threshold=0.10)]
        )
        current = {"A": 0.52, "B": 0.48}
        target = {"A": 0.50, "B": 0.50}
        assert (
            trigger.should_rebalance(date(2023, 1, 20), current, target, date(2023, 1, 2))
            is False
        )


class TestComputeTrades:
    """Trade computation with turnover constraints (FR-014, FR-015)."""

    def test_unconstrained_trades_reach_target(self):
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.5, "B": 0.5},
            target_weights={"A": 0.8, "B": 0.2},
            current_date=date(2023, 2, 1),
        )

        assert isinstance(orders, RebalanceOrders)
        assert orders.trades["A"] == pytest.approx(0.3)
        assert orders.trades["B"] == pytest.approx(-0.3)
        assert orders.turnover == pytest.approx(0.3)
        assert orders.constrained is False
        assert orders.to_weights["A"] == pytest.approx(0.8)
        assert orders.to_weights["B"] == pytest.approx(0.2)

    def test_turnover_definition_half_sum_abs(self):
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
        orders = rebalancer.compute_trades(
            current_weights={"A": 1.0, "B": 0.0, "C": 0.0},
            target_weights={"A": 0.0, "B": 0.5, "C": 0.5},
            current_date=date(2023, 2, 1),
        )
        # sum |delta| = 2.0 -> turnover = 1.0
        assert orders.turnover == pytest.approx(1.0)

    def test_turnover_constraint_caps_trades(self):
        """US3 scenario 3 / SC-005: constrained turnover within 0.1% of limit."""
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY, turnover_limit=0.10)
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.5, "B": 0.5},
            target_weights={"A": 0.8, "B": 0.2},
            current_date=date(2023, 2, 1),
        )

        assert orders.constrained is True
        assert orders.turnover == pytest.approx(0.10, abs=0.001)
        # Trades scaled by 1/3
        assert orders.trades["A"] == pytest.approx(0.1)
        assert orders.trades["B"] == pytest.approx(-0.1)
        assert orders.to_weights["A"] == pytest.approx(0.6)
        assert orders.to_weights["B"] == pytest.approx(0.4)

    def test_constraint_not_applied_when_under_limit(self):
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY, turnover_limit=0.50)
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.5, "B": 0.5},
            target_weights={"A": 0.6, "B": 0.4},
            current_date=date(2023, 2, 1),
        )
        assert orders.constrained is False
        assert orders.turnover == pytest.approx(0.1)

    def test_constrained_trades_preserve_deviation_priority(self):
        """FR-015: the largest deviations receive the largest trades."""
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY, turnover_limit=0.10)
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.6, "B": 0.3, "C": 0.1},
            target_weights={"A": 0.1, "B": 0.4, "C": 0.5},
            current_date=date(2023, 2, 1),
        )

        assert orders.constrained is True
        # Deviations: A -0.5, C +0.4, B +0.1 -> trade magnitudes keep that order
        assert abs(orders.trades["A"]) > abs(orders.trades["C"]) > abs(orders.trades["B"])
        assert sum(abs(v) for v in orders.trades.values()) / 2 == pytest.approx(0.10, abs=0.001)

    def test_orders_properties(self):
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.5, "B": 0.5, "C": 0.0},
            target_weights={"A": 0.3, "B": 0.5, "C": 0.2},
            current_date=date(2023, 2, 1),
        )

        assert orders.n_trades == 2
        assert orders.buy_symbols == ["C"]
        assert orders.sell_symbols == ["A"]
        assert orders.date == date(2023, 2, 1)


class TestRebalanceResult:
    def test_aggregates_orders(self):
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
        o1 = rebalancer.compute_trades({"A": 1.0}, {"A": 0.8, "B": 0.2}, date(2023, 1, 2))
        o2 = rebalancer.compute_trades({"A": 0.8, "B": 0.2}, {"A": 0.6, "B": 0.4}, date(2023, 2, 1))

        result = RebalanceResult(orders=[o1, o2])
        assert result.dates == [date(2023, 1, 2), date(2023, 2, 1)]
        assert result.total_turnover == pytest.approx(o1.turnover + o2.turnover)
        assert result.avg_turnover == pytest.approx((o1.turnover + o2.turnover) / 2)

        df = result.to_dataframe()
        assert list(df.columns) == ["date", "turnover", "n_trades", "constrained"]
        assert len(df) == 2

    def test_empty_result(self):
        result = RebalanceResult()
        assert result.dates == []
        assert result.total_turnover == 0.0
        assert result.avg_turnover == 0.0
        assert list(result.to_dataframe().columns) == [
            "date",
            "turnover",
            "n_trades",
            "constrained",
        ]


class TestRebalancerRun:
    """Full rebalancing simulation over a price history."""

    @pytest.fixture
    def portfolio_data(self) -> PortfolioData:
        dates = pd.date_range("2023-01-02", periods=65, freq="B")  # ~3 months
        n = len(dates)
        rng = np.random.default_rng(3)
        data = {
            sym: make_ohlcv_df(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), dates)
            for sym in ["A", "B", "C"]
        }
        return PortfolioData.from_dict(data)

    def test_monthly_run_rebalances_once_per_month(self, portfolio_data):
        targets = TargetWeights.equal_weight(portfolio_data.symbols, portfolio_data.dates)
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
        result = rebalancer.run(targets, portfolio_data, initial_weights={"A": 1.0})

        months = {(d.year, d.month) for d in result.dates}
        assert len(result.orders) == len(months)  # one rebalance per month

    def test_monthly_run_trades_at_month_start(self, portfolio_data):
        targets = TargetWeights.equal_weight(portfolio_data.symbols, portfolio_data.dates)
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
        result = rebalancer.run(targets, portfolio_data, initial_weights={"A": 1.0})

        # First trading day of each month present in the data
        dates_idx = portfolio_data.dates
        firsts = {
            ts.date()
            for ts in dates_idx.to_series().groupby([dates_idx.year, dates_idx.month]).min()
        }
        for d in result.dates:
            assert d in firsts, f"rebalance on {d} is not a first trading day of a month"

    def test_run_respects_turnover_limit(self, portfolio_data):
        """SC-005: every executed rebalance within 0.1% of the constraint."""
        targets = TargetWeights.equal_weight(portfolio_data.symbols, portfolio_data.dates)
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY, turnover_limit=0.20)
        result = rebalancer.run(targets, portfolio_data, initial_weights={"A": 1.0})

        for order in result.orders:
            assert order.turnover <= 0.20 + 0.001

        # The first rebalance (from 100% A to equal weight) needs 2/3 turnover,
        # so it must be constrained to exactly the limit.
        assert result.orders[0].constrained is True
        assert result.orders[0].turnover == pytest.approx(0.20, abs=0.001)

    def test_drift_rebalancer_run_fires_on_drift_only(self):
        """Drift trigger: rebalance only when weights drift beyond threshold."""
        dates = pd.date_range("2023-01-02", periods=40, freq="B")
        n = len(dates)
        # A grows fast, B flat -> weights drift towards A
        data = {
            "A": make_ohlcv_df(100 * np.cumprod(np.full(n, 1.02)), dates),
            "B": make_ohlcv_df(np.full(n, 100.0), dates),
        }
        pdata = PortfolioData.from_dict(data)
        targets = TargetWeights.equal_weight(pdata.symbols, pdata.dates)

        rebalancer = Rebalancer.drift(threshold=0.05)
        result = rebalancer.run(targets, pdata, initial_weights={"A": 0.5, "B": 0.5})

        # Drift accumulates ~0.5%/day, so rebalances should occur but not daily
        assert 1 <= len(result.orders) < n


class TestFactories:
    def test_calendar_factory(self):
        rebalancer = Rebalancer.calendar(RebalanceFrequency.QUARTERLY, turnover_limit=0.3)
        assert isinstance(rebalancer.trigger, CalendarTrigger)
        assert rebalancer.trigger.frequency == RebalanceFrequency.QUARTERLY
        assert rebalancer.turnover_limit == 0.3

    def test_drift_factory(self):
        rebalancer = Rebalancer.drift(threshold=0.07)
        assert isinstance(rebalancer.trigger, DriftTrigger)
        assert rebalancer.trigger.threshold == 0.07
        assert rebalancer.turnover_limit is None

    def test_hybrid_factory(self):
        rebalancer = Rebalancer.hybrid(
            calendar_frequency=RebalanceFrequency.MONTHLY,
            drift_threshold=0.10,
            turnover_limit=0.25,
        )
        assert isinstance(rebalancer.trigger, HybridTrigger)
        trigger_types = {type(t) for t in rebalancer.trigger.triggers}
        assert trigger_types == {CalendarTrigger, DriftTrigger}
        assert rebalancer.turnover_limit == 0.25
