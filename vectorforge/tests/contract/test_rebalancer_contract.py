"""
Contract Tests for Portfolio Rebalancer

These tests define the expected API contract for rebalancing logic.
"""

from datetime import date

from vectorforge.portfolio.rebalancer import (
    CalendarTrigger,
    DriftTrigger,
    HybridTrigger,
    RebalanceFrequency,
    RebalanceOrders,
    Rebalancer,
    RebalanceResult,
)


class TestRebalanceTriggers:
    """Contract tests for RebalanceTrigger classes."""

    def test_calendar_trigger_monthly(self):
        """CalendarTrigger with MONTHLY should trigger on month change."""
        trigger = CalendarTrigger(RebalanceFrequency.MONTHLY)

        # First call should trigger
        assert trigger.should_rebalance(date(2023, 1, 3), {"A": 0.5}, {"A": 0.5}, None) is True

        # Same month, no trigger
        assert (
            trigger.should_rebalance(date(2023, 1, 15), {"A": 0.5}, {"A": 0.5}, date(2023, 1, 3))
            is False
        )

        # New month, trigger
        assert (
            trigger.should_rebalance(date(2023, 2, 1), {"A": 0.5}, {"A": 0.5}, date(2023, 1, 3))
            is True
        )

    def test_calendar_trigger_quarterly(self):
        """CalendarTrigger with QUARTERLY should trigger on quarter change."""
        trigger = CalendarTrigger(RebalanceFrequency.QUARTERLY)

        # Q1 -> Q2
        assert (
            trigger.should_rebalance(date(2023, 4, 3), {"A": 0.5}, {"A": 0.5}, date(2023, 1, 3))
            is True
        )

        # Within Q2
        assert (
            trigger.should_rebalance(date(2023, 5, 1), {"A": 0.5}, {"A": 0.5}, date(2023, 4, 3))
            is False
        )

    def test_calendar_trigger_daily(self):
        """CalendarTrigger with DAILY should always trigger."""
        trigger = CalendarTrigger(RebalanceFrequency.DAILY)

        assert (
            trigger.should_rebalance(date(2023, 1, 3), {"A": 0.5}, {"A": 0.5}, date(2023, 1, 2))
            is True
        )

    def test_drift_trigger_absolute(self):
        """DriftTrigger should trigger when drift exceeds threshold."""
        trigger = DriftTrigger(threshold=0.05)  # 5%

        # No drift
        assert (
            trigger.should_rebalance(
                date(2023, 1, 3),
                {"A": 0.50, "B": 0.50},
                {"A": 0.50, "B": 0.50},
                date(2023, 1, 2),
            )
            is False
        )

        # 4% drift - below threshold
        assert (
            trigger.should_rebalance(
                date(2023, 1, 3),
                {"A": 0.54, "B": 0.46},
                {"A": 0.50, "B": 0.50},
                date(2023, 1, 2),
            )
            is False
        )

        # 6% drift - above threshold
        assert (
            trigger.should_rebalance(
                date(2023, 1, 3),
                {"A": 0.56, "B": 0.44},
                {"A": 0.50, "B": 0.50},
                date(2023, 1, 2),
            )
            is True
        )

    def test_drift_trigger_get_drifts(self):
        """DriftTrigger.get_drifts should return per-symbol drift."""
        trigger = DriftTrigger(threshold=0.05)

        drifts = trigger.get_drifts(
            {"A": 0.55, "B": 0.45},
            {"A": 0.50, "B": 0.50},
        )

        assert abs(drifts["A"] - 0.05) < 0.001
        assert abs(drifts["B"] - (-0.05)) < 0.001

    def test_hybrid_trigger_or_logic(self):
        """HybridTrigger should trigger if ANY trigger fires."""
        trigger = HybridTrigger(
            [
                CalendarTrigger(RebalanceFrequency.MONTHLY),
                DriftTrigger(threshold=0.10),
            ]
        )

        # Month change (calendar fires)
        assert (
            trigger.should_rebalance(
                date(2023, 2, 1),
                {"A": 0.50},
                {"A": 0.50},
                date(2023, 1, 15),
            )
            is True
        )

        # Large drift (drift fires)
        assert (
            trigger.should_rebalance(
                date(2023, 1, 20),
                {"A": 0.65},
                {"A": 0.50},
                date(2023, 1, 15),
            )
            is True
        )

        # Neither fires
        assert (
            trigger.should_rebalance(
                date(2023, 1, 20),
                {"A": 0.52},
                {"A": 0.50},
                date(2023, 1, 15),
            )
            is False
        )


class TestRebalancerComputeTrades:
    """Contract tests for Rebalancer.compute_trades()."""

    def test_compute_trades_basic(self):
        """compute_trades should return RebalanceOrders with correct trades."""
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)

        orders = rebalancer.compute_trades(
            current_weights={"A": 0.60, "B": 0.40},
            target_weights={"A": 0.50, "B": 0.50},
            current_date=date(2023, 1, 3),
        )

        assert isinstance(orders, RebalanceOrders)
        assert orders.date == date(2023, 1, 3)
        assert abs(orders.trades["A"] - (-0.10)) < 0.001
        assert abs(orders.trades["B"] - 0.10) < 0.001

    def test_compute_trades_turnover(self):
        """compute_trades should calculate correct turnover."""
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)

        orders = rebalancer.compute_trades(
            current_weights={"A": 0.60, "B": 0.40},
            target_weights={"A": 0.50, "B": 0.50},
            current_date=date(2023, 1, 3),
        )

        # Turnover = sum(|trade|) / 2 = (0.10 + 0.10) / 2 = 0.10
        assert abs(orders.turnover - 0.10) < 0.001

    def test_compute_trades_from_to_weights(self):
        """compute_trades should track from and to weights."""
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)

        orders = rebalancer.compute_trades(
            current_weights={"A": 0.60, "B": 0.40},
            target_weights={"A": 0.50, "B": 0.50},
            current_date=date(2023, 1, 3),
        )

        assert orders.from_weights == {"A": 0.60, "B": 0.40}
        assert abs(orders.to_weights["A"] - 0.50) < 0.001
        assert abs(orders.to_weights["B"] - 0.50) < 0.001


class TestTurnoverConstraint:
    """Contract tests for turnover constraint enforcement."""

    def test_turnover_limit_scales_trades(self):
        """Turnover limit should scale down trades if exceeded."""
        rebalancer = Rebalancer.calendar(
            RebalanceFrequency.MONTHLY,
            turnover_limit=0.05,  # 5% limit
        )

        # Request 20% turnover
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.70, "B": 0.30},
            target_weights={"A": 0.50, "B": 0.50},
            current_date=date(2023, 1, 3),
        )

        assert orders.constrained is True
        assert orders.turnover <= 0.05 + 0.001

    def test_no_constraint_when_under_limit(self):
        """No constraint should be applied when under limit."""
        rebalancer = Rebalancer.calendar(
            RebalanceFrequency.MONTHLY,
            turnover_limit=0.20,  # 20% limit
        )

        # Request 10% turnover
        orders = rebalancer.compute_trades(
            current_weights={"A": 0.60, "B": 0.40},
            target_weights={"A": 0.50, "B": 0.50},
            current_date=date(2023, 1, 3),
        )

        assert orders.constrained is False
        assert abs(orders.turnover - 0.10) < 0.001

    def test_unlimited_turnover_when_none(self):
        """No limit when turnover_limit is None."""
        rebalancer = Rebalancer.calendar(
            RebalanceFrequency.MONTHLY,
            turnover_limit=None,
        )

        # Large rebalance
        orders = rebalancer.compute_trades(
            current_weights={"A": 1.00, "B": 0.00},
            target_weights={"A": 0.00, "B": 1.00},
            current_date=date(2023, 1, 3),
        )

        assert orders.constrained is False
        assert abs(orders.turnover - 1.0) < 0.001


class TestRebalanceOrders:
    """Contract tests for RebalanceOrders properties."""

    def test_n_trades_counts_active_trades(self):
        """n_trades should count trades with non-zero changes."""
        orders = RebalanceOrders(
            date=date(2023, 1, 3),
            trades={"A": 0.10, "B": -0.05, "C": 0.0},
            from_weights={"A": 0.40, "B": 0.35, "C": 0.25},
            to_weights={"A": 0.50, "B": 0.30, "C": 0.25},
            turnover=0.075,
        )

        assert orders.n_trades == 2  # A and B, not C

    def test_buy_sell_symbols(self):
        """buy_symbols and sell_symbols should return correct lists."""
        orders = RebalanceOrders(
            date=date(2023, 1, 3),
            trades={"A": 0.10, "B": -0.05, "C": 0.02},
            from_weights={},
            to_weights={},
            turnover=0.085,
        )

        assert set(orders.buy_symbols) == {"A", "C"}
        assert orders.sell_symbols == ["B"]


class TestRebalanceResult:
    """Contract tests for RebalanceResult."""

    def test_rebalance_result_tracks_orders(self):
        """RebalanceResult should track all orders."""
        result = RebalanceResult()

        order1 = RebalanceOrders(
            date=date(2023, 1, 3),
            trades={"A": 0.10},
            from_weights={"A": 0.40},
            to_weights={"A": 0.50},
            turnover=0.05,
        )
        order2 = RebalanceOrders(
            date=date(2023, 2, 1),
            trades={"A": -0.05},
            from_weights={"A": 0.50},
            to_weights={"A": 0.45},
            turnover=0.025,
        )

        result.orders.append(order1)
        result.orders.append(order2)

        assert len(result.orders) == 2
        assert result.dates == [date(2023, 1, 3), date(2023, 2, 1)]

    def test_total_and_avg_turnover(self):
        """total_turnover and avg_turnover should calculate correctly."""
        result = RebalanceResult()
        result.orders = [
            RebalanceOrders(
                date=date(2023, 1, 3), trades={}, from_weights={}, to_weights={}, turnover=0.10
            ),
            RebalanceOrders(
                date=date(2023, 2, 1), trades={}, from_weights={}, to_weights={}, turnover=0.06
            ),
        ]

        assert abs(result.total_turnover - 0.16) < 0.001
        assert abs(result.avg_turnover - 0.08) < 0.001


class TestRebalancerFactories:
    """Contract tests for Rebalancer factory methods."""

    def test_calendar_factory(self):
        """calendar() should create calendar-based rebalancer."""
        rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY, turnover_limit=0.10)

        assert isinstance(rebalancer.trigger, CalendarTrigger)
        assert rebalancer.turnover_limit == 0.10

    def test_drift_factory(self):
        """drift() should create drift-based rebalancer."""
        rebalancer = Rebalancer.drift(threshold=0.05, turnover_limit=0.20)

        assert isinstance(rebalancer.trigger, DriftTrigger)
        assert rebalancer.turnover_limit == 0.20

    def test_hybrid_factory(self):
        """hybrid() should create hybrid rebalancer."""
        rebalancer = Rebalancer.hybrid(
            calendar_frequency=RebalanceFrequency.MONTHLY,
            drift_threshold=0.10,
            turnover_limit=0.15,
        )

        assert isinstance(rebalancer.trigger, HybridTrigger)
        assert len(rebalancer.trigger.triggers) == 2
        assert rebalancer.turnover_limit == 0.15
