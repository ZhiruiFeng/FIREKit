"""
Event-Driven Backtester

Production-grade sequential simulation that mirrors live trading behavior.
Provides realistic execution modeling with slippage and commission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vectorforge.engine.base import BacktestEngine, BacktestResult

if TYPE_CHECKING:
    from vectorforge.config import VectorForgeConfig
    from vectorforge.strategy.base import BaseStrategy


class EventType(Enum):
    """Types of events in the simulation."""

    BAR = "bar"
    ORDER = "order"
    FILL = "fill"
    SIGNAL = "signal"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TRAILING_STOP_LIMIT = "trailing_stop_limit"


class TimeInForce(Enum):
    """Time-in-force options for orders."""
    DAY = "day"           # Valid for the current trading day only
    GTC = "gtc"           # Good 'til canceled
    GTD = "gtd"           # Good 'til date
    IOC = "ioc"           # Immediate or cancel
    FOK = "fok"           # Fill or kill (all or nothing)
    OPG = "opg"           # At the open
    CLS = "cls"           # At the close


@dataclass
class Event:
    """Base event in the simulation."""

    type: EventType
    timestamp: datetime
    data: dict = field(default_factory=dict)


@dataclass
class Bar:
    """OHLCV bar data."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    """Trading order with advanced order type support."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    timestamp: datetime | None = None
    order_id: str = ""

    # Advanced order fields
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_date: datetime | None = None  # For GTD orders
    trail_amount: float | None = None    # Absolute trail amount for trailing stops
    trail_percent: float | None = None   # Percentage trail for trailing stops
    parent_order_id: str | None = None   # For bracket orders (OCO)
    oco_group_id: str | None = None      # One-Cancels-Other group


@dataclass
class BracketOrder:
    """
    Bracket order (entry + take profit + stop loss).

    Creates three linked orders where filling the entry order
    activates the profit target and stop loss (OCO pair).
    """
    entry_order: Order
    take_profit_order: Order
    stop_loss_order: Order
    group_id: str = ""

    def __post_init__(self):
        """Link orders together."""
        if not self.group_id:
            self.group_id = f"BKT-{id(self):06d}"

        self.take_profit_order.oco_group_id = self.group_id
        self.stop_loss_order.oco_group_id = self.group_id
        self.take_profit_order.parent_order_id = self.entry_order.order_id
        self.stop_loss_order.parent_order_id = self.entry_order.order_id


@dataclass
class Fill:
    """Order fill/execution."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    timestamp: datetime


@dataclass
class Position:
    """Current position in a symbol."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class SimulatedBroker:
    """
    Simulates order execution with realistic costs and advanced order types.

    Supports:
    - Market, Limit, Stop, Stop-Limit orders
    - Trailing stop orders (absolute and percentage)
    - Bracket orders (OCO - One Cancels Other)
    - Time-in-force options (DAY, GTC, GTD, IOC, FOK)
    """

    def __init__(self, config: VectorForgeConfig):
        self.config = config
        self.pending_orders: list[Order] = []
        self.active_oco_groups: dict[str, list[Order]] = {}  # OCO group tracking
        self.trailing_stop_prices: dict[str, float] = {}  # Track trailing stop levels
        self.fills: list[Fill] = []
        self._order_counter = 0
        self._current_date: datetime | None = None

    def submit_order(self, order: Order) -> str:
        """Submit an order for execution."""
        self._order_counter += 1
        order.order_id = f"ORD-{self._order_counter:06d}"
        self.pending_orders.append(order)

        # Track OCO groups
        if order.oco_group_id:
            if order.oco_group_id not in self.active_oco_groups:
                self.active_oco_groups[order.oco_group_id] = []
            self.active_oco_groups[order.oco_group_id].append(order)

        return order.order_id

    def submit_bracket_order(self, bracket: BracketOrder) -> tuple[str, str, str]:
        """Submit a bracket order (entry + TP + SL)."""
        entry_id = self.submit_order(bracket.entry_order)
        tp_id = self.submit_order(bracket.take_profit_order)
        sl_id = self.submit_order(bracket.stop_loss_order)
        return entry_id, tp_id, sl_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for i, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                self.pending_orders.pop(i)
                # Clean up OCO group
                if order.oco_group_id and order.oco_group_id in self.active_oco_groups:
                    self.active_oco_groups[order.oco_group_id] = [
                        o for o in self.active_oco_groups[order.oco_group_id]
                        if o.order_id != order_id
                    ]
                return True
        return False

    def _cancel_oco_group(self, group_id: str, except_order_id: str) -> None:
        """Cancel all orders in an OCO group except the filled one."""
        if group_id not in self.active_oco_groups:
            return

        orders_to_cancel = [
            o.order_id for o in self.active_oco_groups[group_id]
            if o.order_id != except_order_id
        ]

        for order_id in orders_to_cancel:
            self.cancel_order(order_id)

        del self.active_oco_groups[group_id]

    def process_bar(self, bar: Bar) -> list[Fill]:
        """Process pending orders against current bar."""
        fills = []
        remaining_orders = []
        self._current_date = bar.timestamp

        # First, update trailing stop prices
        self._update_trailing_stops(bar)

        # Check for expired orders
        self._expire_orders(bar.timestamp)

        for order in self.pending_orders:
            if order.symbol != bar.symbol:
                remaining_orders.append(order)
                continue

            fill = self._try_fill(order, bar)
            if fill:
                fills.append(fill)
                self.fills.append(fill)

                # Handle OCO group cancellation
                if order.oco_group_id:
                    self._cancel_oco_group(order.oco_group_id, order.order_id)
            else:
                # Check IOC and FOK
                if order.time_in_force == TimeInForce.IOC:
                    # IOC orders that don't fill immediately are cancelled
                    continue
                elif order.time_in_force == TimeInForce.FOK:
                    # FOK orders that don't fill completely are cancelled
                    continue
                remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return fills

    def _update_trailing_stops(self, bar: Bar) -> None:
        """Update trailing stop prices based on current bar."""
        for order in self.pending_orders:
            if order.order_type not in (OrderType.TRAILING_STOP, OrderType.TRAILING_STOP_LIMIT):
                continue

            if order.symbol != bar.symbol:
                continue

            key = order.order_id

            if key not in self.trailing_stop_prices:
                # Initialize trailing stop
                if order.side == OrderSide.SELL:
                    # Trailing stop to sell: stop price trails below high
                    if order.trail_percent:
                        self.trailing_stop_prices[key] = bar.high * (1 - order.trail_percent / 100)
                    elif order.trail_amount:
                        self.trailing_stop_prices[key] = bar.high - order.trail_amount
                else:
                    # Trailing stop to buy: stop price trails above low
                    if order.trail_percent:
                        self.trailing_stop_prices[key] = bar.low * (1 + order.trail_percent / 100)
                    elif order.trail_amount:
                        self.trailing_stop_prices[key] = bar.low + order.trail_amount
            else:
                # Update trailing stop
                current_stop = self.trailing_stop_prices[key]
                if order.side == OrderSide.SELL:
                    # Move stop up with higher highs
                    if order.trail_percent:
                        new_stop = bar.high * (1 - order.trail_percent / 100)
                    else:
                        new_stop = bar.high - (order.trail_amount or 0)
                    self.trailing_stop_prices[key] = max(current_stop, new_stop)
                else:
                    # Move stop down with lower lows
                    if order.trail_percent:
                        new_stop = bar.low * (1 + order.trail_percent / 100)
                    else:
                        new_stop = bar.low + (order.trail_amount or 0)
                    self.trailing_stop_prices[key] = min(current_stop, new_stop)

    def _expire_orders(self, current_time: datetime) -> None:
        """Expire orders based on time-in-force."""
        remaining = []
        for order in self.pending_orders:
            if order.time_in_force == TimeInForce.GTD:
                if order.expire_date and current_time > order.expire_date:
                    continue  # Order expired
            # DAY orders expire at end of day (handled elsewhere)
            remaining.append(order)
        self.pending_orders = remaining

    def end_of_day(self) -> None:
        """Called at end of trading day to expire DAY orders."""
        self.pending_orders = [
            o for o in self.pending_orders
            if o.time_in_force != TimeInForce.DAY
        ]

    def _try_fill(self, order: Order, bar: Bar) -> Fill | None:
        """Attempt to fill an order at current bar."""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at open with slippage
            base_price = bar.open
            slippage = self._compute_slippage(order, bar)
            fill_price = base_price * (
                1 + slippage if order.side == OrderSide.BUY else 1 - slippage
            )
            commission = self._compute_commission(order, fill_price)

            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage * base_price * order.quantity,
                timestamp=bar.timestamp,
            )

        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill if price crosses limit
            if order.limit_price is None:
                return None

            if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                fill_price = min(order.limit_price, bar.open)
                commission = self._compute_commission(order, fill_price)
                return Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0,
                    timestamp=bar.timestamp,
                )
            elif order.side == OrderSide.SELL and bar.high >= order.limit_price:
                fill_price = max(order.limit_price, bar.open)
                commission = self._compute_commission(order, fill_price)
                return Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0,
                    timestamp=bar.timestamp,
                )

        elif order.order_type == OrderType.STOP:
            # Stop orders trigger when price crosses stop level
            if order.stop_price is None:
                return None

            triggered = False
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                triggered = True
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                triggered = True

            if triggered:
                # Convert to market order and fill
                slippage = self._compute_slippage(order, bar)
                base_price = order.stop_price
                fill_price = base_price * (1 + slippage if order.side == OrderSide.BUY else 1 - slippage)
                commission = self._compute_commission(order, fill_price)
                return Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=slippage * base_price * order.quantity,
                    timestamp=bar.timestamp,
                )

        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit: triggers at stop, then becomes limit order
            if order.stop_price is None or order.limit_price is None:
                return None

            triggered = False
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                triggered = True
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                triggered = True

            if triggered:
                # Check if limit can be filled
                if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                    fill_price = min(order.limit_price, max(bar.open, order.stop_price))
                    commission = self._compute_commission(order, fill_price)
                    return Fill(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        commission=commission,
                        slippage=0,
                        timestamp=bar.timestamp,
                    )
                elif order.side == OrderSide.SELL and bar.high >= order.limit_price:
                    fill_price = max(order.limit_price, min(bar.open, order.stop_price))
                    commission = self._compute_commission(order, fill_price)
                    return Fill(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        commission=commission,
                        slippage=0,
                        timestamp=bar.timestamp,
                    )

        elif order.order_type == OrderType.TRAILING_STOP:
            # Trailing stop: uses dynamically updated stop price
            stop_price = self.trailing_stop_prices.get(order.order_id)
            if stop_price is None:
                return None

            triggered = False
            if order.side == OrderSide.SELL and bar.low <= stop_price:
                triggered = True
            elif order.side == OrderSide.BUY and bar.high >= stop_price:
                triggered = True

            if triggered:
                slippage = self._compute_slippage(order, bar)
                fill_price = stop_price * (1 + slippage if order.side == OrderSide.BUY else 1 - slippage)
                commission = self._compute_commission(order, fill_price)

                # Clean up trailing stop tracking
                del self.trailing_stop_prices[order.order_id]

                return Fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=slippage * stop_price * order.quantity,
                    timestamp=bar.timestamp,
                )

        elif order.order_type == OrderType.TRAILING_STOP_LIMIT:
            # Trailing stop limit: trailing stop that becomes a limit order
            stop_price = self.trailing_stop_prices.get(order.order_id)
            if stop_price is None or order.limit_price is None:
                return None

            triggered = False
            if order.side == OrderSide.SELL and bar.low <= stop_price:
                triggered = True
            elif order.side == OrderSide.BUY and bar.high >= stop_price:
                triggered = True

            if triggered:
                # Calculate limit price offset from trailing stop
                limit_offset = abs(stop_price - order.limit_price) if order.limit_price else 0

                if order.side == OrderSide.SELL:
                    actual_limit = stop_price - limit_offset
                    if bar.high >= actual_limit:
                        fill_price = max(actual_limit, min(bar.open, stop_price))
                        commission = self._compute_commission(order, fill_price)
                        del self.trailing_stop_prices[order.order_id]
                        return Fill(
                            order_id=order.order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            fill_price=fill_price,
                            commission=commission,
                            slippage=0,
                            timestamp=bar.timestamp,
                        )
                else:
                    actual_limit = stop_price + limit_offset
                    if bar.low <= actual_limit:
                        fill_price = min(actual_limit, max(bar.open, stop_price))
                        commission = self._compute_commission(order, fill_price)
                        del self.trailing_stop_prices[order.order_id]
                        return Fill(
                            order_id=order.order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            fill_price=fill_price,
                            commission=commission,
                            slippage=0,
                            timestamp=bar.timestamp,
                        )

        return None

    def _compute_slippage(self, order: Order, bar: Bar) -> float:
        """Compute slippage for an order."""
        config = self.config.execution.slippage

        if config.model.value == "fixed":
            return config.base_bps / 10000

        elif config.model.value == "volume_dependent":
            participation = order.quantity / max(bar.volume, 1)
            return config.impact_factor * np.sqrt(participation) + config.base_bps / 10000

        elif config.model.value == "almgren_chriss":
            # Simplified Almgren-Chriss model
            participation = order.quantity / max(bar.volume, 1)
            volatility = 0.02  # Assumed daily volatility
            eta = 0.01
            gamma = 0.1
            temporary = eta * volatility * participation
            permanent = gamma * volatility * np.sqrt(participation)
            return temporary + permanent

        return config.base_bps / 10000

    def _compute_commission(self, order: Order, fill_price: float) -> float:
        """Compute commission for an order."""
        config = self.config.execution.commission
        trade_value = order.quantity * fill_price

        if config.model.value == "zero":
            return 0.0

        elif config.model.value == "fixed":
            return config.min_commission

        elif config.model.value == "per_share":
            per_share_cost = order.quantity * config.per_share
            return max(config.min_commission, min(per_share_cost, trade_value * config.max_pct))

        elif config.model.value == "percentage":
            return trade_value * config.per_share

        elif config.model.value == "tiered":
            # IBKR-style tiered pricing
            per_share = max(0.0035, min(0.005, 1.0 / order.quantity))
            cost = per_share * order.quantity
            return max(0.35, min(cost, trade_value * 0.01))

        return config.min_commission


class EventDrivenBacktester(BacktestEngine):
    """
    Event-driven backtesting engine for production validation.

    Processes bars sequentially, simulating realistic market conditions
    with slippage, commission, and fill modeling.

    Example:
        >>> backtester = EventDrivenBacktester()
        >>> strategy = MovingAverageCrossover(fast=10, slow=30)
        >>> results = backtester.run(strategy, data)
    """

    def __init__(self, config: VectorForgeConfig | None = None):
        super().__init__(config)
        self.broker: SimulatedBroker | None = None
        self.positions: dict[str, Position] = {}
        self.cash: float = 0.0
        self.equity_history: list[float] = []
        self.trade_history: list[dict] = []

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """
        Run event-driven backtest.

        Args:
            strategy: Strategy with on_bar method
            data: OHLCV DataFrame with DatetimeIndex
            initial_capital: Starting capital

        Returns:
            BacktestResult with performance metrics
        """
        self.validate_data(data)
        self._is_running = True
        start_time = time.perf_counter()

        try:
            # Initialize
            self.cash = initial_capital or self.config.default_capital
            self.positions = {}
            self.equity_history = [self.cash]
            self.trade_history = []
            self.broker = SimulatedBroker(self.config)

            # Infer symbol from data
            symbol = data.attrs.get("symbol", "UNKNOWN")

            # Process each bar
            for timestamp, row in data.iterrows():
                bar = Bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )

                # Process pending orders
                fills = self.broker.process_bar(bar)
                for fill in fills:
                    self._process_fill(fill, bar)

                # Get strategy signal
                order = strategy.on_bar(bar, self.positions.get(symbol), self.cash)
                if order:
                    self.broker.submit_order(order)

                # Update equity
                equity = self._compute_equity(bar)
                self.equity_history.append(equity)

            # Build result
            result = self._build_result(
                data=data,
                initial_capital=initial_capital or self.config.default_capital,
                execution_time=time.perf_counter() - start_time,
            )

            return result

        finally:
            self._is_running = False

    def run_batch(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> list[BacktestResult]:
        """Run multiple backtests with different parameters."""
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            strategy = strategy_class(**params)
            result = self.run(strategy, data, initial_capital)
            results.append(result)

        return results

    def _process_fill(self, fill: Fill, bar: Bar) -> None:
        """Process an order fill and update positions."""
        symbol = fill.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]

        if fill.side == OrderSide.BUY:
            # Update average price
            total_cost = position.quantity * position.avg_price + fill.quantity * fill.fill_price
            position.quantity += fill.quantity
            position.avg_price = total_cost / position.quantity if position.quantity > 0 else 0
            self.cash -= fill.quantity * fill.fill_price + fill.commission

        else:  # SELL
            if position.quantity > 0:
                realized = (fill.fill_price - position.avg_price) * fill.quantity
                position.realized_pnl += realized
            position.quantity -= fill.quantity
            self.cash += fill.quantity * fill.fill_price - fill.commission

        # Record trade
        self.trade_history.append(
            {
                "timestamp": fill.timestamp,
                "symbol": symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.fill_price,
                "commission": fill.commission,
                "slippage": fill.slippage,
            }
        )

    def _compute_equity(self, bar: Bar) -> float:
        """Compute current portfolio equity."""
        equity = self.cash

        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # Use bar close for mark-to-market
                equity += position.quantity * bar.close

        return equity

    def _build_result(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        execution_time: float,
    ) -> BacktestResult:
        """Build BacktestResult from simulation data."""
        equity = np.array(self.equity_history)
        returns = np.diff(equity) / equity[:-1]

        # Performance metrics
        total_return = equity[-1] / equity[0] - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        daily_std = np.std(returns) if len(returns) > 0 else 1e-10
        sharpe = np.mean(returns) / max(daily_std, 1e-10) * np.sqrt(252)

        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        sortino = np.mean(returns) / max(downside_std, 1e-10) * np.sqrt(252)

        running_max = np.maximum.accumulate(equity)
        drawdowns = equity / running_max - 1
        max_drawdown = np.min(drawdowns)
        calmar = annual_return / max(abs(max_drawdown), 1e-10)

        # Trade statistics
        total_trades = len(self.trade_history)
        if total_trades > 0 and len(returns) > 0:
            win_rate = np.mean(returns > 0)
            gains = np.sum(returns[returns > 0])
            losses = abs(np.sum(returns[returns < 0]))
            profit_factor = gains / max(losses, 1e-10)
            avg_trade = np.mean(returns)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade = 0.0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade,
            equity_curve=pd.Series(equity, index=data.index[: len(equity)]),
            returns=pd.Series(returns, index=data.index[1 : len(returns) + 1]),
            trades=pd.DataFrame(self.trade_history),
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=initial_capital,
            final_capital=equity[-1],
            execution_time=execution_time,
            mode="event_driven",
        )
