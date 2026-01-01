"""
Event-Driven Backtester

Production-grade sequential simulation that mirrors live trading behavior.
Provides realistic execution modeling with slippage and commission.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from vectorforge.engine.base import BacktestEngine, BacktestResult

if TYPE_CHECKING:
    from vectorforge.strategy.base import BaseStrategy
    from vectorforge.config import VectorForgeConfig


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
    """Trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    timestamp: datetime | None = None
    order_id: str = ""


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
    """Simulates order execution with realistic costs."""

    def __init__(self, config: VectorForgeConfig):
        self.config = config
        self.pending_orders: list[Order] = []
        self.fills: list[Fill] = []
        self._order_counter = 0

    def submit_order(self, order: Order) -> str:
        """Submit an order for execution."""
        self._order_counter += 1
        order.order_id = f"ORD-{self._order_counter:06d}"
        self.pending_orders.append(order)
        return order.order_id

    def process_bar(self, bar: Bar) -> list[Fill]:
        """Process pending orders against current bar."""
        fills = []
        remaining_orders = []

        for order in self.pending_orders:
            if order.symbol != bar.symbol:
                remaining_orders.append(order)
                continue

            fill = self._try_fill(order, bar)
            if fill:
                fills.append(fill)
                self.fills.append(fill)
            else:
                remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return fills

    def _try_fill(self, order: Order, bar: Bar) -> Fill | None:
        """Attempt to fill an order at current bar."""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at open with slippage
            base_price = bar.open
            slippage = self._compute_slippage(order, bar)
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
        self.trade_history.append({
            "timestamp": fill.timestamp,
            "symbol": symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.fill_price,
            "commission": fill.commission,
            "slippage": fill.slippage,
        })

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
