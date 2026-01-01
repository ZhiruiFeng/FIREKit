"""
Base Strategy

Abstract base class for all trading strategies in VectorForge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vectorforge.engine.event_driven import Bar, Order, Position


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    """Trading signal with metadata."""
    type: SignalType
    strength: float = 1.0  # Signal strength/confidence [0, 1]
    target_weight: float = 0.0  # Target portfolio weight
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies can implement either or both modes:
    - Vectorized: Implement generate_signals() for fast backtesting
    - Event-driven: Implement on_bar() for production-like simulation

    Example (Vectorized):
        >>> class MomentumStrategy(BaseStrategy):
        ...     def __init__(self, lookback: int = 20):
        ...         self.lookback = lookback
        ...
        ...     def generate_signals(self, close, **kwargs):
        ...         returns = np.diff(close) / close[:-1]
        ...         momentum = np.convolve(returns, np.ones(self.lookback)/self.lookback, 'valid')
        ...         return np.sign(momentum)

    Example (Event-driven):
        >>> class MACrossover(BaseStrategy):
        ...     def __init__(self, fast=10, slow=30):
        ...         self.fast, self.slow = fast, slow
        ...         self.prices = []
        ...
        ...     def on_bar(self, bar, position, cash):
        ...         self.prices.append(bar.close)
        ...         if len(self.prices) < self.slow:
        ...             return None
        ...         fast_ma = np.mean(self.prices[-self.fast:])
        ...         slow_ma = np.mean(self.prices[-self.slow:])
        ...         if fast_ma > slow_ma and (position is None or position.quantity <= 0):
        ...             return Order(bar.symbol, OrderSide.BUY, self.position_size(cash, bar.close))
        ...         return None
    """

    def __init__(self, **params):
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    def generate_signals(
        self,
        close: np.ndarray,
        open: np.ndarray | None = None,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate signals for entire price series (vectorized mode).

        Override this method for vectorized backtesting.

        Args:
            close: Close prices array
            open: Open prices array
            high: High prices array
            low: Low prices array
            volume: Volume array
            **kwargs: Additional data

        Returns:
            Array of signals: 1 (long), -1 (short), 0 (flat)
        """
        raise NotImplementedError(
            f"{self._name} does not implement vectorized signal generation. "
            "Override generate_signals() or use event-driven mode with on_bar()."
        )

    def on_bar(
        self,
        bar: "Bar",
        position: "Position | None",
        cash: float,
    ) -> "Order | None":
        """
        Process a single bar and return order (event-driven mode).

        Override this method for event-driven backtesting.

        Args:
            bar: Current OHLCV bar
            position: Current position (None if no position)
            cash: Available cash

        Returns:
            Order to execute, or None for no action
        """
        raise NotImplementedError(
            f"{self._name} does not implement event-driven processing. "
            "Override on_bar() or use vectorized mode with generate_signals()."
        )

    def on_fill(self, fill: Any) -> None:
        """
        Handle order fill notification.

        Override to track fills and update internal state.

        Args:
            fill: Fill object with execution details
        """
        pass

    def reset(self) -> None:
        """
        Reset strategy state for new backtest run.

        Override to clear any accumulated state.
        """
        pass

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self._name}({params_str})"


class MomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy for demonstration.

    Goes long when recent returns are positive, short when negative.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(lookback=lookback)
        self.lookback = lookback

    def generate_signals(
        self,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Generate momentum signals."""
        if len(close) < self.lookback + 1:
            return np.zeros(len(close))

        # Compute returns
        returns = np.diff(close) / close[:-1]

        # Rolling momentum (mean of recent returns)
        kernel = np.ones(self.lookback) / self.lookback
        momentum = np.convolve(returns, kernel, mode="valid")

        # Pad to match price length
        padding = len(close) - len(momentum)
        signals = np.concatenate([np.zeros(padding), np.sign(momentum)])

        return signals


class MovingAverageCrossover(BaseStrategy):
    """
    Moving average crossover strategy.

    Goes long when fast MA crosses above slow MA, and vice versa.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: list[float] = []

    def generate_signals(
        self,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Generate MA crossover signals."""
        if len(close) < self.slow_period:
            return np.zeros(len(close))

        # Compute moving averages
        fast_ma = np.convolve(close, np.ones(self.fast_period) / self.fast_period, mode="valid")
        slow_ma = np.convolve(close, np.ones(self.slow_period) / self.slow_period, mode="valid")

        # Align arrays
        min_len = min(len(fast_ma), len(slow_ma))
        fast_ma = fast_ma[-min_len:]
        slow_ma = slow_ma[-min_len:]

        # Generate signals
        signals = np.where(fast_ma > slow_ma, 1, -1)

        # Pad to match price length
        padding = len(close) - len(signals)
        signals = np.concatenate([np.zeros(padding), signals])

        return signals

    def on_bar(
        self,
        bar: "Bar",
        position: "Position | None",
        cash: float,
    ) -> "Order | None":
        """Event-driven MA crossover logic."""
        from vectorforge.engine.event_driven import Order, OrderSide, OrderType

        self._prices.append(bar.close)

        if len(self._prices) < self.slow_period:
            return None

        fast_ma = np.mean(self._prices[-self.fast_period :])
        slow_ma = np.mean(self._prices[-self.slow_period :])

        current_qty = position.quantity if position else 0

        if fast_ma > slow_ma and current_qty <= 0:
            # Buy signal
            size = int(cash * 0.95 / bar.close)  # Use 95% of cash
            if size > 0:
                return Order(
                    symbol=bar.symbol,
                    side=OrderSide.BUY,
                    quantity=size,
                    order_type=OrderType.MARKET,
                    timestamp=bar.timestamp,
                )

        elif fast_ma < slow_ma and current_qty > 0:
            # Sell signal
            return Order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                quantity=current_qty,
                order_type=OrderType.MARKET,
                timestamp=bar.timestamp,
            )

        return None

    def reset(self) -> None:
        """Reset price history."""
        self._prices = []
