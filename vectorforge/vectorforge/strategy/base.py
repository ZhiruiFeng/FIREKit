"""
Base Strategy

Abstract base class for all trading strategies in VectorForge.
"""

from __future__ import annotations

from abc import ABC
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
        bar: Bar,
        position: Position | None,
        cash: float,
    ) -> Order | None:
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


class PortfolioStrategy(BaseStrategy):
    """
    Base class for portfolio strategies that generate target weights.

    Portfolio strategies differ from single-asset strategies:
    - Instead of signals (1, -1, 0), they generate weight allocations
    - Weights are per-symbol and sum to 1 (or leverage factor)
    - Strategies can access the entire universe at once

    Example:
        >>> class EqualWeightStrategy(PortfolioStrategy):
        ...     def generate_weights(self, portfolio_data, current_weights=None):
        ...         n = portfolio_data.n_symbols
        ...         return np.ones(n) / n

        >>> class MomentumPortfolio(PortfolioStrategy):
        ...     def __init__(self, lookback=252, top_n=10):
        ...         super().__init__(lookback=lookback, top_n=top_n)
        ...         self.lookback = lookback
        ...         self.top_n = top_n
        ...
        ...     def generate_weights(self, portfolio_data, current_weights=None):
        ...         returns = portfolio_data.returns(self.lookback)
        ...         momentum = returns.sum(axis=1)
        ...         top_idx = np.argsort(momentum)[-self.top_n:]
        ...         weights = np.zeros(portfolio_data.n_symbols)
        ...         weights[top_idx] = 1.0 / self.top_n
        ...         return weights
    """

    def generate_weights(
        self,
        portfolio_data: Any,  # PortfolioData
        current_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate target weights for the portfolio.

        Args:
            portfolio_data: PortfolioData instance with multi-asset prices
            current_weights: Current portfolio weights (for drift-aware strategies)

        Returns:
            Array of target weights, shape (n_symbols,)
            Weights should sum to 1.0 (or desired leverage)
        """
        raise NotImplementedError(
            f"{self._name} does not implement generate_weights(). "
            "Override this method to define portfolio allocation logic."
        )

    def generate_signals(
        self,
        close: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Not used for portfolio strategies - use generate_weights() instead.
        """
        raise NotImplementedError(
            f"{self._name} is a PortfolioStrategy. "
            "Use generate_weights() instead of generate_signals()."
        )


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
        raw_signals = np.where(fast_ma > slow_ma, 1, -1)

        # Pad to match price length
        # First valid signal is at index slow_period (need slow_period bars to compute first MA)
        padding = len(close) - len(raw_signals)
        signals = np.concatenate([np.zeros(padding), raw_signals])

        # Ensure signals during warmup period are zero
        signals[:self.slow_period] = 0

        return signals

    def on_bar(
        self,
        bar: Bar,
        position: Position | None,
        cash: float,
    ) -> Order | None:
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
