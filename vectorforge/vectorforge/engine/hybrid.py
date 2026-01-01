"""
Hybrid Runner

Automatically selects between vectorized and event-driven modes
based on strategy requirements and optimization goals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from vectorforge.engine.base import BacktestEngine, BacktestResult
from vectorforge.engine.vectorized import VectorizedBacktester
from vectorforge.engine.event_driven import EventDrivenBacktester

if TYPE_CHECKING:
    from vectorforge.strategy.base import BaseStrategy
    from vectorforge.config import VectorForgeConfig


class HybridRunner(BacktestEngine):
    """
    Hybrid backtesting runner with automatic mode selection.

    Uses vectorized mode for:
    - Initial parameter sweeps and optimization
    - Strategies that support array-based signal generation
    - When speed is prioritized over execution accuracy

    Uses event-driven mode for:
    - Final validation before live deployment
    - Strategies requiring order-level logic
    - When realistic execution modeling is required

    Example:
        >>> runner = HybridRunner()
        >>> # Auto-selects vectorized for parameter sweep
        >>> results = runner.run_batch(
        ...     strategy_class=MomentumStrategy,
        ...     param_grid={"lookback": range(10, 50)},
        ...     data=data
        ... )
        >>> # Final validation with event-driven
        >>> final = runner.validate(best_strategy, data)
    """

    def __init__(self, config: VectorForgeConfig | None = None):
        super().__init__(config)
        self.vectorized = VectorizedBacktester(config)
        self.event_driven = EventDrivenBacktester(config)

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
        mode: str | None = None,
    ) -> BacktestResult:
        """
        Run backtest with automatic or specified mode.

        Args:
            strategy: Trading strategy
            data: OHLCV DataFrame
            initial_capital: Starting capital
            mode: Force "vectorized" or "event_driven", or None for auto

        Returns:
            BacktestResult
        """
        if mode is None:
            mode = self._select_mode(strategy)

        if mode == "vectorized":
            return self.vectorized.run(strategy, data, initial_capital)
        else:
            return self.event_driven.run(strategy, data, initial_capital)

    def run_batch(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> list[BacktestResult]:
        """
        Run parameter sweep using vectorized mode for speed.

        Args:
            strategy_class: Strategy class to test
            param_grid: Parameter combinations
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            List of BacktestResult
        """
        # Use vectorized for batch operations
        return self.vectorized.run_batch(
            strategy_class, param_grid, data, initial_capital
        )

    def validate(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """
        Validate strategy using event-driven mode.

        Use this for final validation before live deployment.

        Args:
            strategy: Strategy to validate
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            BacktestResult with realistic execution modeling
        """
        return self.event_driven.run(strategy, data, initial_capital)

    def compare_modes(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Run strategy in both modes and compare results.

        Useful for understanding the gap between idealized
        vectorized results and realistic event-driven execution.

        Args:
            strategy: Strategy to test
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            Dict with "vectorized" and "event_driven" results
        """
        vectorized_result = self.vectorized.run(strategy, data, initial_capital)
        event_driven_result = self.event_driven.run(strategy, data, initial_capital)

        return {
            "vectorized": vectorized_result,
            "event_driven": event_driven_result,
            "sharpe_gap": vectorized_result.sharpe_ratio - event_driven_result.sharpe_ratio,
            "return_gap": vectorized_result.total_return - event_driven_result.total_return,
        }

    def _select_mode(self, strategy: BaseStrategy) -> str:
        """
        Automatically select execution mode based on strategy capabilities.

        Returns:
            "vectorized" or "event_driven"
        """
        # Check if strategy supports vectorized operations
        if hasattr(strategy, "generate_signals"):
            return "vectorized"

        # Check if strategy requires event-driven features
        if hasattr(strategy, "on_bar"):
            return "event_driven"

        # Default to vectorized for speed
        return "vectorized"
