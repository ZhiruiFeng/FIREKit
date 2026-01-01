"""
Walk-Forward Optimizer

Rolling window optimization for robust parameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vectorforge.engine.base import BacktestResult
from vectorforge.optimization.grid_search import GridSearch

if TYPE_CHECKING:
    from vectorforge.engine.base import BacktestEngine
    from vectorforge.strategy.base import BaseStrategy


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward period."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    degradation: float  # train_sharpe - test_sharpe


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust parameter selection.

    Prevents overfitting by using rolling in-sample/out-of-sample
    windows for parameter optimization and validation.

    Example:
        >>> wfo = WalkForwardOptimizer(
        ...     engine=backtester,
        ...     train_period=252,  # 1 year training
        ...     test_period=63,    # 1 quarter testing
        ... )
        >>> results = wfo.run(
        ...     strategy_class=MomentumStrategy,
        ...     param_grid={"lookback": range(10, 50)},
        ...     data=price_data,
        ... )
    """

    def __init__(
        self,
        engine: BacktestEngine,
        train_period: int = 252,
        test_period: int = 63,
        step_period: int | None = None,
        anchored: bool = False,
        min_trades: int = 30,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            engine: Backtesting engine
            train_period: Training window size in bars
            test_period: Test window size in bars
            step_period: Step size for rolling (default=test_period)
            anchored: If True, training window expands from start
            min_trades: Minimum trades required for valid optimization
        """
        self.engine = engine
        self.train_period = train_period
        self.test_period = test_period
        self.step_period = step_period or test_period
        self.anchored = anchored
        self.min_trades = min_trades
        self.grid_search = GridSearch(engine)

    def run(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_capital: float | None = None,
        metric: str = "sharpe_ratio",
    ) -> list[WalkForwardResult]:
        """
        Run walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Parameter combinations to test
            data: Full OHLCV dataset
            initial_capital: Starting capital
            metric: Metric to optimize

        Returns:
            List of WalkForwardResult for each period
        """
        results = []
        windows = self._get_windows(data)

        for train_start, train_end, test_start, test_end in windows:
            # Get data slices
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]

            # Optimize on training data
            search_results = self.grid_search.search(
                strategy_class=strategy_class,
                param_grid=param_grid,
                data=train_data,
                initial_capital=initial_capital,
                metric=metric,
            )

            if not search_results:
                continue

            # Get best parameters
            best = search_results[0]
            best_params = best["params"]
            train_result = best["result"]

            # Validate on test data
            strategy = strategy_class(**best_params)
            test_result = self.engine.run(strategy, test_data, initial_capital)

            # Check minimum trades
            if train_result.total_trades < self.min_trades:
                continue

            # Record results
            wf_result = WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_sharpe=train_result.sharpe_ratio,
                test_sharpe=test_result.sharpe_ratio,
                train_return=train_result.total_return,
                test_return=test_result.total_return,
                degradation=train_result.sharpe_ratio - test_result.sharpe_ratio,
            )
            results.append(wf_result)

        return results

    def _get_windows(
        self, data: pd.DataFrame
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test window pairs."""
        windows = []
        n = len(data)
        idx = data.index

        if self.anchored:
            # Expanding window from start
            train_start = 0
            test_start = self.train_period

            while test_start + self.test_period <= n:
                train_end = test_start - 1
                test_end = test_start + self.test_period - 1

                windows.append((
                    idx[train_start],
                    idx[train_end],
                    idx[test_start],
                    idx[test_end],
                ))

                test_start += self.step_period
        else:
            # Rolling window
            start = 0
            while start + self.train_period + self.test_period <= n:
                train_start = start
                train_end = start + self.train_period - 1
                test_start = start + self.train_period
                test_end = test_start + self.test_period - 1

                windows.append((
                    idx[train_start],
                    idx[train_end],
                    idx[test_start],
                    idx[test_end],
                ))

                start += self.step_period

        return windows

    def summary(self, results: list[WalkForwardResult]) -> dict[str, Any]:
        """Generate summary statistics from walk-forward results."""
        if not results:
            return {}

        test_sharpes = [r.test_sharpe for r in results]
        train_sharpes = [r.train_sharpe for r in results]
        degradations = [r.degradation for r in results]

        return {
            "periods": len(results),
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "avg_train_sharpe": np.mean(train_sharpes),
            "avg_degradation": np.mean(degradations),
            "consistency": sum(1 for s in test_sharpes if s > 0) / len(test_sharpes),
            "efficiency_ratio": np.mean(test_sharpes) / np.mean(train_sharpes)
            if np.mean(train_sharpes) != 0
            else 0,
        }
