"""
Grid Search Optimizer

Exhaustive parameter search for strategy optimization.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from vectorforge.engine.base import BacktestResult

if TYPE_CHECKING:
    from vectorforge.engine.base import BacktestEngine
    from vectorforge.strategy.base import BaseStrategy


class GridSearch:
    """
    Exhaustive grid search parameter optimizer.

    Tests all combinations of parameters and returns
    results sorted by specified metric.

    Example:
        >>> grid = GridSearch(backtester)
        >>> results = grid.search(
        ...     strategy_class=MomentumStrategy,
        ...     param_grid={"lookback": range(10, 50, 5)},
        ...     data=price_data,
        ...     metric="sharpe_ratio"
        ... )
        >>> print(results[0])  # Best parameters
    """

    def __init__(self, engine: BacktestEngine):
        """
        Initialize grid search.

        Args:
            engine: Backtesting engine to use
        """
        self.engine = engine

    def search(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_capital: float | None = None,
        metric: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Run exhaustive parameter search.

        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dict of param names to value lists
            data: OHLCV data for backtesting
            initial_capital: Starting capital
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            ascending: Sort ascending (False = maximize)

        Returns:
            List of dicts with params and results, sorted by metric
        """
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Run backtest
            strategy = strategy_class(**params)
            result = self.engine.run(strategy, data, initial_capital)

            # Extract metric value
            metric_value = getattr(result, metric, 0.0)

            results.append({
                "params": params,
                "result": result,
                metric: metric_value,
            })

        # Sort by metric
        results.sort(key=lambda x: x[metric], reverse=not ascending)

        return results

    def search_with_filter(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        filter_func: Callable[[BacktestResult], bool],
        initial_capital: float | None = None,
        metric: str = "sharpe_ratio",
    ) -> list[dict[str, Any]]:
        """
        Search with custom result filter.

        Args:
            strategy_class: Strategy class
            param_grid: Parameter grid
            data: OHLCV data
            filter_func: Function to filter valid results
            initial_capital: Starting capital
            metric: Metric to optimize

        Returns:
            Filtered and sorted results
        """
        all_results = self.search(
            strategy_class, param_grid, data, initial_capital, metric
        )

        return [r for r in all_results if filter_func(r["result"])]
