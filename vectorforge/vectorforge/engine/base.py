"""
Base Backtest Engine

Defines the abstract interface for all backtesting engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vectorforge.strategy.base import BaseStrategy
    from vectorforge.config import VectorForgeConfig


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float

    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Metadata
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_capital: float = 100000.0
    final_capital: float = 0.0
    execution_time: float = 0.0
    mode: str = "unknown"

    def __repr__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  total_return={self.total_return:.2%},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.2f},\n"
            f"  max_drawdown={self.max_drawdown:.2%},\n"
            f"  total_trades={self.total_trades},\n"
            f"  win_rate={self.win_rate:.2%}\n"
            f")"
        )

    def summary(self) -> dict[str, Any]:
        """Generate summary dictionary."""
        return {
            "Total Return": f"{self.total_return:.2%}",
            "Annual Return": f"{self.annual_return:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Initial Capital": f"${self.initial_capital:,.2f}",
            "Final Capital": f"${self.final_capital:,.2f}",
        }


StrategyT = TypeVar("StrategyT", bound="BaseStrategy")


class BacktestEngine(ABC, Generic[StrategyT]):
    """Abstract base class for backtesting engines."""

    def __init__(self, config: VectorForgeConfig | None = None):
        """
        Initialize the backtest engine.

        Args:
            config: VectorForge configuration. Uses defaults if not provided.
        """
        from vectorforge.config import VectorForgeConfig
        self.config = config or VectorForgeConfig.default()
        self._is_running = False

    @abstractmethod
    def run(
        self,
        strategy: StrategyT,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """
        Run a backtest with the given strategy and data.

        Args:
            strategy: Trading strategy to test
            data: Historical price data with OHLCV columns
            initial_capital: Starting capital (uses config default if not specified)

        Returns:
            BacktestResult containing performance metrics and trade data
        """
        pass

    @abstractmethod
    def run_batch(
        self,
        strategy_class: type[StrategyT],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> list[BacktestResult]:
        """
        Run multiple backtests with different parameter combinations.

        Args:
            strategy_class: Strategy class to instantiate with each parameter set
            param_grid: Dictionary of parameter names to lists of values
            data: Historical price data
            initial_capital: Starting capital

        Returns:
            List of BacktestResult for each parameter combination
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data has required columns and format.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        required_columns = {"open", "high", "low", "close", "volume"}
        data_columns = {col.lower() for col in data.columns}

        missing = required_columns - data_columns
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")

        if data.index.duplicated().any():
            raise ValueError("Data contains duplicate timestamps")

        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted in ascending order")

    @property
    def is_running(self) -> bool:
        """Check if a backtest is currently running."""
        return self._is_running
