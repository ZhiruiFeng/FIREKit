"""
Performance Metrics

Comprehensive strategy performance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """Risk-related metrics."""
    volatility: float  # Annualized volatility
    downside_volatility: float  # Downside deviation
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    skewness: float
    kurtosis: float


@dataclass
class ReturnMetrics:
    """Return-related metrics."""
    total_return: float
    annual_return: float
    monthly_return: float
    daily_mean: float
    best_day: float
    worst_day: float


class PerformanceMetrics:
    """
    Comprehensive strategy performance analysis.

    Calculates risk-adjusted returns, drawdown statistics,
    and trade analytics from backtest results.

    Example:
        >>> metrics = PerformanceMetrics(results.returns)
        >>> print(metrics.sharpe_ratio())
        >>> print(metrics.generate_report())
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """
        Initialize with returns series.

        Args:
            returns: Daily returns series
            benchmark: Optional benchmark returns for relative metrics
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Trading days per year (default 252)
        """
        self.returns = returns.dropna()
        self.benchmark = benchmark.dropna() if benchmark is not None else None
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._daily_rf = risk_free_rate / periods_per_year

    def total_return(self) -> float:
        """Calculate total cumulative return."""
        return float((1 + self.returns).prod() - 1)

    def annual_return(self) -> float:
        """Calculate annualized return."""
        total = self.total_return()
        n_years = len(self.returns) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return float((1 + total) ** (1 / n_years) - 1)

    def volatility(self) -> float:
        """Calculate annualized volatility."""
        return float(self.returns.std() * np.sqrt(self.periods_per_year))

    def downside_volatility(self, threshold: float = 0.0) -> float:
        """Calculate downside deviation."""
        downside = self.returns[self.returns < threshold]
        if len(downside) == 0:
            return 0.0
        return float(downside.std() * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio.

        Returns:
            Risk-adjusted return measure
        """
        excess = self.returns - self._daily_rf
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.periods_per_year))

    def sortino_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Sortino ratio.

        Uses downside deviation instead of total volatility.

        Args:
            threshold: Minimum acceptable return (default 0)

        Returns:
            Downside risk-adjusted return
        """
        excess = self.returns - self._daily_rf
        downside = excess[excess < threshold]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(self.periods_per_year))

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.

        Annualized return divided by maximum drawdown.

        Returns:
            Return per unit of drawdown risk
        """
        ann_ret = self.annual_return()
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return float(ann_ret / max_dd)

    def information_ratio(self) -> float:
        """
        Calculate Information ratio vs. benchmark.

        Returns:
            Active return per unit of tracking error
        """
        if self.benchmark is None:
            raise ValueError("Benchmark required for Information Ratio")

        # Align returns
        aligned = pd.concat([self.returns, self.benchmark], axis=1, join="inner")
        active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]

        if active_returns.std() == 0:
            return 0.0
        return float(active_returns.mean() / active_returns.std() * np.sqrt(self.periods_per_year))

    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Maximum peak-to-trough decline (negative value)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        return float(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        """
        Calculate drawdown series.

        Returns:
            Series of drawdown values at each point
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        return cumulative / running_max - 1

    def var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.

        Args:
            confidence: Confidence level (default 0.95)

        Returns:
            VaR at specified confidence level
        """
        return float(np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            confidence: Confidence level (default 0.95)

        Returns:
            Expected loss beyond VaR
        """
        var = self.var(confidence)
        return float(self.returns[self.returns <= var].mean())

    def tail_ratio(self) -> float:
        """
        Calculate tail ratio.

        Ratio of right tail (95th percentile) to left tail (5th percentile).

        Returns:
            Tail ratio (>1 means positive skew in returns)
        """
        right_tail = np.percentile(self.returns, 95)
        left_tail = np.percentile(self.returns, 5)
        if left_tail == 0:
            return 0.0
        return float(abs(right_tail / left_tail))

    def win_rate(self) -> float:
        """Calculate percentage of positive return days."""
        if len(self.returns) == 0:
            return 0.0
        return float((self.returns > 0).mean())

    def profit_factor(self) -> float:
        """
        Calculate profit factor.

        Ratio of gross profits to gross losses.

        Returns:
            Profit factor (>1 is profitable)
        """
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Probability-weighted ratio of gains vs losses.

        Args:
            threshold: Target return threshold

        Returns:
            Omega ratio
        """
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = (threshold - self.returns).clip(lower=0).sum()
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def skewness(self) -> float:
        """Calculate return distribution skewness."""
        return float(self.returns.skew())

    def kurtosis(self) -> float:
        """Calculate return distribution kurtosis."""
        return float(self.returns.kurtosis())

    def generate_report(self) -> dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary with formatted metrics
        """
        return {
            # Returns
            "Total Return": f"{self.total_return():.2%}",
            "Annual Return": f"{self.annual_return():.2%}",
            "Annual Volatility": f"{self.volatility():.2%}",
            # Risk-adjusted
            "Sharpe Ratio": f"{self.sharpe_ratio():.2f}",
            "Sortino Ratio": f"{self.sortino_ratio():.2f}",
            "Calmar Ratio": f"{self.calmar_ratio():.2f}",
            # Risk
            "Max Drawdown": f"{self.max_drawdown():.2%}",
            "VaR (95%)": f"{self.var():.2%}",
            "CVaR (95%)": f"{self.cvar():.2%}",
            # Distribution
            "Skewness": f"{self.skewness():.2f}",
            "Kurtosis": f"{self.kurtosis():.2f}",
            "Tail Ratio": f"{self.tail_ratio():.2f}",
            # Trading
            "Win Rate": f"{self.win_rate():.2%}",
            "Profit Factor": f"{self.profit_factor():.2f}",
            "Omega Ratio": f"{self.omega_ratio():.2f}",
            # Meta
            "Trading Days": len(self.returns),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to DataFrame."""
        report = self.generate_report()
        return pd.DataFrame.from_dict(report, orient="index", columns=["Value"])
