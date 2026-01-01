"""
Drawdown Analyzer

Detailed drawdown analysis for trading strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class DrawdownPeriod:
    """Represents a single drawdown period."""
    start_date: datetime
    end_date: datetime | None
    trough_date: datetime
    peak_value: float
    trough_value: float
    recovery_date: datetime | None
    drawdown: float  # Negative value
    duration: int  # Days in drawdown
    recovery_duration: int | None  # Days to recover


class DrawdownAnalyzer:
    """
    Detailed drawdown analysis.

    Analyzes underwater periods, recovery times, and drawdown statistics.

    Example:
        >>> analyzer = DrawdownAnalyzer(equity_curve)
        >>> print(analyzer.max_drawdown())
        >>> print(analyzer.longest_drawdown())
        >>> periods = analyzer.get_drawdown_periods()
    """

    def __init__(self, equity: pd.Series):
        """
        Initialize with equity curve.

        Args:
            equity: Equity curve series with datetime index
        """
        self.equity = equity
        self._running_max = equity.expanding().max()
        self._drawdowns = equity / self._running_max - 1

    def max_drawdown(self) -> float:
        """Get maximum drawdown value."""
        return float(self._drawdowns.min())

    def average_drawdown(self) -> float:
        """Get average drawdown when underwater."""
        underwater = self._drawdowns[self._drawdowns < 0]
        if len(underwater) == 0:
            return 0.0
        return float(underwater.mean())

    def drawdown_series(self) -> pd.Series:
        """Get drawdown time series."""
        return self._drawdowns

    def get_drawdown_periods(self, min_drawdown: float = -0.01) -> list[DrawdownPeriod]:
        """
        Identify distinct drawdown periods.

        Args:
            min_drawdown: Minimum drawdown to consider (default -1%)

        Returns:
            List of DrawdownPeriod objects
        """
        periods = []
        in_drawdown = False
        start_date = None
        peak_value = 0.0
        trough_value = float("inf")
        trough_date = None

        for date, dd in self._drawdowns.items():
            if not in_drawdown:
                if dd < min_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    start_date = date
                    peak_value = self._running_max[date]
                    trough_value = self.equity[date]
                    trough_date = date
            else:
                if dd < 0:
                    # Still in drawdown
                    if self.equity[date] < trough_value:
                        trough_value = self.equity[date]
                        trough_date = date
                else:
                    # Recovered
                    period = DrawdownPeriod(
                        start_date=start_date,
                        end_date=trough_date,
                        trough_date=trough_date,
                        peak_value=peak_value,
                        trough_value=trough_value,
                        recovery_date=date,
                        drawdown=(trough_value / peak_value - 1),
                        duration=(trough_date - start_date).days if hasattr(trough_date, "days") else 0,
                        recovery_duration=(date - trough_date).days if hasattr(date, "days") else 0,
                    )
                    periods.append(period)
                    in_drawdown = False
                    trough_value = float("inf")

        # Handle ongoing drawdown
        if in_drawdown:
            period = DrawdownPeriod(
                start_date=start_date,
                end_date=None,
                trough_date=trough_date,
                peak_value=peak_value,
                trough_value=trough_value,
                recovery_date=None,
                drawdown=(trough_value / peak_value - 1),
                duration=0,
                recovery_duration=None,
            )
            periods.append(period)

        return periods

    def longest_drawdown(self) -> DrawdownPeriod | None:
        """Get the longest drawdown period by duration."""
        periods = self.get_drawdown_periods()
        if not periods:
            return None
        return max(periods, key=lambda p: p.duration)

    def deepest_drawdowns(self, n: int = 5) -> list[DrawdownPeriod]:
        """Get the N deepest drawdown periods."""
        periods = self.get_drawdown_periods()
        return sorted(periods, key=lambda p: p.drawdown)[:n]

    def time_underwater(self) -> float:
        """Calculate percentage of time spent in drawdown."""
        if len(self._drawdowns) == 0:
            return 0.0
        return float((self._drawdowns < 0).mean())

    def recovery_factor(self) -> float:
        """
        Calculate recovery factor.

        Net profit divided by maximum drawdown.
        """
        total_return = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return float(total_return / max_dd)

    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index.

        Measures downside volatility based on drawdowns.
        Lower is better.
        """
        squared_dd = self._drawdowns**2
        return float(np.sqrt(squared_dd.mean()))

    def pain_index(self) -> float:
        """
        Calculate Pain Index.

        Average drawdown over the period.
        """
        return float(abs(self._drawdowns.mean()))

    def summary(self) -> dict:
        """Generate drawdown summary."""
        periods = self.get_drawdown_periods()
        return {
            "Max Drawdown": f"{self.max_drawdown():.2%}",
            "Average Drawdown": f"{self.average_drawdown():.2%}",
            "Time Underwater": f"{self.time_underwater():.2%}",
            "Number of Drawdowns": len(periods),
            "Longest Drawdown Duration": max((p.duration for p in periods), default=0),
            "Recovery Factor": f"{self.recovery_factor():.2f}",
            "Ulcer Index": f"{self.ulcer_index():.4f}",
            "Pain Index": f"{self.pain_index():.4f}",
        }
