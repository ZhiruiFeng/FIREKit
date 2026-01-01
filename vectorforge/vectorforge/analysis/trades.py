"""
Trade Analyzer

Analysis of individual trades and trading patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeStats:
    """Summary statistics for trades."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade: float
    profit_factor: float
    expectancy: float
    avg_holding_period: float


class TradeAnalyzer:
    """
    Analysis of individual trades.

    Provides detailed statistics on trading performance,
    patterns, and trade distribution.

    Example:
        >>> analyzer = TradeAnalyzer(trades_df)
        >>> stats = analyzer.get_stats()
        >>> print(analyzer.monthly_breakdown())
    """

    def __init__(self, trades: pd.DataFrame):
        """
        Initialize with trades DataFrame.

        Args:
            trades: DataFrame with columns:
                - timestamp: Trade datetime
                - symbol: Traded symbol
                - side: 'buy' or 'sell'
                - quantity: Trade size
                - price: Fill price
                - pnl: Realized P&L (optional)
                - commission: Trading cost (optional)
        """
        self.trades = trades
        if len(trades) > 0 and "pnl" in trades.columns:
            self._pnl = trades["pnl"].values
        else:
            self._pnl = np.array([])

    @property
    def total_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return int((self._pnl > 0).sum())

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return int((self._pnl < 0).sum())

    def win_rate(self) -> float:
        """Calculate win rate."""
        if len(self._pnl) == 0:
            return 0.0
        return float(self.winning_trades / len(self._pnl))

    def average_win(self) -> float:
        """Average profit on winning trades."""
        wins = self._pnl[self._pnl > 0]
        if len(wins) == 0:
            return 0.0
        return float(wins.mean())

    def average_loss(self) -> float:
        """Average loss on losing trades."""
        losses = self._pnl[self._pnl < 0]
        if len(losses) == 0:
            return 0.0
        return float(losses.mean())

    def largest_win(self) -> float:
        """Largest winning trade."""
        if len(self._pnl) == 0:
            return 0.0
        return float(self._pnl.max())

    def largest_loss(self) -> float:
        """Largest losing trade."""
        if len(self._pnl) == 0:
            return 0.0
        return float(self._pnl.min())

    def average_trade(self) -> float:
        """Average P&L per trade."""
        if len(self._pnl) == 0:
            return 0.0
        return float(self._pnl.mean())

    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses."""
        gross_profits = self._pnl[self._pnl > 0].sum()
        gross_losses = abs(self._pnl[self._pnl < 0].sum())
        if gross_losses == 0:
            return float("inf") if gross_profits > 0 else 0.0
        return float(gross_profits / gross_losses)

    def expectancy(self) -> float:
        """
        Calculate trade expectancy.

        Expected value per trade based on win rate and avg win/loss.
        """
        wr = self.win_rate()
        avg_win = self.average_win()
        avg_loss = abs(self.average_loss())
        return float(wr * avg_win - (1 - wr) * avg_loss)

    def payoff_ratio(self) -> float:
        """Ratio of average win to average loss."""
        avg_win = self.average_win()
        avg_loss = abs(self.average_loss())
        if avg_loss == 0:
            return float("inf") if avg_win > 0 else 0.0
        return float(avg_win / avg_loss)

    def sqn(self) -> float:
        """
        Calculate System Quality Number (SQN).

        Measures system quality: sqrt(n) * expectancy / stdev(R)
        """
        if len(self._pnl) < 30:
            return 0.0
        expectancy = self._pnl.mean()
        std = self._pnl.std()
        if std == 0:
            return 0.0
        return float(np.sqrt(len(self._pnl)) * expectancy / std)

    def consecutive_wins(self) -> int:
        """Maximum consecutive winning trades."""
        if len(self._pnl) == 0:
            return 0
        wins = self._pnl > 0
        max_consecutive = 0
        current = 0
        for w in wins:
            if w:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return max_consecutive

    def consecutive_losses(self) -> int:
        """Maximum consecutive losing trades."""
        if len(self._pnl) == 0:
            return 0
        losses = self._pnl < 0
        max_consecutive = 0
        current = 0
        for l in losses:
            if l:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return max_consecutive

    def get_stats(self) -> TradeStats:
        """Get comprehensive trade statistics."""
        return TradeStats(
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            win_rate=self.win_rate(),
            avg_win=self.average_win(),
            avg_loss=self.average_loss(),
            largest_win=self.largest_win(),
            largest_loss=self.largest_loss(),
            avg_trade=self.average_trade(),
            profit_factor=self.profit_factor(),
            expectancy=self.expectancy(),
            avg_holding_period=0.0,  # Requires holding period data
        )

    def monthly_breakdown(self) -> pd.DataFrame:
        """Break down trades by month."""
        if len(self.trades) == 0 or "timestamp" not in self.trades.columns:
            return pd.DataFrame()

        df = self.trades.copy()
        df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")

        grouped = df.groupby("month").agg(
            trades=("price", "count"),
            total_pnl=("pnl", "sum") if "pnl" in df.columns else ("price", "count"),
            avg_pnl=("pnl", "mean") if "pnl" in df.columns else ("price", "count"),
        )

        return grouped

    def symbol_breakdown(self) -> pd.DataFrame:
        """Break down trades by symbol."""
        if len(self.trades) == 0 or "symbol" not in self.trades.columns:
            return pd.DataFrame()

        df = self.trades.copy()
        grouped = df.groupby("symbol").agg(
            trades=("price", "count"),
            total_pnl=("pnl", "sum") if "pnl" in df.columns else ("price", "count"),
            avg_pnl=("pnl", "mean") if "pnl" in df.columns else ("price", "count"),
        )

        return grouped.sort_values("total_pnl", ascending=False)

    def summary(self) -> dict[str, Any]:
        """Generate trade summary."""
        return {
            "Total Trades": self.total_trades,
            "Winning Trades": self.winning_trades,
            "Losing Trades": self.losing_trades,
            "Win Rate": f"{self.win_rate():.2%}",
            "Average Win": f"${self.average_win():.2f}",
            "Average Loss": f"${self.average_loss():.2f}",
            "Largest Win": f"${self.largest_win():.2f}",
            "Largest Loss": f"${self.largest_loss():.2f}",
            "Average Trade": f"${self.average_trade():.2f}",
            "Profit Factor": f"{self.profit_factor():.2f}",
            "Payoff Ratio": f"{self.payoff_ratio():.2f}",
            "Expectancy": f"${self.expectancy():.2f}",
            "SQN": f"{self.sqn():.2f}",
            "Max Consecutive Wins": self.consecutive_wins(),
            "Max Consecutive Losses": self.consecutive_losses(),
        }
