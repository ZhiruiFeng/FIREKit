"""
Point-in-Time Universe

Maintains historical index composition to prevent survivorship bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class UniverseMember:
    """Represents a symbol's membership in a universe."""
    symbol: str
    start_date: datetime
    end_date: datetime | None
    reason: str = ""  # e.g., "IPO", "delisted", "acquired", "bankruptcy"


class PointInTimeUniverse:
    """
    Maintains historical universe composition.

    Prevents survivorship bias by tracking which symbols were
    available at each point in time.

    Example:
        >>> universe = PointInTimeUniverse("SP500")
        >>> universe.load("sp500_historical.csv")
        >>> symbols = universe.get_universe(date=datetime(2020, 1, 15))
        >>> # Returns symbols that were in S&P 500 on that date
    """

    def __init__(self, name: str = "custom"):
        """
        Initialize universe.

        Args:
            name: Universe name (e.g., "SP500", "NASDAQ100")
        """
        self.name = name
        self._members: list[UniverseMember] = []
        self._composition: pd.DataFrame | None = None

    def add_member(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime | None = None,
        reason: str = "",
    ) -> None:
        """
        Add a member to the universe.

        Args:
            symbol: Stock symbol
            start_date: Date symbol entered universe
            end_date: Date symbol left universe (None if still active)
            reason: Reason for entry/exit
        """
        self._members.append(UniverseMember(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            reason=reason,
        ))
        self._composition = None  # Invalidate cache

    def load(self, path: str) -> None:
        """
        Load universe composition from CSV.

        Expected columns: symbol, start_date, end_date, reason

        Args:
            path: Path to CSV file
        """
        df = pd.read_csv(path, parse_dates=["start_date", "end_date"])

        self._members = []
        for _, row in df.iterrows():
            self.add_member(
                symbol=row["symbol"],
                start_date=row["start_date"],
                end_date=row["end_date"] if pd.notna(row["end_date"]) else None,
                reason=row.get("reason", ""),
            )

    def get_universe(self, date: datetime) -> list[str]:
        """
        Get symbols that were in universe at given date.

        Args:
            date: Point-in-time date

        Returns:
            List of symbols active on that date
        """
        active = []
        for member in self._members:
            if member.start_date <= date:
                if member.end_date is None or member.end_date >= date:
                    active.append(member.symbol)
        return sorted(active)

    def get_additions(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[str]:
        """Get symbols added to universe in date range."""
        additions = []
        for member in self._members:
            if start_date <= member.start_date <= end_date:
                additions.append(member.symbol)
        return additions

    def get_removals(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[str]:
        """Get symbols removed from universe in date range."""
        removals = []
        for member in self._members:
            if member.end_date and start_date <= member.end_date <= end_date:
                removals.append(member.symbol)
        return removals

    def get_changes(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, list[str]]:
        """Get all universe changes in date range."""
        return {
            "additions": self.get_additions(start_date, end_date),
            "removals": self.get_removals(start_date, end_date),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert universe to DataFrame."""
        if self._composition is None:
            self._composition = pd.DataFrame([
                {
                    "symbol": m.symbol,
                    "start_date": m.start_date,
                    "end_date": m.end_date,
                    "reason": m.reason,
                }
                for m in self._members
            ])
        return self._composition

    def __len__(self) -> int:
        """Number of members (current and historical)."""
        return len(self._members)

    def __repr__(self) -> str:
        return f"PointInTimeUniverse(name='{self.name}', members={len(self)})"
