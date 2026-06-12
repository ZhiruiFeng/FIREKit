"""Point-in-time universe membership to avoid survivorship bias.

Membership intervals are half-open on the right at the *day* level:
a symbol with ``start=2021-01-04, end=2022-06-30`` is a member on
2021-01-04 through 2022-06-29 and **not** a member on the ``end`` date.
``end=None`` means "still a member".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Membership:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp | None = None  # exclusive; None = open-ended

    def active_on(self, date: pd.Timestamp) -> bool:
        if date < self.start:
            return False
        return self.end is None or date < self.end


class PointInTimeUniverse:
    """Tracks symbols entering and leaving an index/universe over time."""

    def __init__(self) -> None:
        self._memberships: list[Membership] = []

    def add(
        self,
        symbol: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> Membership:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize() if end is not None else None
        if end_ts is not None and end_ts <= start_ts:
            raise ValueError(f"end ({end_ts.date()}) must be after start ({start_ts.date()})")
        membership = Membership(symbol.strip().upper(), start_ts, end_ts)
        self._memberships.append(membership)
        return membership

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, object]]
    ) -> PointInTimeUniverse:
        """Build from dicts with keys ``symbol``, ``start`` and optional ``end``."""
        universe = cls()
        for rec in records:
            universe.add(str(rec["symbol"]), rec["start"], rec.get("end"))  # type: ignore[arg-type]
        return universe

    @property
    def memberships(self) -> list[Membership]:
        return list(self._memberships)

    def all_symbols(self) -> list[str]:
        """Every symbol that was *ever* a member (includes dead symbols)."""
        return sorted({m.symbol for m in self._memberships})

    def members_as_of(self, date: str | pd.Timestamp) -> list[str]:
        """Sorted member symbols on the given date (point-in-time, no lookahead)."""
        ts = pd.Timestamp(date).normalize()
        return sorted({m.symbol for m in self._memberships if m.active_on(ts)})

    def membership_mask(self, dates: Iterable[str | pd.Timestamp]) -> pd.DataFrame:
        """Boolean frame (dates x all symbols ever) of point-in-time membership."""
        index = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])
        symbols = self.all_symbols()
        mask = pd.DataFrame(False, index=index, columns=symbols)
        for m in self._memberships:
            active = index >= m.start
            if m.end is not None:
                active &= index < m.end
            mask.loc[active, m.symbol] = True
        return mask

    def member_counts(self, dates: Iterable[str | pd.Timestamp]) -> pd.Series:
        """Number of active members per date."""
        return self.membership_mask(dates).sum(axis=1)

    def filter_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows of a canonical long frame where the symbol was a member."""
        if df.empty:
            return df.copy()
        mask = pd.Series(False, index=df.index)
        days = df["timestamp"].dt.normalize()
        for m in self._memberships:
            active = (df["symbol"] == m.symbol) & (days >= m.start)
            if m.end is not None:
                active &= days < m.end
            mask |= active
        return df[mask].reset_index(drop=True)
