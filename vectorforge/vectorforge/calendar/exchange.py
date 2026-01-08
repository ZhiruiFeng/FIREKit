"""
Exchange Calendar Implementation

Provides trading calendars for major exchanges with:
- Regular trading hours
- Pre-market and after-hours sessions
- Market holidays
- Early close days
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

import pandas as pd


class SessionType(Enum):
    """Types of trading sessions."""

    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


@dataclass
class TradingSession:
    """Represents a trading session."""

    session_type: SessionType
    start: time
    end: time
    timezone: str = "America/New_York"

    def contains(self, t: time) -> bool:
        """Check if a time falls within this session."""
        return self.start <= t < self.end

    def to_datetime(self, dt: date) -> tuple[datetime, datetime]:
        """Convert session times to datetime for a given date."""
        tz = ZoneInfo(self.timezone)
        start_dt = datetime.combine(dt, self.start, tzinfo=tz)
        end_dt = datetime.combine(dt, self.end, tzinfo=tz)
        return start_dt, end_dt


@dataclass
class MarketHoliday:
    """Represents a market holiday."""

    date: date
    name: str
    early_close: bool = False
    early_close_time: time | None = None


@dataclass
class ExchangeCalendar:
    """
    Exchange calendar with trading hours and holidays.

    Supports:
    - Regular market hours
    - Pre-market and after-hours sessions
    - Market holidays
    - Early close days
    - Timezone-aware scheduling
    """

    name: str
    timezone: str
    regular_open: time
    regular_close: time
    pre_market_open: time | None = None
    after_hours_close: time | None = None
    holidays: list[MarketHoliday] = field(default_factory=list)
    weekends_closed: bool = True

    # Cached holiday dates for fast lookup
    _holiday_dates: set[date] = field(default_factory=set, init=False, repr=False)
    _early_close_dates: dict[date, time] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Build holiday lookup caches."""
        self._holiday_dates = {h.date for h in self.holidays if not h.early_close}
        self._early_close_dates = {
            h.date: h.early_close_time
            for h in self.holidays
            if h.early_close and h.early_close_time
        }

    def is_open(self, dt: datetime) -> bool:
        """Check if the exchange is open at the given datetime."""
        d = dt.date()

        # Check weekends
        if self.weekends_closed and dt.weekday() >= 5:
            return False

        # Check holidays
        if d in self._holiday_dates:
            return False

        # Check early close
        close_time = self._early_close_dates.get(d, self.regular_close)

        t = dt.time()
        return self.regular_open <= t < close_time

    def is_pre_market(self, dt: datetime) -> bool:
        """Check if it's pre-market hours."""
        if not self.pre_market_open:
            return False

        d = dt.date()
        if self.weekends_closed and dt.weekday() >= 5:
            return False
        if d in self._holiday_dates:
            return False

        t = dt.time()
        return self.pre_market_open <= t < self.regular_open

    def is_after_hours(self, dt: datetime) -> bool:
        """Check if it's after-hours session."""
        if not self.after_hours_close:
            return False

        d = dt.date()
        if self.weekends_closed and dt.weekday() >= 5:
            return False
        if d in self._holiday_dates:
            return False

        close_time = self._early_close_dates.get(d, self.regular_close)
        t = dt.time()
        return close_time <= t < self.after_hours_close

    def get_session(self, dt: datetime) -> SessionType:
        """Get the current trading session type."""
        if self.is_open(dt):
            return SessionType.REGULAR
        elif self.is_pre_market(dt):
            return SessionType.PRE_MARKET
        elif self.is_after_hours(dt):
            return SessionType.AFTER_HOURS
        return SessionType.CLOSED

    def next_open(self, dt: datetime) -> datetime:
        """Get the next market open time after the given datetime."""
        tz = ZoneInfo(self.timezone)
        current = dt if dt.tzinfo else dt.replace(tzinfo=tz)

        # If currently during market hours, return as is
        if self.is_open(current):
            return current

        # Check if we can open later today
        today = current.date()
        if (not self.weekends_closed or current.weekday() < 5) and today not in self._holiday_dates:
            today_open = datetime.combine(today, self.regular_open, tzinfo=tz)
            if current < today_open:
                return today_open

        # Find next trading day
        next_day = today + timedelta(days=1)
        while True:
            if self.weekends_closed and next_day.weekday() >= 5:
                next_day += timedelta(days=1)
                continue
            if next_day in self._holiday_dates:
                next_day += timedelta(days=1)
                continue
            break

        return datetime.combine(next_day, self.regular_open, tzinfo=tz)

    def next_close(self, dt: datetime) -> datetime:
        """Get the next market close time after the given datetime."""
        tz = ZoneInfo(self.timezone)
        current = dt if dt.tzinfo else dt.replace(tzinfo=tz)

        today = current.date()

        # Check if market is open today
        if (not self.weekends_closed or current.weekday() < 5) and today not in self._holiday_dates:
            close_time = self._early_close_dates.get(today, self.regular_close)
            today_close = datetime.combine(today, close_time, tzinfo=tz)
            if current < today_close:
                return today_close

        # Find next trading day's close
        next_day = today + timedelta(days=1)
        while True:
            if self.weekends_closed and next_day.weekday() >= 5:
                next_day += timedelta(days=1)
                continue
            if next_day in self._holiday_dates:
                next_day += timedelta(days=1)
                continue
            break

        close_time = self._early_close_dates.get(next_day, self.regular_close)
        return datetime.combine(next_day, close_time, tzinfo=tz)

    def trading_days(
        self,
        start: date,
        end: date,
    ) -> Iterator[date]:
        """Iterate over trading days in a date range."""
        current = start
        while current <= end:
            if self.weekends_closed and current.weekday() >= 5:
                current += timedelta(days=1)
                continue
            if current in self._holiday_dates:
                current += timedelta(days=1)
                continue
            yield current
            current += timedelta(days=1)

    def trading_days_count(self, start: date, end: date) -> int:
        """Count trading days in a date range."""
        return sum(1 for _ in self.trading_days(start, end))

    def get_sessions(self, d: date) -> list[TradingSession]:
        """Get all trading sessions for a date."""
        sessions = []

        if self.weekends_closed and d.weekday() >= 5:
            return sessions
        if d in self._holiday_dates:
            return sessions

        if self.pre_market_open:
            sessions.append(
                TradingSession(
                    session_type=SessionType.PRE_MARKET,
                    start=self.pre_market_open,
                    end=self.regular_open,
                    timezone=self.timezone,
                )
            )

        close_time = self._early_close_dates.get(d, self.regular_close)
        sessions.append(
            TradingSession(
                session_type=SessionType.REGULAR,
                start=self.regular_open,
                end=close_time,
                timezone=self.timezone,
            )
        )

        if self.after_hours_close and d not in self._early_close_dates:
            sessions.append(
                TradingSession(
                    session_type=SessionType.AFTER_HOURS,
                    start=close_time,
                    end=self.after_hours_close,
                    timezone=self.timezone,
                )
            )

        return sessions

    def to_dataframe(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Generate a DataFrame of trading sessions."""
        records = []
        for d in self.trading_days(start, end):
            for session in self.get_sessions(d):
                start_dt, end_dt = session.to_datetime(d)
                records.append(
                    {
                        "date": d,
                        "session": session.session_type.value,
                        "open": start_dt,
                        "close": end_dt,
                    }
                )
        return pd.DataFrame(records)


# ============================================================================
# Pre-defined Calendars
# ============================================================================


def _us_holidays_2024_2025() -> list[MarketHoliday]:
    """US market holidays for 2024-2025."""
    return [
        # 2024
        MarketHoliday(date(2024, 1, 1), "New Year's Day"),
        MarketHoliday(date(2024, 1, 15), "MLK Day"),
        MarketHoliday(date(2024, 2, 19), "Presidents Day"),
        MarketHoliday(date(2024, 3, 29), "Good Friday"),
        MarketHoliday(date(2024, 5, 27), "Memorial Day"),
        MarketHoliday(date(2024, 6, 19), "Juneteenth"),
        MarketHoliday(date(2024, 7, 4), "Independence Day"),
        MarketHoliday(
            date(2024, 7, 3),
            "Independence Day (early)",
            early_close=True,
            early_close_time=time(13, 0),
        ),
        MarketHoliday(date(2024, 9, 2), "Labor Day"),
        MarketHoliday(date(2024, 11, 28), "Thanksgiving"),
        MarketHoliday(
            date(2024, 11, 29),
            "Day after Thanksgiving",
            early_close=True,
            early_close_time=time(13, 0),
        ),
        MarketHoliday(date(2024, 12, 25), "Christmas"),
        MarketHoliday(
            date(2024, 12, 24), "Christmas Eve", early_close=True, early_close_time=time(13, 0)
        ),
        # 2025
        MarketHoliday(date(2025, 1, 1), "New Year's Day"),
        MarketHoliday(date(2025, 1, 20), "MLK Day"),
        MarketHoliday(date(2025, 2, 17), "Presidents Day"),
        MarketHoliday(date(2025, 4, 18), "Good Friday"),
        MarketHoliday(date(2025, 5, 26), "Memorial Day"),
        MarketHoliday(date(2025, 6, 19), "Juneteenth"),
        MarketHoliday(date(2025, 7, 4), "Independence Day"),
        MarketHoliday(
            date(2025, 7, 3),
            "Independence Day (early)",
            early_close=True,
            early_close_time=time(13, 0),
        ),
        MarketHoliday(date(2025, 9, 1), "Labor Day"),
        MarketHoliday(date(2025, 11, 27), "Thanksgiving"),
        MarketHoliday(
            date(2025, 11, 28),
            "Day after Thanksgiving",
            early_close=True,
            early_close_time=time(13, 0),
        ),
        MarketHoliday(date(2025, 12, 25), "Christmas"),
        MarketHoliday(
            date(2025, 12, 24), "Christmas Eve", early_close=True, early_close_time=time(13, 0)
        ),
    ]


# Pre-defined calendars
CALENDARS: dict[str, ExchangeCalendar] = {
    "NYSE": ExchangeCalendar(
        name="New York Stock Exchange",
        timezone="America/New_York",
        regular_open=time(9, 30),
        regular_close=time(16, 0),
        pre_market_open=time(4, 0),
        after_hours_close=time(20, 0),
        holidays=_us_holidays_2024_2025(),
    ),
    "NASDAQ": ExchangeCalendar(
        name="NASDAQ",
        timezone="America/New_York",
        regular_open=time(9, 30),
        regular_close=time(16, 0),
        pre_market_open=time(4, 0),
        after_hours_close=time(20, 0),
        holidays=_us_holidays_2024_2025(),
    ),
    "CME": ExchangeCalendar(
        name="Chicago Mercantile Exchange",
        timezone="America/Chicago",
        regular_open=time(8, 30),
        regular_close=time(15, 15),
        holidays=_us_holidays_2024_2025(),
    ),
    "LSE": ExchangeCalendar(
        name="London Stock Exchange",
        timezone="Europe/London",
        regular_open=time(8, 0),
        regular_close=time(16, 30),
        holidays=[
            MarketHoliday(date(2024, 1, 1), "New Year's Day"),
            MarketHoliday(date(2024, 3, 29), "Good Friday"),
            MarketHoliday(date(2024, 4, 1), "Easter Monday"),
            MarketHoliday(date(2024, 5, 6), "Early May Bank Holiday"),
            MarketHoliday(date(2024, 5, 27), "Spring Bank Holiday"),
            MarketHoliday(date(2024, 8, 26), "Summer Bank Holiday"),
            MarketHoliday(date(2024, 12, 25), "Christmas"),
            MarketHoliday(date(2024, 12, 26), "Boxing Day"),
        ],
    ),
    "TSE": ExchangeCalendar(
        name="Tokyo Stock Exchange",
        timezone="Asia/Tokyo",
        regular_open=time(9, 0),
        regular_close=time(15, 0),
        holidays=[
            MarketHoliday(date(2024, 1, 1), "New Year's Day"),
            MarketHoliday(date(2024, 1, 2), "New Year Holiday"),
            MarketHoliday(date(2024, 1, 3), "New Year Holiday"),
            MarketHoliday(date(2024, 1, 8), "Coming of Age Day"),
            MarketHoliday(date(2024, 2, 11), "National Foundation Day"),
            MarketHoliday(date(2024, 2, 23), "Emperor's Birthday"),
            MarketHoliday(date(2024, 3, 20), "Vernal Equinox Day"),
        ],
    ),
    "CRYPTO": ExchangeCalendar(
        name="Cryptocurrency (24/7)",
        timezone="UTC",
        regular_open=time(0, 0),
        regular_close=time(23, 59, 59),
        weekends_closed=False,
        holidays=[],
    ),
}


def get_calendar(name: str) -> ExchangeCalendar:
    """
    Get an exchange calendar by name.

    Args:
        name: Exchange name (NYSE, NASDAQ, CME, LSE, TSE, CRYPTO)

    Returns:
        ExchangeCalendar instance

    Raises:
        KeyError: If calendar not found
    """
    name_upper = name.upper()
    if name_upper not in CALENDARS:
        available = ", ".join(CALENDARS.keys())
        raise KeyError(f"Calendar '{name}' not found. Available: {available}")
    return CALENDARS[name_upper]


def list_calendars() -> list[str]:
    """List available calendar names."""
    return list(CALENDARS.keys())
