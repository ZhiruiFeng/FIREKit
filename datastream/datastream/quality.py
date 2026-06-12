"""Data quality engine: detection, optional repair, and per-symbol/aggregate scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from datastream.schema import normalize_frame


class IssueType(StrEnum):
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    MISSING_VALUE = "missing_value"
    NONPOSITIVE_PRICE = "nonpositive_price"
    NEGATIVE_VOLUME = "negative_volume"
    OHLC_INCONSISTENT = "ohlc_inconsistent"
    RETURN_OUTLIER = "return_outlier"
    GAP = "gap"


#: Issues the engine can repair in-place; the rest are flag-only.
REPAIRABLE: frozenset[IssueType] = frozenset(
    {
        IssueType.DUPLICATE_TIMESTAMP,
        IssueType.MISSING_VALUE,
        IssueType.NONPOSITIVE_PRICE,
        IssueType.NEGATIVE_VOLUME,
        IssueType.OHLC_INCONSISTENT,
    }
)


@dataclass(frozen=True)
class QualityIssue:
    symbol: str
    timestamp: pd.Timestamp
    issue_type: IssueType
    detail: str
    repaired: bool = False


@dataclass
class SymbolQualityReport:
    symbol: str
    n_rows: int
    n_rows_clean: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.issue_type.value] = counts.get(issue.issue_type.value, 0) + 1
        return counts

    @property
    def n_repaired(self) -> int:
        return sum(1 for issue in self.issues if issue.repaired)

    @property
    def score(self) -> float:
        """Quality score in [0, 100]: share of rows untouched by any issue."""
        if self.n_rows == 0:
            return 0.0
        return max(0.0, 100.0 * (1.0 - len(self.issues) / self.n_rows))


@dataclass
class AggregateQualityReport:
    per_symbol: dict[str, SymbolQualityReport]

    @property
    def n_rows(self) -> int:
        return sum(r.n_rows for r in self.per_symbol.values())

    @property
    def n_issues(self) -> int:
        return sum(len(r.issues) for r in self.per_symbol.values())

    @property
    def n_repaired(self) -> int:
        return sum(r.n_repaired for r in self.per_symbol.values())

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for report in self.per_symbol.values():
            for key, value in report.issue_counts.items():
                counts[key] = counts.get(key, 0) + value
        return counts

    @property
    def score(self) -> float:
        """Row-weighted average of per-symbol scores."""
        total = self.n_rows
        if total == 0:
            return 0.0
        return sum(r.score * r.n_rows for r in self.per_symbol.values()) / total

    def worst_symbols(self, n: int = 10) -> list[SymbolQualityReport]:
        return sorted(self.per_symbol.values(), key=lambda r: r.score)[:n]


class QualityEngine:
    """Detects (and optionally repairs) data-quality problems in canonical frames.

    Checks, in order:

    1. duplicate timestamps (repair: keep first occurrence)
    2. missing OHLCV values (repair: drop row)
    3. non-positive prices (repair: drop row)
    4. negative volume (repair: clamp to 0)
    5. OHLC consistency: ``high >= max(open, close)``, ``low <= min(open, close)``,
       ``high >= low`` (repair: clamp high/low)
    6. return outliers: |z-score| of close-to-close returns above ``outlier_z``
       (flag only — could be a genuine move)
    7. calendar gaps: more than ``max_gap_bdays`` business days between
       consecutive bars (flag only)
    """

    def __init__(
        self,
        outlier_z: float = 6.0,
        max_gap_bdays: int = 3,
        repair: bool = True,
    ) -> None:
        if outlier_z <= 0:
            raise ValueError("outlier_z must be positive")
        if max_gap_bdays < 1:
            raise ValueError("max_gap_bdays must be >= 1")
        self.outlier_z = outlier_z
        self.max_gap_bdays = max_gap_bdays
        self.repair = repair

    # -- public API -------------------------------------------------------

    def check_symbol(
        self, df: pd.DataFrame, symbol: str | None = None
    ) -> tuple[pd.DataFrame, SymbolQualityReport]:
        """Check one symbol's bars; returns ``(clean_frame, report)``."""
        df = normalize_frame(df, symbol=symbol)
        symbols = df["symbol"].unique()
        if len(symbols) > 1:
            raise ValueError(f"check_symbol got multiple symbols: {sorted(symbols)}")
        sym = symbol.upper() if symbol else (str(symbols[0]) if len(symbols) else "?")
        n_rows = len(df)
        issues: list[QualityIssue] = []

        df = self._check_duplicates(df, sym, issues)
        df = self._check_missing(df, sym, issues)
        df = self._check_nonpositive(df, sym, issues)
        df = self._check_volume(df, sym, issues)
        df = self._check_ohlc(df, sym, issues)
        self._check_outliers(df, sym, issues)
        self._check_gaps(df, sym, issues)

        report = SymbolQualityReport(symbol=sym, n_rows=n_rows, n_rows_clean=len(df), issues=issues)
        return df.reset_index(drop=True), report

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, AggregateQualityReport]:
        """Check a multi-symbol long frame; returns ``(clean_frame, aggregate_report)``."""
        df = normalize_frame(df)
        clean_frames: list[pd.DataFrame] = []
        reports: dict[str, SymbolQualityReport] = {}
        for sym, group in df.groupby("symbol", sort=True):
            clean, report = self.check_symbol(group.reset_index(drop=True))
            clean_frames.append(clean)
            reports[str(sym)] = report
        clean_df = (
            pd.concat(clean_frames, ignore_index=True) if clean_frames else df.iloc[0:0].copy()
        )
        return clean_df, AggregateQualityReport(per_symbol=reports)

    # -- individual checks --------------------------------------------------

    def _check_duplicates(
        self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]
    ) -> pd.DataFrame:
        dup_mask = df["timestamp"].duplicated(keep="first")
        for ts in df.loc[dup_mask, "timestamp"]:
            issues.append(
                QualityIssue(sym, ts, IssueType.DUPLICATE_TIMESTAMP, "duplicate bar", self.repair)
            )
        if self.repair and dup_mask.any():
            df = df[~dup_mask]
        return df

    def _check_missing(self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]) -> pd.DataFrame:
        na_mask = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
        for ts in df.loc[na_mask, "timestamp"]:
            issues.append(
                QualityIssue(sym, ts, IssueType.MISSING_VALUE, "NaN in OHLCV", self.repair)
            )
        if self.repair and na_mask.any():
            df = df[~na_mask]
        return df

    def _check_nonpositive(
        self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]
    ) -> pd.DataFrame:
        bad_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        for ts in df.loc[bad_mask, "timestamp"]:
            issues.append(
                QualityIssue(sym, ts, IssueType.NONPOSITIVE_PRICE, "price <= 0", self.repair)
            )
        if self.repair and bad_mask.any():
            df = df[~bad_mask]
        return df

    def _check_volume(self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]) -> pd.DataFrame:
        neg_mask = df["volume"] < 0
        for ts in df.loc[neg_mask, "timestamp"]:
            issues.append(
                QualityIssue(sym, ts, IssueType.NEGATIVE_VOLUME, "volume < 0", self.repair)
            )
        if self.repair and neg_mask.any():
            df = df.copy()
            df.loc[neg_mask, "volume"] = 0.0
        return df

    def _check_ohlc(self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]) -> pd.DataFrame:
        body_high = df[["open", "close"]].max(axis=1)
        body_low = df[["open", "close"]].min(axis=1)
        bad_mask = (df["high"] < body_high) | (df["low"] > body_low) | (df["high"] < df["low"])
        for ts in df.loc[bad_mask, "timestamp"]:
            issues.append(
                QualityIssue(
                    sym, ts, IssueType.OHLC_INCONSISTENT, "high/low violate OHLC bounds", self.repair
                )
            )
        if self.repair and bad_mask.any():
            df = df.copy()
            df.loc[bad_mask, "high"] = np.maximum(df.loc[bad_mask, "high"], body_high[bad_mask])
            df.loc[bad_mask, "low"] = np.minimum(df.loc[bad_mask, "low"], body_low[bad_mask])
        return df

    def _check_outliers(self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]) -> None:
        if len(df) < 20:
            return
        returns = df["close"].pct_change()
        std = returns.std()
        if std is np.nan or not np.isfinite(std) or std == 0:
            return
        z = (returns - returns.mean()) / std
        out_mask = z.abs() > self.outlier_z
        for ts, zval in zip(df.loc[out_mask, "timestamp"], z[out_mask], strict=True):
            issues.append(
                QualityIssue(sym, ts, IssueType.RETURN_OUTLIER, f"return z-score {zval:.1f}", False)
            )

    def _check_gaps(self, df: pd.DataFrame, sym: str, issues: list[QualityIssue]) -> None:
        if len(df) < 2:
            return
        ts = pd.DatetimeIndex(df["timestamp"])
        # Business days strictly between consecutive bars.
        gaps = np.busday_count(
            ts[:-1].values.astype("datetime64[D]") + 1, ts[1:].values.astype("datetime64[D]")
        )
        for i in np.nonzero(gaps >= self.max_gap_bdays)[0]:
            issues.append(
                QualityIssue(
                    sym,
                    ts[i + 1],
                    IssueType.GAP,
                    f"{int(gaps[i])} missing business days before this bar",
                    False,
                )
            )
