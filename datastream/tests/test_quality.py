"""Tests for the data quality engine (exact issue counts on hand-crafted frames)."""

import numpy as np
import pandas as pd
import pytest

from datastream.quality import IssueType, QualityEngine
from datastream.sources import SyntheticSource


def make_frame(n: int = 30, symbol: str = "TST") -> pd.DataFrame:
    """A perfectly clean frame of n business-day bars with constant-ish prices."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100.0 + 0.1 * np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "open": close - 0.05,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def issue_count(report, issue_type: IssueType) -> int:
    return report.issue_counts.get(issue_type.value, 0)


def test_clean_frame_scores_100() -> None:
    clean, report = QualityEngine().check_symbol(make_frame())
    assert report.issues == []
    assert report.score == 100.0
    assert len(clean) == 30


def test_duplicate_timestamps_detected_and_repaired() -> None:
    df = make_frame()
    df = pd.concat([df, df.iloc[[3, 7]]], ignore_index=True)
    clean, report = QualityEngine().check_symbol(df)
    assert issue_count(report, IssueType.DUPLICATE_TIMESTAMP) == 2
    assert report.n_repaired >= 2
    assert len(clean) == 30
    assert not clean["timestamp"].duplicated().any()


def test_missing_values_dropped() -> None:
    df = make_frame()
    df.loc[5, "close"] = np.nan
    clean, report = QualityEngine().check_symbol(df)
    assert issue_count(report, IssueType.MISSING_VALUE) == 1
    assert len(clean) == 29
    assert not clean[["open", "high", "low", "close", "volume"]].isna().any().any()


def test_nonpositive_prices_dropped() -> None:
    df = make_frame()
    df.loc[4, "close"] = -5.0
    df.loc[9, "open"] = 0.0
    clean, report = QualityEngine().check_symbol(df)
    assert issue_count(report, IssueType.NONPOSITIVE_PRICE) == 2
    assert (clean[["open", "high", "low", "close"]] > 0).all().all()
    assert len(clean) == 28


def test_negative_volume_clamped() -> None:
    df = make_frame()
    df.loc[6, "volume"] = -100.0
    clean, report = QualityEngine().check_symbol(df)
    assert issue_count(report, IssueType.NEGATIVE_VOLUME) == 1
    assert clean.loc[6, "volume"] == 0.0
    assert len(clean) == 30


def test_ohlc_inconsistency_clamped_exact() -> None:
    df = make_frame()
    df.loc[5, "high"] = df.loc[5, "close"] - 2.0  # high below body
    df.loc[8, "low"] = df.loc[8, "open"] + 3.0  # low above body
    clean, report = QualityEngine().check_symbol(df)
    assert issue_count(report, IssueType.OHLC_INCONSISTENT) == 2
    body_high = max(clean.loc[5, "open"], clean.loc[5, "close"])
    body_low = min(clean.loc[8, "open"], clean.loc[8, "close"])
    assert clean.loc[5, "high"] == body_high
    assert clean.loc[8, "low"] == body_low
    assert (clean["high"] >= clean["low"]).all()


def test_return_outlier_flagged_not_repaired() -> None:
    df = make_frame(60)
    df.loc[30, "close"] = df.loc[30, "close"] * 5  # massive one-day jump
    df.loc[30, "high"] = df.loc[30, "close"] + 1
    clean, report = QualityEngine(outlier_z=4.0).check_symbol(df)
    outliers = [i for i in report.issues if i.issue_type is IssueType.RETURN_OUTLIER]
    assert len(outliers) >= 1
    assert all(not i.repaired for i in outliers)
    assert len(clean) == 60  # outliers are kept


def test_gap_detection_exact() -> None:
    df = make_frame(30)
    df = df.drop(index=[10, 11, 12, 13]).reset_index(drop=True)  # 4 missing bdays
    _, report = QualityEngine(max_gap_bdays=3).check_symbol(df)
    gaps = [i for i in report.issues if i.issue_type is IssueType.GAP]
    assert len(gaps) == 1
    assert gaps[0].timestamp == pd.Timestamp("2023-01-02") + pd.offsets.BDay(14)
    assert "4 missing business days" in gaps[0].detail


def test_weekend_is_not_a_gap() -> None:
    _, report = QualityEngine(max_gap_bdays=1).check_symbol(make_frame(30))
    assert issue_count(report, IssueType.GAP) == 0


def test_score_exact_value() -> None:
    df = make_frame(20)
    df.loc[3, "close"] = -1.0  # exactly one issue in 20 rows
    _, report = QualityEngine().check_symbol(df)
    assert len(report.issues) == 1
    assert report.score == pytest.approx(95.0)


def test_no_repair_mode_keeps_rows() -> None:
    df = make_frame()
    df.loc[4, "close"] = -5.0
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    clean, report = QualityEngine(repair=False).check_symbol(df)
    assert len(clean) == 31
    assert report.n_repaired == 0
    assert len(report.issues) >= 2


def test_run_multi_symbol_aggregate() -> None:
    a = make_frame(20, "AAA")
    b = make_frame(20, "BBB")
    b.loc[5, "close"] = -1.0
    clean, agg = QualityEngine().run(pd.concat([a, b], ignore_index=True))
    assert set(agg.per_symbol) == {"AAA", "BBB"}
    assert agg.per_symbol["AAA"].score == 100.0
    assert agg.per_symbol["BBB"].score == pytest.approx(95.0)
    assert agg.score == pytest.approx(97.5)  # row-weighted, equal rows
    assert agg.n_rows == 40
    assert agg.n_issues == 1
    assert set(clean["symbol"]) == {"AAA", "BBB"}


def test_engine_finds_injected_synthetic_issues() -> None:
    src = SyntheticSource(n_symbols=5, start="2021-01-04", end="2023-12-29", seed=99,
                          issue_rate=0.01)
    clean, report = QualityEngine().run(src.fetch())
    assert report.n_issues > 0
    assert report.n_repaired > 0
    assert report.score < 100.0
    # cleaned data must satisfy hard invariants
    assert (clean[["open", "high", "low", "close"]] > 0).all().all()
    assert (clean["high"] >= clean["low"]).all()
    assert not clean.duplicated(subset=["symbol", "timestamp"]).any()


def test_engine_validates_params() -> None:
    with pytest.raises(ValueError):
        QualityEngine(outlier_z=0)
    with pytest.raises(ValueError):
        QualityEngine(max_gap_bdays=0)


def test_check_symbol_rejects_mixed_symbols() -> None:
    df = pd.concat([make_frame(10, "AAA"), make_frame(10, "BBB")], ignore_index=True)
    with pytest.raises(ValueError, match="multiple symbols"):
        QualityEngine().check_symbol(df)
