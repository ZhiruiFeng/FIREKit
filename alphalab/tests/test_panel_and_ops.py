"""Tests for the Panel container, synthetic generator, and operators."""

import numpy as np
import pandas as pd
import pytest
from conftest import tiny_panel_frames

from alphalab import ops
from alphalab.panel import Panel, make_synthetic_panel

# -- Panel ---------------------------------------------------------------------


def test_panel_properties(tiny_panel: Panel) -> None:
    assert tiny_panel.symbols == ["AAA", "BBB", "CCC"]
    assert tiny_panel.shape == (6, 3)
    assert isinstance(tiny_panel.dates, pd.DatetimeIndex)


def test_panel_returns_exact(tiny_panel: Panel) -> None:
    rets = tiny_panel.returns
    assert rets.iloc[1]["AAA"] == pytest.approx(0.1)  # 10 -> 11
    assert rets.iloc[1]["BBB"] == pytest.approx(-0.05)  # 20 -> 19
    assert np.isnan(rets.iloc[0]["AAA"])


def test_panel_vwap_exact(tiny_panel: Panel) -> None:
    expected = (tiny_panel.high + tiny_panel.low + tiny_panel.close) / 3.0
    pd.testing.assert_frame_equal(tiny_panel.vwap, expected)


def test_panel_misaligned_raises() -> None:
    frames = tiny_panel_frames()
    frames["volume"] = frames["volume"].iloc[:, :2]  # drop a symbol
    with pytest.raises(ValueError, match="not aligned"):
        Panel(**frames)


def test_panel_requires_datetime_index() -> None:
    frames = tiny_panel_frames()
    frames = {k: v.reset_index(drop=True) for k, v in frames.items()}
    with pytest.raises(TypeError, match="DatetimeIndex"):
        Panel(**frames)


def test_panel_from_long_roundtrip(tiny_panel: Panel) -> None:
    long_rows = []
    for field in ["open", "high", "low", "close", "volume"]:
        frame = getattr(tiny_panel, field)
        melted = frame.reset_index(names="timestamp").melt(
            id_vars="timestamp", var_name="symbol", value_name=field
        )
        long_rows.append(melted.set_index(["timestamp", "symbol"]))
    long_df = pd.concat(long_rows, axis=1).reset_index()
    rebuilt = Panel.from_long(long_df)
    pd.testing.assert_frame_equal(rebuilt.close, tiny_panel.close, check_freq=False)
    pd.testing.assert_frame_equal(rebuilt.volume, tiny_panel.volume, check_freq=False)


def test_panel_from_long_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        Panel.from_long(pd.DataFrame({"timestamp": [], "symbol": [], "close": []}))


def test_synthetic_panel_deterministic_and_shaped() -> None:
    a = make_synthetic_panel(n_symbols=5, n_days=60, seed=3)
    b = make_synthetic_panel(n_symbols=5, n_days=60, seed=3)
    c = make_synthetic_panel(n_symbols=5, n_days=60, seed=4)
    pd.testing.assert_frame_equal(a.close, b.close)
    assert not a.close.equals(c.close)
    assert a.shape == (60, 5)
    assert (a.high.to_numpy() >= a.low.to_numpy()).all()
    assert (a.close.to_numpy() > 0).all()
    assert (a.volume.to_numpy() >= 0).all()


def test_synthetic_panel_validates_args() -> None:
    with pytest.raises(ValueError):
        make_synthetic_panel(n_symbols=1)
    with pytest.raises(ValueError):
        make_synthetic_panel(n_days=10)


# -- ops -------------------------------------------------------------------------


def test_cs_rank_exact(tiny_panel: Panel) -> None:
    ranked = ops.cs_rank(tiny_panel.close)
    # first row: 10 < 20 < 30 -> ranks 1/3, 2/3, 3/3
    assert ranked.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_cs_zscore_row_properties(tiny_panel: Panel) -> None:
    z = ops.cs_zscore(tiny_panel.close)
    assert z.mean(axis=1).abs().max() < 1e-12
    assert z.std(axis=1).iloc[0] == pytest.approx(1.0)


def test_delay_and_delta_exact(tiny_panel: Panel) -> None:
    delayed = ops.delay(tiny_panel.close, 2)
    assert delayed.iloc[2]["AAA"] == 10.0
    diffed = ops.delta(tiny_panel.close, 1)
    assert diffed.iloc[1]["AAA"] == pytest.approx(1.0)
    assert diffed.iloc[3]["CCC"] == pytest.approx(-2.0)


def test_ts_aggregations_exact(tiny_panel: Panel) -> None:
    close = tiny_panel.close
    assert ops.ts_sum(close, 3).iloc[2]["AAA"] == pytest.approx(33.0)
    assert ops.ts_mean(close, 3).iloc[2]["AAA"] == pytest.approx(11.0)
    assert ops.ts_max(close, 3).iloc[4]["CCC"] == pytest.approx(32.0)
    assert ops.ts_min(close, 3).iloc[4]["CCC"] == pytest.approx(29.0)
    assert ops.ts_std(close, 3).iloc[2]["AAA"] == pytest.approx(1.0)


def test_ts_rank_exact(tiny_panel: Panel) -> None:
    tsr = ops.ts_rank(tiny_panel.close, 3)
    # AAA strictly increasing -> always highest in its window
    assert tsr.iloc[5]["AAA"] == pytest.approx(1.0)
    # BBB strictly decreasing -> always lowest
    assert tsr.iloc[5]["BBB"] == pytest.approx(1 / 3)


def test_ts_corr_perfect_correlation(tiny_panel: Panel) -> None:
    x = tiny_panel.close
    corr = ops.ts_corr(x, x * 2.0 + 1.0, 4)
    assert corr.iloc[5]["AAA"] == pytest.approx(1.0)
    anti = ops.ts_corr(x, -x, 4)
    assert anti.iloc[5]["CCC"] == pytest.approx(-1.0)


def test_clean_replaces_inf() -> None:
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf]})
    out = ops.clean(df)
    assert out["a"].isna().tolist() == [False, True, True]


def test_returns_multiday(tiny_panel: Panel) -> None:
    r5 = ops.returns(tiny_panel.close, 5)
    assert r5.iloc[5]["AAA"] == pytest.approx(0.5)  # 10 -> 15
