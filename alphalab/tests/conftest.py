import numpy as np
import pandas as pd
import pytest

from alphalab.panel import Panel, make_synthetic_panel


def tiny_panel_frames() -> dict[str, pd.DataFrame]:
    """A 6-day x 3-symbol panel with hand-pickable numbers."""
    dates = pd.bdate_range("2023-01-02", periods=6)
    symbols = ["AAA", "BBB", "CCC"]
    close = pd.DataFrame(
        [
            [10.0, 20.0, 30.0],
            [11.0, 19.0, 30.0],
            [12.0, 18.0, 31.0],
            [13.0, 17.0, 29.0],
            [14.0, 16.0, 32.0],
            [15.0, 15.0, 33.0],
        ],
        index=dates,
        columns=symbols,
    )
    open_ = close - 0.5
    high = close + 1.0
    low = open_ - 1.0
    volume = pd.DataFrame(
        np.array(
            [
                [100.0, 200.0, 300.0],
                [110.0, 190.0, 310.0],
                [120.0, 180.0, 290.0],
                [130.0, 170.0, 320.0],
                [140.0, 160.0, 330.0],
                [150.0, 150.0, 340.0],
            ]
        ),
        index=dates,
        columns=symbols,
    )
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


@pytest.fixture()
def tiny_panel() -> Panel:
    return Panel(**tiny_panel_frames())


@pytest.fixture(scope="session")
def synthetic_panel() -> Panel:
    return make_synthetic_panel(n_symbols=30, n_days=500, seed=11)
