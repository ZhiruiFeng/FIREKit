"""Volatility targeting: scale exposure to hit an annualized vol target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def realized_vol(
    returns: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling realized volatility, annualized (sample std, ddof=1)."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


@dataclass(frozen=True)
class VolTargetResult:
    """Output of :meth:`VolatilityTargeter.apply`."""

    exposure: pd.Series
    scaled_returns: pd.Series
    realized_vol: pd.Series


class VolatilityTargeter:
    """Scale strategy exposure so realized volatility tracks a target.

    Exposure at time t uses only information through t-1 (the realized-vol
    estimate is lagged by one period), so applying it to returns is free of
    look-ahead bias.
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        window: int = 20,
        max_leverage: float = 2.0,
        min_leverage: float = 0.0,
        periods_per_year: int = 252,
        initial_exposure: float = 1.0,
    ):
        if target_vol <= 0.0:
            raise ValueError(f"target_vol must be positive, got {target_vol}")
        if max_leverage <= 0.0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")
        if min_leverage < 0.0 or min_leverage > max_leverage:
            raise ValueError("min_leverage must be in [0, max_leverage]")
        self.target_vol = target_vol
        self.window = window
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.periods_per_year = periods_per_year
        self.initial_exposure = initial_exposure

    def compute_exposure(self, returns: pd.Series) -> pd.Series:
        """Exposure multiplier per period (lagged, clipped to leverage limits)."""
        vol = realized_vol(returns, self.window, self.periods_per_year)
        raw = (self.target_vol / vol).shift(1)
        exposure = raw.clip(lower=self.min_leverage, upper=self.max_leverage)
        return exposure.fillna(self.initial_exposure)

    def apply(self, returns: pd.Series) -> VolTargetResult:
        """Scale ``returns`` by the vol-target exposure series."""
        exposure = self.compute_exposure(returns)
        scaled = exposure * returns
        vol = realized_vol(returns, self.window, self.periods_per_year)
        return VolTargetResult(exposure=exposure, scaled_returns=scaled, realized_vol=vol)
