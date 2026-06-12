"""Drawdown tracking and circuit-breaker protection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Drawdown series (>= 0) from an equity curve: (peak - value) / peak."""
    peak = equity.cummax()
    return (peak - equity) / peak


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (positive fraction)."""
    if len(equity) == 0:
        return 0.0
    return float(drawdown_curve(equity).max())


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound a simple-return series into an equity curve."""
    return initial * (1.0 + returns).cumprod()


@dataclass(frozen=True)
class DrawdownProtectionResult:
    """Output of :meth:`DrawdownCircuitBreaker.apply`."""

    filtered_returns: pd.Series
    exposure: pd.Series
    drawdown: pd.Series  # drawdown of the *underlying* strategy
    protected_equity: pd.Series
    n_triggers: int


class DrawdownCircuitBreaker:
    """Stateful max-drawdown circuit breaker.

    Monitors the underlying strategy's equity. When its drawdown breaches
    ``max_drawdown``, exposure is cut to ``reduced_exposure`` starting the
    *next* period (no look-ahead). Exposure is restored once the underlying
    drawdown recovers to ``reentry_drawdown`` or better.
    """

    ACTIVE = "active"
    TRIGGERED = "triggered"

    def __init__(
        self,
        max_drawdown: float = 0.15,
        reentry_drawdown: float = 0.05,
        reduced_exposure: float = 0.0,
    ):
        if not 0.0 < max_drawdown < 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1), got {max_drawdown}")
        if not 0.0 <= reentry_drawdown < max_drawdown:
            raise ValueError("reentry_drawdown must be in [0, max_drawdown)")
        if not 0.0 <= reduced_exposure < 1.0:
            raise ValueError("reduced_exposure must be in [0, 1)")
        self.max_dd = max_drawdown
        self.reentry_dd = reentry_drawdown
        self.reduced_exposure = reduced_exposure
        self.reset()

    def reset(self) -> None:
        """Clear all internal state."""
        self.state = self.ACTIVE
        self.peak: float | None = None
        self.current_drawdown = 0.0
        self.n_triggers = 0

    def step(self, equity_value: float) -> float:
        """Observe one equity value; return the exposure for the NEXT period."""
        if equity_value <= 0.0:
            raise ValueError(f"equity must be positive, got {equity_value}")
        if self.peak is None or equity_value > self.peak:
            self.peak = equity_value
        self.current_drawdown = (self.peak - equity_value) / self.peak

        if self.state == self.ACTIVE and self.current_drawdown >= self.max_dd:
            self.state = self.TRIGGERED
            self.n_triggers += 1
        elif self.state == self.TRIGGERED and self.current_drawdown <= self.reentry_dd:
            self.state = self.ACTIVE

        return self.reduced_exposure if self.state == self.TRIGGERED else 1.0

    def apply(self, returns: pd.Series, initial_equity: float = 1.0) -> DrawdownProtectionResult:
        """Run the breaker over a return series as a stateful filter.

        The breaker watches the underlying (unprotected) equity curve and
        produces an exposure series applied to ``returns``.
        """
        self.reset()
        underlying = equity_from_returns(returns, initial_equity)
        n = len(returns)
        exposure = np.empty(n)
        current = 1.0  # exposure for the first period
        for t in range(n):
            exposure[t] = current
            current = self.step(float(underlying.iloc[t]))
        exposure_s = pd.Series(exposure, index=returns.index, name="exposure")
        filtered = exposure_s * returns
        protected = equity_from_returns(filtered, initial_equity)
        return DrawdownProtectionResult(
            filtered_returns=filtered,
            exposure=exposure_s,
            drawdown=drawdown_curve(underlying),
            protected_equity=protected,
            n_triggers=self.n_triggers,
        )
