"""Self-contained gym-like trading environment (discrete short/flat/long)."""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_NAMES = (
    "ret_1",  # last return (x100)
    "mom_3",  # 3-period mean return (x100)
    "autocorr_20",  # sign-agreement of consecutive returns over 20 periods
    "ret_x_ac",  # interaction: last return * autocorr signal
    "mom_x_ac",  # interaction: momentum * autocorr signal
)


def build_features(returns: np.ndarray, ac_window: int = 20) -> np.ndarray:
    """Feature rows for t in ``[ac_window, len(returns) - 1]``.

    Row ``j`` describes time ``t = ac_window + j`` using information up to and
    including ``returns[t]`` (no lookahead).
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n <= ac_window:
        raise ValueError(f"need more than {ac_window} returns")
    ts = np.arange(ac_window, n)

    ret_1 = 100.0 * returns[ts]
    mom_3 = 100.0 * np.array([returns[t - 2 : t + 1].mean() for t in ts])
    sign_agree = np.sign(returns[1:] * returns[:-1])  # index i -> sign(r_i * r_{i-1}), i >= 1
    autocorr = np.array([sign_agree[t - ac_window : t].mean() for t in ts])
    return np.column_stack([ret_1, mom_3, autocorr, ret_1 * autocorr, mom_3 * autocorr])


class TradingEnv:
    """Episodic trading over one price series.

    Gym-like but dependency-free: ``reset() -> obs`` and
    ``step(action) -> (obs, reward, done, info)``.

    - Actions: 0 = short (-1), 1 = flat (0), 2 = long (+1).
    - Observation: vector of recent-return features (see ``FEATURE_NAMES``).
    - Reward: ``position * next_return - cost_rate * |position - prev_position|``.
    """

    ACTION_POSITIONS = (-1, 0, 1)

    def __init__(self, prices: np.ndarray, cost_bps: float = 5.0, ac_window: int = 20):
        prices = np.asarray(prices, dtype=float)
        if len(prices) < ac_window + 4:
            raise ValueError("price series too short for the feature window")
        if cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        self.prices = prices
        self.returns = np.diff(prices) / prices[:-1]
        self.ac_window = ac_window
        self.cost_rate = cost_bps / 1e4
        self.cost_bps = cost_bps
        self.features = build_features(self.returns, ac_window)
        self._start_t = ac_window
        self._end_t = len(self.returns) - 1  # last decision index is _end_t - 1
        self.n_steps = self._end_t - self._start_t
        self._t = self._start_t
        self.position = 0

    # ------------------------------------------------------------- properties
    @property
    def observation_dim(self) -> int:
        return self.features.shape[1]

    @property
    def n_actions(self) -> int:
        return len(self.ACTION_POSITIONS)

    @property
    def observations(self) -> np.ndarray:
        """All decision-time observations (e.g. for fitting a discretizer)."""
        return self.features[:-1]

    # ------------------------------------------------------------------- API
    def _obs(self) -> np.ndarray:
        return self.features[self._t - self.ac_window]

    def reset(self) -> np.ndarray:
        self._t = self._start_t
        self.position = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._t >= self._end_t:
            raise RuntimeError("episode is over; call reset()")
        if action not in (0, 1, 2):
            raise ValueError(f"invalid action {action}")
        position = self.ACTION_POSITIONS[action]
        next_return = self.returns[self._t + 1]
        cost = self.cost_rate * abs(position - self.position)
        reward = position * next_return - cost
        self.position = position
        self._t += 1
        done = self._t >= self._end_t
        info = {
            "position": position,
            "step_return": float(next_return),
            "cost": float(cost),
            "t": self._t,
        }
        return self._obs(), float(reward), done, info


def train_test_split_envs(
    prices: np.ndarray,
    train_frac: float = 0.7,
    cost_bps: float = 5.0,
    ac_window: int = 20,
) -> tuple[TradingEnv, TradingEnv]:
    """Chronological train/test environments with no shared rewarded returns.

    The test environment keeps ``ac_window + 1`` price points of history from
    the end of the training segment purely as feature warm-up (past data is
    legitimately observable); every *rewarded* return in the test environment
    is strictly out-of-sample.
    """
    if not 0.1 <= train_frac <= 0.95:
        raise ValueError("train_frac must be in [0.1, 0.95]")
    prices = np.asarray(prices, dtype=float)
    split = int(len(prices) * train_frac)
    train_env = TradingEnv(prices[:split], cost_bps=cost_bps, ac_window=ac_window)
    test_env = TradingEnv(prices[split - (ac_window + 1) :], cost_bps=cost_bps, ac_window=ac_window)
    return train_env, test_env
