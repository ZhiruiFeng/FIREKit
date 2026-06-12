"""TradingEnv mechanics: rewards, costs, episode flow, train/test split."""

import numpy as np
import pytest

from deeptrader.env import TradingEnv, build_features, train_test_split_envs

RETURNS = np.array([0.010, -0.020, 0.005, 0.015, -0.010, 0.020, -0.005, 0.012])


def prices_from_returns(returns: np.ndarray) -> np.ndarray:
    return 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + returns)])


@pytest.fixture
def env() -> TradingEnv:
    # ac_window=3 -> decisions at t = 3..6 (4 steps), rewards use returns 4..7.
    return TradingEnv(prices_from_returns(RETURNS), cost_bps=10.0, ac_window=3)


class TestEnvBasics:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert obs.shape == (5,)
        assert env.observation_dim == 5
        assert env.n_actions == 3

    def test_step_count(self, env):
        assert env.n_steps == len(RETURNS) - 1 - 3
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(1)
            steps += 1
        assert steps == env.n_steps

    def test_observations_matrix_shape(self, env):
        assert env.observations.shape == (env.n_steps, 5)

    def test_step_after_done_raises(self, env):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(1)
        with pytest.raises(RuntimeError):
            env.step(1)

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step(5)

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            TradingEnv(prices_from_returns(RETURNS), cost_bps=-1.0)


class TestRewardArithmetic:
    def test_long_reward_is_next_return_minus_entry_cost(self, env):
        env.reset()
        _, reward, _, info = env.step(2)  # go long from flat: one unit of cost
        # env reward uses exact price-based returns, equal to RETURNS by construction
        assert reward == pytest.approx(RETURNS[4] - 0.001, abs=1e-12)
        assert info["position"] == 1

    def test_no_cost_when_position_unchanged(self, env):
        env.reset()
        env.step(2)
        _, reward, _, _ = env.step(2)  # stay long
        assert reward == pytest.approx(RETURNS[5], abs=1e-12)

    def test_flat_position_earns_nothing_but_pays_exit(self, env):
        env.reset()
        env.step(2)
        env.step(2)
        _, reward, _, _ = env.step(1)  # long -> flat
        assert reward == pytest.approx(-0.001, abs=1e-12)

    def test_flip_costs_double(self, env):
        env.reset()
        env.step(2)  # flat -> long
        _, reward, _, _ = env.step(0)  # long -> short: |(-1) - 1| = 2 cost units
        assert reward == pytest.approx(-RETURNS[5] - 0.002, abs=1e-12)

    def test_short_sign(self, env):
        env.reset()
        _, reward, _, _ = env.step(0)
        assert reward == pytest.approx(-RETURNS[4] - 0.001, abs=1e-12)

    def test_zero_cost_env(self):
        env = TradingEnv(prices_from_returns(RETURNS), cost_bps=0.0, ac_window=3)
        env.reset()
        _, reward, _, _ = env.step(2)
        assert reward == pytest.approx(RETURNS[4], abs=1e-12)


class TestFeatures:
    def test_no_lookahead(self):
        # The feature row for time t must not change when future returns change.
        full = build_features(RETURNS, ac_window=3)
        for t in range(3, len(RETURNS)):
            truncated = build_features(RETURNS[: t + 1], ac_window=3)
            np.testing.assert_allclose(truncated[-1], full[t - 3])

    def test_feature_values_exact(self):
        feats = build_features(RETURNS, ac_window=3)
        t = 4
        row = feats[t - 3]
        assert row[0] == pytest.approx(100 * RETURNS[t])
        assert row[1] == pytest.approx(100 * RETURNS[t - 2 : t + 1].mean())
        expected_ac = np.mean(np.sign(RETURNS[t - 2 : t + 1] * RETURNS[t - 3 : t]))
        assert row[2] == pytest.approx(expected_ac)
        assert row[3] == pytest.approx(row[0] * row[2])
        assert row[4] == pytest.approx(row[1] * row[2])

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError):
            build_features(RETURNS[:3], ac_window=3)


class TestTrainTestSplit:
    def test_no_rewarded_return_overlap(self):
        prices = prices_from_returns(np.linspace(-0.01, 0.01, 200))
        train_env, test_env = train_test_split_envs(prices, train_frac=0.7, cost_bps=0.0)
        split = int(len(prices) * 0.7)
        # Train env sees only the first `split` prices.
        np.testing.assert_array_equal(train_env.prices, prices[:split])
        # Test env starts ac_window+1 prices before the split (feature warm-up only).
        np.testing.assert_array_equal(test_env.prices, prices[split - 21 :])
        # First rewarded return in the test env is strictly after the train segment.
        first_rewarded = test_env.returns[test_env.ac_window + 1]
        global_first = (prices[split + 1] - prices[split]) / prices[split]
        assert first_rewarded == pytest.approx(global_first)

    def test_invalid_fraction_raises(self):
        prices = prices_from_returns(np.linspace(-0.01, 0.01, 200))
        with pytest.raises(ValueError):
            train_test_split_envs(prices, train_frac=0.99)
