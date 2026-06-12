"""Training harness: curves, reproducibility, eval isolation."""

import numpy as np
import pytest

from deeptrader.agents import QLearningAgent
from deeptrader.discretize import Discretizer
from deeptrader.env import train_test_split_envs
from deeptrader.market import regime_switching_series
from deeptrader.training import run_episode, train


@pytest.fixture(scope="module")
def envs():
    series = regime_switching_series(n=900, seed=5)
    return train_test_split_envs(series.prices, train_frac=0.7, cost_bps=5.0)


def make_agent(env, seed: int = 0) -> QLearningAgent:
    disc = Discretizer(n_bins=3).fit(env.observations)
    return QLearningAgent(disc, alpha=0.1, gamma=0.5, epsilon=1.0, epsilon_decay=0.8, seed=seed)


class TestTraining:
    def test_training_curve_length(self, envs):
        train_env, _ = envs
        result = train(make_agent(train_env), train_env, episodes=5)
        assert len(result.episode_rewards) == 5

    def test_reproducible_with_seed(self, envs):
        train_env, _ = envs
        r1 = train(make_agent(train_env, seed=3), train_env, episodes=6)
        r2 = train(make_agent(train_env, seed=3), train_env, episodes=6)
        assert r1.episode_rewards == r2.episode_rewards

    def test_seed_argument_reseeds_agent(self, envs):
        train_env, _ = envs
        r1 = train(make_agent(train_env, seed=1), train_env, episodes=6, seed=42)
        r2 = train(make_agent(train_env, seed=2), train_env, episodes=6, seed=42)
        assert r1.episode_rewards == r2.episode_rewards

    def test_periodic_eval_cadence(self, envs):
        train_env, test_env = envs
        result = train(
            make_agent(train_env), train_env, episodes=10, eval_env=test_env, eval_every=5
        )
        assert result.eval_episodes == [5, 10]
        assert len(result.eval_rewards) == 2

    def test_evaluation_does_not_train(self, envs):
        train_env, test_env = envs
        agent = make_agent(train_env)
        train(agent, train_env, episodes=3)
        q_before = agent.q.copy()
        epsilon_before = agent.epsilon
        run_episode(test_env, agent, explore=False, learn=False)
        np.testing.assert_array_equal(agent.q, q_before)
        assert agent.epsilon == epsilon_before

    def test_invalid_episode_count(self, envs):
        train_env, _ = envs
        with pytest.raises(ValueError):
            train(make_agent(train_env), train_env, episodes=0)
