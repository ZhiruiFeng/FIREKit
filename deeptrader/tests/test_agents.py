"""Agent behaviour: Q-update arithmetic, REINFORCE gradients, baselines, save/load."""

import numpy as np
import pytest

from deeptrader.agents import (
    ACTION_FLAT,
    ACTION_LONG,
    ACTION_SHORT,
    BuyAndHoldAgent,
    QLearningAgent,
    RandomAgent,
    REINFORCEAgent,
    SMACrossoverAgent,
)
from deeptrader.discretize import Discretizer


def single_state_discretizer() -> Discretizer:
    """Everything maps to state 0: isolates the Q-update arithmetic."""
    return Discretizer(n_bins=1).fit(np.zeros((4, 1)))


def two_state_discretizer() -> Discretizer:
    """1-D split at 0: negative -> state 0, positive -> state 1."""
    return Discretizer(n_bins=2).fit(np.array([[-1.0], [-0.5], [0.5], [1.0]]))


OBS = np.array([0.0])


class TestQLearning:
    def test_q_update_arithmetic_hand_built(self):
        agent = QLearningAgent(
            single_state_discretizer(), alpha=0.5, gamma=0.9, epsilon=0.0
        )
        agent.update(OBS, 2, 1.0, OBS, done=False)
        # q = 0 + 0.5 * (1 + 0.9*0 - 0) = 0.5
        assert agent.q[0, 2] == pytest.approx(0.5)
        agent.update(OBS, 2, 1.0, OBS, done=False)
        # target = 1 + 0.9*0.5 = 1.45; q = 0.5 + 0.5*(1.45 - 0.5) = 0.975
        assert agent.q[0, 2] == pytest.approx(0.975)

    def test_terminal_update_has_no_bootstrap(self):
        agent = QLearningAgent(
            single_state_discretizer(), alpha=1.0, gamma=0.9, epsilon=0.5, epsilon_decay=0.8
        )
        agent.q[0, :] = 100.0  # large future values must be ignored at terminal
        agent.update(OBS, 1, 2.0, OBS, done=True)
        assert agent.q[0, 1] == pytest.approx(2.0)

    def test_epsilon_decays_on_episode_end_with_floor(self):
        agent = QLearningAgent(
            single_state_discretizer(), epsilon=1.0, epsilon_min=0.5, epsilon_decay=0.5
        )
        agent.update(OBS, 0, 0.0, OBS, done=True)
        assert agent.epsilon == pytest.approx(0.5)
        agent.update(OBS, 0, 0.0, OBS, done=True)
        assert agent.epsilon == pytest.approx(0.5)  # floored

    def test_cross_state_bootstrap(self):
        disc = two_state_discretizer()
        agent = QLearningAgent(disc, alpha=1.0, gamma=0.5, epsilon=0.0)
        agent.q[1, 0] = 4.0  # best value in next state
        agent.update(np.array([-1.0]), 2, 1.0, np.array([1.0]), done=False)
        assert agent.q[0, 2] == pytest.approx(1.0 + 0.5 * 4.0)

    def test_greedy_act_is_argmax(self):
        agent = QLearningAgent(two_state_discretizer(), epsilon=0.0)
        agent.q[0] = [0.1, 0.9, 0.3]
        agent.q[1] = [0.7, 0.2, 0.1]
        assert agent.act(np.array([-1.0])) == 1
        assert agent.act(np.array([1.0])) == 0

    def test_exploration_uses_epsilon(self):
        agent = QLearningAgent(single_state_discretizer(), epsilon=1.0, seed=0)
        agent.q[0] = [0.0, 0.0, 1.0]
        actions = {agent.act(OBS, explore=True) for _ in range(50)}
        assert actions == {0, 1, 2}  # fully random at epsilon = 1
        assert agent.act(OBS, explore=False) == 2

    def test_save_load_roundtrip(self, tmp_path):
        agent = QLearningAgent(two_state_discretizer(), alpha=0.2, gamma=0.7, epsilon=0.3)
        agent.q[0] = [0.1, 0.9, 0.3]
        agent.q[1] = [0.7, 0.2, 0.8]
        path = tmp_path / "q.json"
        agent.save(path)
        clone = QLearningAgent.load(path)
        np.testing.assert_allclose(clone.q, agent.q)
        assert clone.alpha == 0.2 and clone.gamma == 0.7
        for obs in ([-1.0], [1.0]):
            assert clone.act(np.array(obs)) == agent.act(np.array(obs))


class TestReinforce:
    def test_initial_policy_is_uniform(self):
        agent = REINFORCEAgent(obs_dim=3)
        probs = agent.policy(np.array([0.5, -1.0, 2.0]))
        np.testing.assert_allclose(probs, [1 / 3] * 3)
        assert probs.sum() == pytest.approx(1.0)

    def test_gradient_moves_probability_toward_rewarded_action(self):
        agent = REINFORCEAgent(obs_dim=2, lr=0.5, gamma=1.0, seed=0)
        obs = np.array([1.0, -1.0])
        before = agent.policy(obs)[ACTION_LONG]
        # Episode: LONG earns +1 twice, SHORT loses -1, episode ends.
        agent.update(obs, ACTION_LONG, 1.0, obs, done=False)
        agent.update(obs, ACTION_LONG, 1.0, obs, done=False)
        agent.update(obs, ACTION_SHORT, -1.0, obs, done=True)
        after = agent.policy(obs)[ACTION_LONG]
        assert after > before

    def test_buffer_cleared_after_episode(self):
        agent = REINFORCEAgent(obs_dim=1)
        agent.update(np.array([1.0]), 1, 0.5, np.array([1.0]), done=True)
        assert agent._episode == []

    def test_scaler_standardizes_features(self):
        agent = REINFORCEAgent(obs_dim=2)
        observations = np.array([[0.0, 10.0], [2.0, 30.0], [4.0, 50.0]])
        agent.fit_scaler(observations)
        phi = agent._phi(np.array([2.0, 30.0]))
        np.testing.assert_allclose(phi, [0.0, 0.0, 1.0])  # centred + bias

    def test_save_load_roundtrip(self, tmp_path):
        agent = REINFORCEAgent(obs_dim=2, lr=0.3, seed=5)
        agent.theta = np.arange(9, dtype=float).reshape(3, 3)
        agent.fit_scaler(np.array([[0.0, 1.0], [2.0, 3.0]]))
        path = tmp_path / "pg.json"
        agent.save(path)
        clone = REINFORCEAgent.load(path)
        np.testing.assert_allclose(clone.theta, agent.theta)
        probe = np.array([0.7, -0.2])
        np.testing.assert_allclose(clone.policy(probe), agent.policy(probe))


class TestBaselines:
    def test_buy_and_hold_always_long(self):
        agent = BuyAndHoldAgent()
        for _ in range(5):
            assert agent.act(np.zeros(5)) == ACTION_LONG

    def test_sma_flat_until_history(self):
        agent = SMACrossoverAgent(fast=2, slow=4)
        obs = np.zeros(5)
        obs[0] = 1.0
        assert agent.act(obs) == ACTION_FLAT
        assert agent.act(obs) == ACTION_FLAT
        assert agent.act(obs) == ACTION_FLAT

    def test_sma_goes_long_in_uptrend_short_in_downtrend(self):
        agent = SMACrossoverAgent(fast=2, slow=4)
        up = np.zeros(5)
        up[0] = 1.0
        for _ in range(3):
            agent.act(up)
        assert agent.act(up) == ACTION_LONG
        down = np.zeros(5)
        down[0] = -1.0
        for _ in range(6):
            action = agent.act(down)
        assert action == ACTION_SHORT

    def test_sma_reset_clears_history(self):
        agent = SMACrossoverAgent(fast=2, slow=4)
        obs = np.zeros(5)
        obs[0] = 1.0
        for _ in range(5):
            agent.act(obs)
        agent.reset()
        assert agent.act(obs) == ACTION_FLAT

    def test_random_agent_deterministic_per_seed(self):
        agent1, agent2 = RandomAgent(seed=9), RandomAgent(seed=9)
        seq1 = [agent1.act(np.zeros(5)) for _ in range(20)]
        seq2 = [agent2.act(np.zeros(5)) for _ in range(20)]
        assert seq1 == seq2
        assert set(seq1) <= {0, 1, 2}
