"""Evaluation metrics, the learning guarantee, and cost sensitivity."""

import numpy as np
import pytest

from deeptrader.agents import BuyAndHoldAgent, QLearningAgent, RandomAgent
from deeptrader.discretize import Discretizer
from deeptrader.env import TradingEnv, train_test_split_envs
from deeptrader.evaluation import (
    _trade_stats,
    cost_sensitivity,
    evaluate,
    max_drawdown,
    sharpe_ratio,
)
from deeptrader.market import regime_switching_series
from deeptrader.training import train


def prices_from_returns(returns) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    return 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + returns)])


class TestMetricArithmetic:
    def test_max_drawdown_exact(self):
        assert max_drawdown(np.array([1.0, 1.1, 0.99, 1.2])) == pytest.approx(0.11 / 1.1)
        assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0

    def test_sharpe_exact(self):
        rets = np.array([0.01, -0.01, 0.01, -0.01, 0.02])
        expected = rets.mean() / rets.std() * np.sqrt(252)
        assert sharpe_ratio(rets) == pytest.approx(expected)
        assert sharpe_ratio(np.zeros(10)) == 0.0

    def test_trade_stats_counts_and_win_rate(self):
        positions = np.array([0, 1, 1, 0, -1, -1])
        rewards = np.array([0.0, 0.02, 0.01, 0.0, -0.01, 0.02])
        n_trades, win_rate = _trade_stats(positions, rewards)
        # changes: 0->1, 1->0, 0->-1 = 3 trades
        assert n_trades == 3
        # closed segments: long (+0.03 win), short (+0.01 win) -> 2/2
        assert win_rate == pytest.approx(1.0)

    def test_trade_stats_losing_segment(self):
        positions = np.array([1, 1, 0])
        rewards = np.array([-0.02, 0.01, 0.0])
        n_trades, win_rate = _trade_stats(positions, rewards)
        assert n_trades == 2
        assert win_rate == 0.0


class TestEvaluate:
    def test_equity_curve_compounds_rewards(self):
        returns = [0.010, -0.020, 0.005, 0.015, -0.010, 0.020, -0.005, 0.012]
        env = TradingEnv(prices_from_returns(returns), cost_bps=10.0, ac_window=3)
        result = evaluate(env, BuyAndHoldAgent())
        # Buy & hold: first step pays entry cost, then rides returns 4..7.
        step_rewards = [returns[4] - 0.001, returns[5], returns[6], returns[7]]
        expected = np.cumprod([1.0] + [1 + r for r in step_rewards])
        np.testing.assert_allclose(result.equity_curve, expected, rtol=1e-10)
        assert result.total_return == pytest.approx(expected[-1] - 1.0)
        assert result.n_trades == 1

    def test_positions_recorded(self):
        returns = [0.01] * 10
        env = TradingEnv(prices_from_returns(returns), cost_bps=0.0, ac_window=3)
        result = evaluate(env, BuyAndHoldAgent())
        assert (result.positions == 1).all()


class TestLearningGuarantee:
    def test_trained_q_agent_beats_random_oos(self):
        """The headline check: on the seeded regime-switching series, a trained
        tabular Q-agent must outperform a random policy out-of-sample."""
        series = regime_switching_series(n=1600, seed=42)
        train_env, test_env = train_test_split_envs(series.prices, train_frac=0.7, cost_bps=5.0)
        disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
        agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, epsilon=1.0, epsilon_decay=0.85, seed=1)
        train(agent, train_env, episodes=15)

        q_result = evaluate(test_env, agent)
        random_result = evaluate(test_env, RandomAgent(seed=3))
        assert q_result.total_return > random_result.total_return
        assert q_result.total_return > 0
        assert q_result.sharpe > random_result.sharpe

    def test_trained_q_agent_beats_buy_and_hold_on_this_series(self):
        series = regime_switching_series(n=1600, seed=42)
        train_env, test_env = train_test_split_envs(series.prices, train_frac=0.7, cost_bps=5.0)
        disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
        agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, epsilon=1.0, epsilon_decay=0.85, seed=1)
        train(agent, train_env, episodes=15)
        assert evaluate(test_env, agent).total_return > evaluate(
            test_env, BuyAndHoldAgent()
        ).total_return


class TestCostSensitivity:
    def test_rows_cover_agents_and_levels(self):
        series = regime_switching_series(n=900, seed=5)
        rows = cost_sensitivity(
            {"B&H": BuyAndHoldAgent(), "Random": RandomAgent(seed=1)},
            series.prices,
            cost_bps_levels=(0.0, 10.0),
        )
        assert len(rows) == 4
        assert {r["cost_bps"] for r in rows} == {0.0, 10.0}

    def test_higher_costs_never_help_an_active_trader(self):
        series = regime_switching_series(n=1600, seed=42)
        train_env, _ = train_test_split_envs(series.prices, train_frac=0.7, cost_bps=5.0)
        disc = Discretizer(n_bins=[3, 3, 3, 5, 3]).fit(train_env.observations)
        agent = QLearningAgent(disc, alpha=0.1, gamma=0.5, epsilon=1.0, epsilon_decay=0.85, seed=1)
        train(agent, train_env, episodes=15)
        rows = cost_sensitivity({"Q": agent}, series.prices, cost_bps_levels=(0.0, 5.0, 10.0))
        returns = [r["total_return"] for r in rows]
        assert returns[0] >= returns[1] >= returns[2]
