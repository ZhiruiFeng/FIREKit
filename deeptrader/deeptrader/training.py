"""Multi-episode training harness with seed control and periodic OOS evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deeptrader.agents import Agent
from deeptrader.env import TradingEnv


@dataclass
class TrainingResult:
    episode_rewards: list[float] = field(default_factory=list)
    eval_episodes: list[int] = field(default_factory=list)
    eval_rewards: list[float] = field(default_factory=list)  # greedy OOS total reward


def run_episode(env: TradingEnv, agent: Agent, explore: bool, learn: bool) -> float:
    """One full pass over the environment; returns total reward."""
    agent.reset()
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        action = agent.act(obs, explore=explore)
        next_obs, reward, done, _ = env.step(action)
        if learn:
            agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
        total += reward
    return total


def train(
    agent: Agent,
    env: TradingEnv,
    episodes: int,
    eval_env: TradingEnv | None = None,
    eval_every: int = 10,
    seed: int | None = None,
) -> TrainingResult:
    """Train ``agent`` on ``env`` for ``episodes`` full passes.

    The agent is never trained on ``eval_env``: periodic evaluations run the
    greedy policy with learning disabled. ``seed`` reseeds the agent's RNG for
    reproducible exploration.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if seed is not None and hasattr(agent, "rng"):
        agent.rng = np.random.default_rng(seed)
    result = TrainingResult()
    for episode in range(1, episodes + 1):
        total = run_episode(env, agent, explore=True, learn=True)
        result.episode_rewards.append(total)
        if eval_env is not None and episode % eval_every == 0:
            oos = run_episode(eval_env, agent, explore=False, learn=False)
            result.eval_episodes.append(episode)
            result.eval_rewards.append(oos)
    return result
