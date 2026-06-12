"""Trading agents: tabular Q-learning, linear REINFORCE, and rule-based baselines.

Design-doc substitution: the original spec calls for PPO/SAC via FinRL and
Stable-Baselines3 (torch). This offline MVP implements tabular Q-learning and
a NumPy linear-softmax REINFORCE policy gradient instead, behind one common
``Agent`` interface (act / update / save / load).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from deeptrader.discretize import Discretizer

ACTION_SHORT, ACTION_FLAT, ACTION_LONG = 0, 1, 2


class Agent(ABC):
    """Common interface for all trading agents."""

    name: str = "agent"

    def reset(self) -> None:
        """Clear per-episode internal state (called at every episode start)."""

    @abstractmethod
    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        """Choose an action (0 short, 1 flat, 2 long)."""

    def update(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        """Learn from one transition (no-op for rule-based agents)."""

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.name} does not support save")

    @classmethod
    def load(cls, path: str | Path) -> Agent:
        raise NotImplementedError


class QLearningAgent(Agent):
    """Tabular Q-learning over discretized observations, epsilon-greedy with decay.

    Epsilon decays multiplicatively at the end of each episode (when ``update``
    sees ``done=True``).
    """

    name = "Q-learning"

    def __init__(
        self,
        discretizer: Discretizer,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.92,
        seed: int = 0,
    ):
        if not discretizer.is_fit:
            raise ValueError("discretizer must be fit before building the agent")
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha in (0,1], gamma in [0,1]")
        self.discretizer = discretizer
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((discretizer.n_states, n_actions))

    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        if explore and self.rng.uniform() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        state = self.discretizer.transform(obs)
        return int(np.argmax(self.q[state]))

    def update(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        s = self.discretizer.transform(obs)
        s_next = self.discretizer.transform(next_obs)
        target = reward + (0.0 if done else self.gamma * float(np.max(self.q[s_next])))
        self.q[s, action] += self.alpha * (target - self.q[s, action])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "QLearningAgent",
            "q": self.q.tolist(),
            "discretizer": self.discretizer.to_dict(),
            "params": {
                "n_actions": self.n_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "seed": self.seed,
            },
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> QLearningAgent:
        payload = json.loads(Path(path).read_text())
        if payload.get("type") != "QLearningAgent":
            raise ValueError("not a QLearningAgent checkpoint")
        agent = cls(Discretizer.from_dict(payload["discretizer"]), **payload["params"])
        agent.q = np.asarray(payload["q"], dtype=float)
        return agent


class REINFORCEAgent(Agent):
    """Linear-softmax policy gradient (REINFORCE) with a mean baseline.

    The policy is ``pi(a | x) = softmax(theta @ [x_standardized, 1])`` and the
    parameters are updated by NumPy gradient ascent at each episode end using
    normalized returns-to-go.
    """

    name = "REINFORCE"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 3,
        lr: float = 0.05,
        gamma: float = 0.99,
        seed: int = 0,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros((n_actions, obs_dim + 1))
        self.obs_mean = np.zeros(obs_dim)
        self.obs_scale = np.ones(obs_dim)
        self._episode: list[tuple[np.ndarray, int, float]] = []

    def fit_scaler(self, observations: np.ndarray) -> None:
        """Standardize features using (training) observation statistics."""
        observations = np.asarray(observations, dtype=float)
        self.obs_mean = observations.mean(axis=0)
        self.obs_scale = np.where(observations.std(axis=0) > 1e-12, observations.std(axis=0), 1.0)

    def _phi(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, dtype=float) - self.obs_mean) / self.obs_scale
        return np.concatenate([x, [1.0]])

    def policy(self, obs: np.ndarray) -> np.ndarray:
        logits = self.theta @ self._phi(obs)
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        probs = self.policy(obs)
        if explore:
            return int(self.rng.choice(self.n_actions, p=probs))
        return int(np.argmax(probs))

    def update(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self._episode.append((np.asarray(obs, dtype=float), action, reward))
        if done:
            self._learn_episode()
            self._episode = []

    def reset(self) -> None:
        self._episode = []

    def _learn_episode(self) -> None:
        if not self._episode:
            return
        rewards = np.array([r for _, _, r in self._episode])
        # Returns-to-go with discounting.
        returns = np.empty_like(rewards)
        acc = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            acc = rewards[i] + self.gamma * acc
            returns[i] = acc
        # Baseline + normalization keeps step sizes stable across reward scales.
        advantage = returns - returns.mean()
        std = advantage.std()
        if std > 1e-12:
            advantage = advantage / std

        x = np.stack([self._phi(obs) for obs, _, _ in self._episode])  # (T, d+1)
        logits = x @ self.theta.T
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)  # (T, A)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(probs)), [a for _, a, _ in self._episode]] = 1.0
        grad = ((onehot - probs) * advantage[:, None]).T @ x / len(probs)
        self.theta += self.lr * grad

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "REINFORCEAgent",
            "theta": self.theta.tolist(),
            "obs_mean": self.obs_mean.tolist(),
            "obs_scale": self.obs_scale.tolist(),
            "params": {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "lr": self.lr,
                "gamma": self.gamma,
                "seed": self.seed,
            },
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> REINFORCEAgent:
        payload = json.loads(Path(path).read_text())
        if payload.get("type") != "REINFORCEAgent":
            raise ValueError("not a REINFORCEAgent checkpoint")
        agent = cls(**payload["params"])
        agent.theta = np.asarray(payload["theta"], dtype=float)
        agent.obs_mean = np.asarray(payload["obs_mean"], dtype=float)
        agent.obs_scale = np.asarray(payload["obs_scale"], dtype=float)
        return agent


class BuyAndHoldAgent(Agent):
    """Always long: the benchmark."""

    name = "Buy & Hold"

    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        return ACTION_LONG


class SMACrossoverAgent(Agent):
    """Rule-based baseline: long when the fast SMA of (reconstructed) log price
    is above the slow SMA, short otherwise; flat until enough history.

    Reconstructs a price proxy by accumulating the last-return feature
    (``obs[0]``), so it only consumes the same observations as the RL agents.
    """

    name = "SMA crossover"

    def __init__(self, fast: int = 5, slow: int = 20):
        if not 0 < fast < slow:
            raise ValueError("need 0 < fast < slow")
        self.fast = fast
        self.slow = slow
        self._log_price = 0.0
        self._history: list[float] = []

    def reset(self) -> None:
        self._log_price = 0.0
        self._history = []

    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        self._log_price += float(obs[0])  # scaled return; scaling cancels in the SMA comparison
        self._history.append(self._log_price)
        if len(self._history) < self.slow:
            return ACTION_FLAT
        fast_ma = float(np.mean(self._history[-self.fast :]))
        slow_ma = float(np.mean(self._history[-self.slow :]))
        return ACTION_LONG if fast_ma > slow_ma else ACTION_SHORT


class RandomAgent(Agent):
    """Uniform random policy: the noise floor any learner must beat."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, explore: bool = False) -> int:
        return int(self.rng.integers(3))
