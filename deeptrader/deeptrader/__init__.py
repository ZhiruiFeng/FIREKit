"""DeepTrader: offline RL trading agents on a self-contained trading environment."""

from deeptrader.agents import (
    ACTION_FLAT,
    ACTION_LONG,
    ACTION_SHORT,
    Agent,
    BuyAndHoldAgent,
    QLearningAgent,
    RandomAgent,
    REINFORCEAgent,
    SMACrossoverAgent,
)
from deeptrader.discretize import Discretizer
from deeptrader.env import FEATURE_NAMES, TradingEnv, build_features, train_test_split_envs
from deeptrader.evaluation import (
    EvalResult,
    cost_sensitivity,
    evaluate,
    max_drawdown,
    sharpe_ratio,
)
from deeptrader.market import RegimeSeries, regime_switching_series
from deeptrader.training import TrainingResult, run_episode, train

__version__ = "0.1.0"

__all__ = [
    "ACTION_FLAT",
    "ACTION_LONG",
    "ACTION_SHORT",
    "FEATURE_NAMES",
    "Agent",
    "BuyAndHoldAgent",
    "Discretizer",
    "EvalResult",
    "QLearningAgent",
    "REINFORCEAgent",
    "RandomAgent",
    "RegimeSeries",
    "SMACrossoverAgent",
    "TradingEnv",
    "TrainingResult",
    "build_features",
    "cost_sensitivity",
    "evaluate",
    "max_drawdown",
    "regime_switching_series",
    "run_episode",
    "sharpe_ratio",
    "train",
    "train_test_split_envs",
]
