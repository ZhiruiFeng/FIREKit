# DeepTrader: Reinforcement Learning Trading Agents

## Product Overview

**DeepTrader** is the advanced AI component of the FIREKit ecosystem, implementing reinforcement learning agents for autonomous trading decisions. Built on FinRL and Stable-Baselines3, it provides customizable trading environments, reward shaping, and robust training pipelines.

### Key Value Propositions

- **Production-Ready Environments**: Realistic trading simulation with costs, slippage, and market impact
- **Multiple Algorithms**: PPO, SAC, A2C, DDPG, TD3 with financial adaptations
- **Reward Engineering**: Risk-adjusted rewards, drawdown penalties, transaction cost awareness
- **Safe Training**: Curriculum learning, constrained RL, simulation-to-reality gap mitigation
- **Interpretable Actions**: Understand why the agent makes specific decisions

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DeepTrader                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Environment Layer                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │  Stock   │  │  Crypto  │  │Portfolio │  │  Custom  │     │    │
│  │  │   Env    │  │   Env    │  │   Env    │  │   Env    │     │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │    │
│  └───────┴─────────────┴─────────────┴─────────────┴───────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                    Agent Layer                               │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │   PPO    │  │   SAC    │  │   A2C    │  │   TD3    │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                    Training Layer                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │Curriculum │  │  Reward  │  │Simulation│  │Hyperopt  │     │    │
│  │  │ Learning │  │ Shaping  │  │  Noise   │  │  Tuning  │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Evaluation Layer                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │Backtesting│  │  Paper   │  │  Live    │                   │    │
│  │  │   Eval   │  │ Trading  │  │ Monitor  │                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Algorithm Comparison

| Algorithm | Best For | Sample Efficiency | Stability | Continuous Actions |
|-----------|----------|-------------------|-----------|-------------------|
| PPO | General purpose | Medium | High | Yes |
| SAC | Sample efficient | High | Medium | Yes |
| A2C | Fast training | Low | High | Yes |
| DDPG | Deterministic | High | Low | Yes |
| TD3 | Robust continuous | High | Medium | Yes |

## Technical Specification

### Trading Environments

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class TradingEnvironment(gym.Env):
    """
    Base trading environment following OpenAI Gym interface.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_pct: float = 0.2,
        features: list = None,
        reward_scaling: float = 1e-4
    ):
        super().__init__()

        self.df = df
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.reward_scaling = reward_scaling

        # Define features used in state
        self.features = features or ['close', 'volume', 'returns', 'volatility']
        self.n_features = len(self.features)
        self.n_assets = df['symbol'].nunique() if 'symbol' in df.columns else 1

        # Action space: continuous weights for each asset (-1 to 1)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Observation space: features + portfolio state
        obs_dim = self.n_features * self.n_assets + self.n_assets + 2
        # Features + current positions + cash ratio + total value
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [self.initial_capital]

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.

        Args:
            action: Target portfolio weights (-1 to 1 for each asset)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to valid portfolio weights
        target_weights = self._normalize_action(action)

        # Get current prices
        prices = self._get_prices()

        # Calculate trades needed
        current_weights = self._get_current_weights(prices)
        trades = target_weights - current_weights

        # Execute trades with costs
        self._execute_trades(trades, prices)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.portfolio_value <= 0

        # Calculate reward
        reward = self._calculate_reward()

        # Update history
        self.portfolio_history.append(self.portfolio_value)

        info = {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'cash': self.cash
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert raw action to valid portfolio weights."""
        # Softmax-style normalization
        exp_action = np.exp(action - np.max(action))
        weights = exp_action / (exp_action.sum() + 1e-8)

        # Apply position limits
        weights = np.clip(weights, -self.max_position_pct, self.max_position_pct)

        return weights

    def _get_observation(self) -> np.ndarray:
        """Build observation from current state."""
        row = self.df.iloc[self.current_step]

        # Feature values
        features = []
        for feat in self.features:
            if feat in row:
                features.append(row[feat])

        # Portfolio state
        prices = self._get_prices()
        position_values = self.positions * prices
        weights = position_values / (self.portfolio_value + 1e-8)

        cash_ratio = self.cash / (self.portfolio_value + 1e-8)
        normalized_value = self.portfolio_value / self.initial_capital

        obs = np.concatenate([
            np.array(features),
            weights,
            [cash_ratio, normalized_value]
        ]).astype(np.float32)

        return obs

    def _get_prices(self) -> np.ndarray:
        """Get current asset prices."""
        row = self.df.iloc[self.current_step]
        return np.array([row['close']])

    def _get_current_weights(self, prices: np.ndarray) -> np.ndarray:
        """Calculate current portfolio weights."""
        position_values = self.positions * prices
        total_value = position_values.sum() + self.cash
        return position_values / (total_value + 1e-8)

    def _execute_trades(self, trades: np.ndarray, prices: np.ndarray):
        """Execute trades with transaction costs and slippage."""
        for i, trade_weight in enumerate(trades):
            if abs(trade_weight) < 0.001:  # Skip tiny trades
                continue

            trade_value = trade_weight * self.portfolio_value
            trade_shares = trade_value / prices[i]

            # Apply slippage
            if trade_shares > 0:  # Buying
                execution_price = prices[i] * (1 + self.slippage_pct)
            else:  # Selling
                execution_price = prices[i] * (1 - self.slippage_pct)

            # Transaction cost
            cost = abs(trade_value) * self.transaction_cost_pct

            # Update positions
            self.positions[i] += trade_shares
            self.cash -= trade_shares * execution_price + cost

        # Update portfolio value
        self.portfolio_value = (self.positions * prices).sum() + self.cash

    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        # Simple return-based reward
        if len(self.portfolio_history) < 2:
            return 0

        prev_value = self.portfolio_history[-1]
        curr_value = self.portfolio_value

        returns = (curr_value - prev_value) / (prev_value + 1e-8)

        return returns * self.reward_scaling


class MultiAssetTradingEnv(TradingEnvironment):
    """Extended environment for portfolio of multiple assets."""

    def __init__(
        self,
        df: pd.DataFrame,
        symbols: list,
        **kwargs
    ):
        self.symbols = symbols
        super().__init__(df, **kwargs)
        self.n_assets = len(symbols)

        # Reshape action/observation spaces
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32
        )

    def _get_prices(self) -> np.ndarray:
        """Get prices for all assets."""
        row = self.df.iloc[self.current_step]
        return np.array([row[f'{sym}_close'] for sym in self.symbols])


class CryptoTradingEnv(TradingEnvironment):
    """Crypto-specific environment with 24/7 trading."""

    def __init__(
        self,
        df: pd.DataFrame,
        trading_pair: str = 'BTC/USDT',
        leverage: float = 1.0,
        funding_rate: float = 0.0001,
        **kwargs
    ):
        super().__init__(df, **kwargs)
        self.trading_pair = trading_pair
        self.leverage = leverage
        self.funding_rate = funding_rate

    def _execute_trades(self, trades: np.ndarray, prices: np.ndarray):
        """Execute trades with leverage and funding costs."""
        super()._execute_trades(trades, prices)

        # Apply funding rate for leveraged positions
        if self.leverage > 1:
            position_value = abs(self.positions * prices).sum()
            funding_cost = position_value * self.funding_rate
            self.cash -= funding_cost
            self.portfolio_value -= funding_cost
```

### Reward Engineering

```python
class RewardShaper:
    """Advanced reward shaping for trading RL."""

    def __init__(
        self,
        base_reward: str = 'returns',
        risk_penalty: float = 0.5,
        drawdown_penalty: float = 2.0,
        turnover_penalty: float = 0.01,
        sharpe_window: int = 20
    ):
        self.base_reward = base_reward
        self.risk_penalty = risk_penalty
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.sharpe_window = sharpe_window

        self.returns_history = []
        self.peak_value = None

    def shape_reward(
        self,
        portfolio_value: float,
        prev_portfolio_value: float,
        positions: np.ndarray,
        prev_positions: np.ndarray
    ) -> float:
        """
        Compute shaped reward with risk adjustments.

        Components:
        1. Base return
        2. Risk penalty (volatility)
        3. Drawdown penalty
        4. Turnover penalty
        """
        # Base return
        returns = (portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
        self.returns_history.append(returns)

        reward = returns

        # Risk penalty (rolling volatility)
        if len(self.returns_history) >= self.sharpe_window:
            recent_returns = np.array(self.returns_history[-self.sharpe_window:])
            volatility = np.std(recent_returns)
            reward -= self.risk_penalty * volatility

        # Drawdown penalty
        if self.peak_value is None:
            self.peak_value = portfolio_value
        else:
            self.peak_value = max(self.peak_value, portfolio_value)

        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        reward -= self.drawdown_penalty * drawdown ** 2

        # Turnover penalty
        turnover = np.abs(positions - prev_positions).sum()
        reward -= self.turnover_penalty * turnover

        return reward

    def reset(self):
        """Reset reward shaper state."""
        self.returns_history = []
        self.peak_value = None


class DifferentialSharpeReward:
    """
    Differential Sharpe Ratio reward.
    Reference: Moody & Saffell (2001)
    """

    def __init__(self, eta: float = 0.001):
        self.eta = eta
        self.A = 0  # Running mean of returns
        self.B = 0  # Running mean of squared returns

    def compute(self, returns: float) -> float:
        """Compute differential Sharpe ratio."""
        delta_A = returns - self.A
        delta_B = returns ** 2 - self.B

        # Differential Sharpe
        if self.B - self.A ** 2 > 0:
            D_sharpe = (self.B * delta_A - 0.5 * self.A * delta_B) / \
                      (self.B - self.A ** 2) ** 1.5
        else:
            D_sharpe = 0

        # Update running statistics
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return D_sharpe

    def reset(self):
        self.A = 0
        self.B = 0
```

### Agent Training

```python
from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

class TradingAgentTrainer:
    """Training wrapper for RL trading agents."""

    ALGORITHMS = {
        'ppo': PPO,
        'sac': SAC,
        'a2c': A2C,
        'td3': TD3
    }

    DEFAULT_PARAMS = {
        'ppo': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'policy_kwargs': {
                'net_arch': [dict(pi=[256, 256], vf=[256, 256])]
            }
        },
        'sac': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'ent_coef': 'auto',
            'policy_kwargs': {
                'net_arch': [256, 256]
            }
        }
    }

    def __init__(
        self,
        env: gym.Env,
        algorithm: str = 'ppo',
        params: dict = None,
        device: str = 'auto'
    ):
        self.env = env
        self.algorithm = algorithm.lower()

        # Merge default and custom params
        self.params = {**self.DEFAULT_PARAMS.get(self.algorithm, {}), **(params or {})}

        # Create agent
        agent_class = self.ALGORITHMS[self.algorithm]
        self.agent = agent_class(
            'MlpPolicy',
            env,
            verbose=1,
            device=device,
            tensorboard_log='./tensorboard/',
            **self.params
        )

    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: gym.Env = None,
        eval_freq: int = 5000,
        save_path: str = './models/'
    ):
        """Train the agent."""
        callbacks = []

        if eval_env:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # Add custom trading metrics callback
        callbacks.append(TradingMetricsCallback())

        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

    def save(self, path: str):
        """Save trained agent."""
        self.agent.save(path)

    def load(self, path: str):
        """Load trained agent."""
        agent_class = self.ALGORITHMS[self.algorithm]
        self.agent = agent_class.load(path, env=self.env)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get agent's action for observation."""
        return self.agent.predict(observation, deterministic=deterministic)


class TradingMetricsCallback(BaseCallback):
    """Callback to log trading-specific metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_sharpes = []

    def _on_step(self) -> bool:
        # Log portfolio metrics from info dict
        for info in self.locals.get('infos', []):
            if 'portfolio_value' in info:
                self.logger.record('trading/portfolio_value', info['portfolio_value'])

        return True

    def _on_rollout_end(self):
        # Calculate and log trading metrics
        if len(self.episode_returns) > 0:
            self.logger.record('trading/mean_return', np.mean(self.episode_returns))
            self.logger.record('trading/sharpe', np.mean(self.episode_sharpes))


class CurriculumTrainer:
    """Curriculum learning for trading agents."""

    def __init__(
        self,
        env_factory,
        agent_class,
        stages: list
    ):
        """
        Args:
            env_factory: Function to create environment with difficulty param
            agent_class: RL algorithm class
            stages: List of dicts with difficulty params and timesteps
        """
        self.env_factory = env_factory
        self.agent_class = agent_class
        self.stages = stages

    def train(self):
        """Train through curriculum stages."""
        agent = None

        for i, stage in enumerate(self.stages):
            print(f"Stage {i+1}/{len(self.stages)}: {stage.get('name', 'unnamed')}")

            # Create environment with current difficulty
            env = self.env_factory(**stage.get('env_params', {}))

            if agent is None:
                # First stage - create new agent
                agent = self.agent_class('MlpPolicy', env, verbose=1)
            else:
                # Transfer learning - update environment
                agent.set_env(env)

            # Train for this stage
            agent.learn(
                total_timesteps=stage.get('timesteps', 50000),
                reset_num_timesteps=False
            )

        return agent
```

### Simulation-to-Reality Gap

```python
class RealityGapMitigation:
    """Techniques to reduce sim-to-real gap in trading RL."""

    @staticmethod
    def add_execution_noise(env, noise_std: float = 0.001):
        """Add noise to execution prices."""
        original_execute = env._execute_trades

        def noisy_execute(trades, prices):
            noisy_prices = prices * (1 + np.random.normal(0, noise_std, prices.shape))
            return original_execute(trades, noisy_prices)

        env._execute_trades = noisy_execute
        return env

    @staticmethod
    def add_observation_noise(env, noise_std: float = 0.01):
        """Add noise to observations."""
        original_obs = env._get_observation

        def noisy_obs():
            obs = original_obs()
            return obs + np.random.normal(0, noise_std, obs.shape)

        env._get_observation = noisy_obs
        return env

    @staticmethod
    def domain_randomization(
        env_factory,
        n_envs: int = 10,
        param_ranges: dict = None
    ):
        """Create ensemble of environments with randomized parameters."""
        default_ranges = {
            'transaction_cost_pct': (0.0005, 0.002),
            'slippage_pct': (0.0002, 0.001),
            'reward_scaling': (1e-5, 1e-3)
        }
        param_ranges = param_ranges or default_ranges

        envs = []
        for _ in range(n_envs):
            params = {}
            for param, (low, high) in param_ranges.items():
                params[param] = np.random.uniform(low, high)
            envs.append(env_factory(**params))

        return SubprocVecEnv([lambda e=env: e for env in envs])

    @staticmethod
    def conservative_q_learning(agent, conservative_alpha: float = 5.0):
        """Apply conservative Q-learning penalty."""
        # Modify SAC/TD3 to penalize OOD actions
        # This is a placeholder - full implementation requires modifying the algorithm
        pass


class ActionInterpreter:
    """Interpret agent actions for debugging and monitoring."""

    def __init__(self, symbols: list):
        self.symbols = symbols

    def interpret(
        self,
        action: np.ndarray,
        observation: np.ndarray,
        prev_positions: np.ndarray
    ) -> dict:
        """Explain agent's action decision."""
        # Normalize action to weights
        weights = np.exp(action) / np.exp(action).sum()

        # Determine trade direction for each asset
        trades = []
        for i, symbol in enumerate(self.symbols):
            weight_change = weights[i] - prev_positions[i]

            if weight_change > 0.01:
                direction = 'BUY'
            elif weight_change < -0.01:
                direction = 'SELL'
            else:
                direction = 'HOLD'

            trades.append({
                'symbol': symbol,
                'direction': direction,
                'target_weight': f"{weights[i]:.1%}",
                'current_weight': f"{prev_positions[i]:.1%}",
                'change': f"{weight_change:+.1%}"
            })

        return {
            'trades': trades,
            'total_exposure': f"{weights.sum():.1%}",
            'cash_weight': f"{1 - weights.sum():.1%}"
        }
```

## Configuration

```yaml
# deeptrader.yaml
environment:
  type: stock  # stock, crypto, portfolio
  initial_capital: 100000
  transaction_cost_pct: 0.001
  slippage_pct: 0.0005
  max_position_pct: 0.2

  features:
    - close
    - volume
    - returns
    - volatility
    - rsi_14
    - macd

agent:
  algorithm: ppo  # ppo, sac, a2c, td3
  params:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    gamma: 0.99

reward:
  type: differential_sharpe  # returns, sharpe, differential_sharpe
  risk_penalty: 0.5
  drawdown_penalty: 2.0
  turnover_penalty: 0.01

training:
  total_timesteps: 1000000
  eval_freq: 10000
  n_eval_episodes: 10

  curriculum:
    enabled: true
    stages:
      - name: low_volatility
        timesteps: 200000
        env_params:
          volatility_filter: low
      - name: medium_volatility
        timesteps: 300000
        env_params:
          volatility_filter: medium
      - name: full_market
        timesteps: 500000
        env_params:
          volatility_filter: null

reality_gap:
  execution_noise: 0.001
  observation_noise: 0.01
  domain_randomization: true

evaluation:
  backtesting:
    enabled: true
    test_periods: 5
  paper_trading:
    enabled: true
    duration_days: 30
```

## Realistic Expectations

### What RL Can Achieve
- **5-15% improvement** over buy-and-hold after careful tuning
- Better risk-adjusted returns through dynamic position sizing
- Adaptive behavior in changing market conditions
- Learned transaction cost awareness

### What RL Cannot Guarantee
- Consistent alpha generation
- Outperformance in all market regimes
- Transfer across different assets without retraining
- Real-time adaptation without periodic retraining

### Common Failure Modes
1. **Overfitting to training period**: Use proper train/test splits
2. **Reward hacking**: Agent exploits simulation artifacts
3. **Sample inefficiency**: Requires millions of timesteps
4. **Reality gap**: Simulated fills != real execution

## Roadmap

### v1.0 (Core)
- [x] Basic trading environment
- [x] PPO/SAC integration
- [x] Reward shaping
- [x] Training pipeline

### v1.1 (Robustness)
- [ ] Curriculum learning
- [ ] Domain randomization
- [ ] Reality gap mitigation
- [ ] Conservative Q-learning

### v1.2 (Multi-Asset)
- [ ] Portfolio environment
- [ ] Multi-agent trading
- [ ] Hierarchical RL

### v2.0 (Production)
- [ ] Paper trading integration
- [ ] Live monitoring dashboard
- [ ] Automated retraining
- [ ] Action interpretation UI
