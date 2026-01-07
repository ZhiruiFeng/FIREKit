"""
Accelerated Backtest Core

High-performance computation kernels for backtesting using JAX and Numba.
Provides JIT-compiled functions for vectorized operations.
"""

from __future__ import annotations

from typing import Callable, Tuple, NamedTuple
import numpy as np

# Type alias for arrays (supports both numpy and jax arrays)
Array = np.ndarray


class BacktestArrays(NamedTuple):
    """Container for backtest computation arrays."""
    returns: Array
    equity_curve: Array
    signals: Array
    trades: Array
    costs: Array


class MetricsResult(NamedTuple):
    """Container for computed metrics."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float


# ============================================================================
# NumPy Implementation (Fallback)
# ============================================================================

def numpy_compute_returns(
    prices: np.ndarray,
    signals: np.ndarray,
    slippage_bps: float = 5.0,
    commission_pct: float = 0.001,
) -> BacktestArrays:
    """
    Compute strategy returns using NumPy.

    Args:
        prices: Close price array
        signals: Position signals (-1 to 1)
        slippage_bps: Slippage in basis points
        commission_pct: Commission as percentage

    Returns:
        BacktestArrays with computed values
    """
    # Price returns (shifted by 1 to avoid lookahead)
    price_returns = np.diff(prices) / prices[:-1]

    # Align signals with future returns
    aligned_signals = signals[:-1] if len(signals) == len(prices) else signals
    if len(aligned_signals) > len(price_returns):
        aligned_signals = aligned_signals[:len(price_returns)]

    # Compute trades (position changes)
    trades = np.diff(np.concatenate([[0], aligned_signals]))

    # Execution costs
    slippage = np.abs(trades) * slippage_bps / 10000
    commission = np.abs(trades) * commission_pct
    costs = slippage + commission
    if len(costs) > len(price_returns):
        costs = costs[:len(price_returns)]

    # Strategy returns
    strategy_returns = aligned_signals * price_returns[:len(aligned_signals)]
    strategy_returns = strategy_returns - costs[:len(strategy_returns)]

    # Equity curve
    equity_multiplier = np.cumprod(1 + strategy_returns)
    equity_curve = np.concatenate([[1.0], equity_multiplier])

    return BacktestArrays(
        returns=strategy_returns,
        equity_curve=equity_curve,
        signals=aligned_signals,
        trades=trades,
        costs=costs,
    )


def numpy_compute_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
) -> MetricsResult:
    """
    Compute performance metrics using NumPy.

    Args:
        returns: Strategy return array
        equity_curve: Equity curve array
        periods_per_year: Trading periods per year

    Returns:
        MetricsResult with computed metrics
    """
    if len(returns) == 0:
        return MetricsResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            win_rate=0.0,
            profit_factor=0.0,
        )

    # Total and annual return
    total_return = equity_curve[-1] / equity_curve[0] - 1
    n_years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Volatility
    volatility = np.std(returns) * np.sqrt(periods_per_year)

    # Sharpe ratio
    mean_return = np.mean(returns)
    daily_std = np.std(returns)
    sharpe_ratio = mean_return / max(daily_std, 1e-10) * np.sqrt(periods_per_year)

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    sortino_ratio = mean_return / max(downside_std, 1e-10) * np.sqrt(periods_per_year)

    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1
    max_drawdown = np.min(drawdowns)

    # Calmar ratio
    calmar_ratio = annual_return / max(abs(max_drawdown), 1e-10)

    # Win rate and profit factor
    win_rate = np.mean(returns > 0)
    gains = np.sum(returns[returns > 0])
    losses = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gains / max(losses, 1e-10)

    return MetricsResult(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        volatility=volatility,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def numpy_batch_backtest(
    prices: np.ndarray,
    signal_batches: np.ndarray,
    slippage_bps: float = 5.0,
    commission_pct: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run batch backtests using NumPy broadcasting.

    Args:
        prices: Close price array (n_periods,)
        signal_batches: Signal arrays for each param combo (n_combos, n_periods)
        slippage_bps: Slippage in basis points
        commission_pct: Commission percentage

    Returns:
        Tuple of (returns_batch, metrics_batch)
    """
    n_combos = signal_batches.shape[0]

    # Price returns
    price_returns = np.diff(prices) / prices[:-1]

    # Align signals
    aligned_signals = signal_batches[:, :-1]
    if aligned_signals.shape[1] > len(price_returns):
        aligned_signals = aligned_signals[:, :len(price_returns)]

    # Trades
    zeros = np.zeros((n_combos, 1))
    trades = np.diff(np.concatenate([zeros, aligned_signals], axis=1), axis=1)

    # Costs (broadcast across all combos)
    costs = np.abs(trades) * (slippage_bps / 10000 + commission_pct)
    if costs.shape[1] > len(price_returns):
        costs = costs[:, :len(price_returns)]

    # Strategy returns (broadcast price_returns across all combos)
    strategy_returns = aligned_signals * price_returns[np.newaxis, :aligned_signals.shape[1]]
    strategy_returns = strategy_returns - costs[:, :strategy_returns.shape[1]]

    # Compute metrics for each combo
    sharpe_ratios = np.zeros(n_combos)
    total_returns = np.zeros(n_combos)

    for i in range(n_combos):
        ret = strategy_returns[i]
        equity = np.cumprod(1 + ret)
        total_returns[i] = equity[-1] - 1
        daily_std = np.std(ret)
        sharpe_ratios[i] = np.mean(ret) / max(daily_std, 1e-10) * np.sqrt(252)

    return strategy_returns, np.column_stack([total_returns, sharpe_ratios])


# ============================================================================
# JAX Implementation (GPU-accelerated)
# ============================================================================

_jax_available = False
_jax_compute_returns = None
_jax_compute_metrics = None
_jax_batch_backtest = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap

    _jax_available = True

    @jit
    def _jax_compute_returns_impl(
        prices: jnp.ndarray,
        signals: jnp.ndarray,
        slippage_bps: float,
        commission_pct: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JIT-compiled return computation."""
        # Price returns
        price_returns = jnp.diff(prices) / prices[:-1]

        # Align signals
        aligned_signals = signals[:-1]

        # Trades
        padded = jnp.concatenate([jnp.zeros(1), aligned_signals])
        trades = jnp.diff(padded)

        # Costs
        costs = jnp.abs(trades) * (slippage_bps / 10000 + commission_pct)

        # Strategy returns
        n = min(len(aligned_signals), len(price_returns), len(costs))
        strategy_returns = aligned_signals[:n] * price_returns[:n] - costs[:n]

        # Equity curve
        equity_multiplier = jnp.cumprod(1 + strategy_returns)
        equity_curve = jnp.concatenate([jnp.ones(1), equity_multiplier])

        return strategy_returns, equity_curve, trades

    @jit
    def _jax_compute_metrics_impl(
        returns: jnp.ndarray,
        equity_curve: jnp.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """JIT-compiled metrics computation."""
        # Total return
        total_return = equity_curve[-1] / equity_curve[0] - 1

        # Annual return
        n_years = len(returns) / 252.0
        annual_return = jnp.power(1 + total_return, 1 / jnp.maximum(n_years, 0.01)) - 1

        # Sharpe ratio
        mean_ret = jnp.mean(returns)
        std_ret = jnp.std(returns)
        sharpe = mean_ret / jnp.maximum(std_ret, 1e-10) * jnp.sqrt(252)

        # Max drawdown
        running_max = jnp.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / running_max - 1
        max_dd = jnp.min(drawdowns)

        # Sortino (simplified - full downside computation not JIT-friendly)
        negative_mask = returns < 0
        downside_var = jnp.sum(jnp.where(negative_mask, returns ** 2, 0)) / jnp.maximum(jnp.sum(negative_mask), 1)
        downside_std = jnp.sqrt(downside_var)
        sortino = mean_ret / jnp.maximum(downside_std, 1e-10) * jnp.sqrt(252)

        return total_return, annual_return, sharpe, sortino, max_dd

    @jit
    def _jax_single_backtest(
        prices: jnp.ndarray,
        signals: jnp.ndarray,
        slippage_bps: float,
        commission_pct: float,
    ) -> Tuple[float, float]:
        """Single backtest returning (total_return, sharpe)."""
        returns, equity, _ = _jax_compute_returns_impl(prices, signals, slippage_bps, commission_pct)
        total_ret, _, sharpe, _, _ = _jax_compute_metrics_impl(returns, equity)
        return total_ret, sharpe

    # Vectorized batch backtest using vmap
    _jax_batch_backtest_impl = vmap(
        lambda signals, prices, slip, comm: _jax_single_backtest(prices, signals, slip, comm),
        in_axes=(0, None, None, None),
    )

    def jax_compute_returns(
        prices: np.ndarray,
        signals: np.ndarray,
        slippage_bps: float = 5.0,
        commission_pct: float = 0.001,
    ) -> BacktestArrays:
        """JAX-accelerated return computation."""
        jax_prices = jnp.array(prices)
        jax_signals = jnp.array(signals)

        returns, equity, trades = _jax_compute_returns_impl(
            jax_prices, jax_signals, slippage_bps, commission_pct
        )

        # Convert back to numpy
        return BacktestArrays(
            returns=np.array(returns),
            equity_curve=np.array(equity),
            signals=np.array(jax_signals[:-1]),
            trades=np.array(trades),
            costs=np.abs(np.array(trades)) * (slippage_bps / 10000 + commission_pct),
        )

    def jax_compute_metrics(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        periods_per_year: int = 252,
    ) -> MetricsResult:
        """JAX-accelerated metrics computation."""
        jax_returns = jnp.array(returns)
        jax_equity = jnp.array(equity_curve)

        total_ret, annual_ret, sharpe, sortino, max_dd = _jax_compute_metrics_impl(
            jax_returns, jax_equity
        )

        # Compute remaining metrics with NumPy (not performance-critical)
        volatility = float(np.std(returns) * np.sqrt(periods_per_year))
        win_rate = float(np.mean(returns > 0))
        gains = float(np.sum(returns[returns > 0]))
        losses = float(np.abs(np.sum(returns[returns < 0])))
        profit_factor = gains / max(losses, 1e-10)
        calmar = float(annual_ret) / max(abs(float(max_dd)), 1e-10)

        return MetricsResult(
            total_return=float(total_ret),
            annual_return=float(annual_ret),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=calmar,
            max_drawdown=float(max_dd),
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def jax_batch_backtest(
        prices: np.ndarray,
        signal_batches: np.ndarray,
        slippage_bps: float = 5.0,
        commission_pct: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run batch backtests using JAX vmap for GPU parallelization.

        Args:
            prices: Close price array (n_periods,)
            signal_batches: Signal arrays (n_combos, n_periods)
            slippage_bps: Slippage in basis points
            commission_pct: Commission percentage

        Returns:
            Tuple of (total_returns, sharpe_ratios) arrays
        """
        jax_prices = jnp.array(prices)
        jax_signals = jnp.array(signal_batches)

        results = _jax_batch_backtest_impl(jax_signals, jax_prices, slippage_bps, commission_pct)

        total_returns = np.array(results[0])
        sharpe_ratios = np.array(results[1])

        return total_returns, sharpe_ratios

    _jax_compute_returns = jax_compute_returns
    _jax_compute_metrics = jax_compute_metrics
    _jax_batch_backtest = jax_batch_backtest

except ImportError:
    pass


# ============================================================================
# Numba Implementation (CPU-optimized)
# ============================================================================

_numba_available = False
_numba_compute_returns = None
_numba_compute_metrics = None

try:
    import numba
    from numba import njit, prange

    _numba_available = True

    @njit(cache=True, fastmath=True)
    def _numba_compute_returns_impl(
        prices: np.ndarray,
        signals: np.ndarray,
        slippage_bps: float,
        commission_pct: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-compiled return computation."""
        n = len(prices) - 1

        # Pre-allocate arrays
        price_returns = np.empty(n)
        strategy_returns = np.empty(n)
        trades = np.empty(n)
        equity_curve = np.empty(n + 1)
        equity_curve[0] = 1.0

        prev_signal = 0.0
        equity = 1.0

        for i in range(n):
            # Price return
            price_returns[i] = (prices[i + 1] - prices[i]) / prices[i]

            # Current signal
            sig = signals[i] if i < len(signals) else 0.0

            # Trade (position change)
            trade = sig - prev_signal
            trades[i] = trade

            # Costs
            cost = abs(trade) * (slippage_bps / 10000 + commission_pct)

            # Strategy return
            strategy_returns[i] = sig * price_returns[i] - cost

            # Update equity
            equity *= (1 + strategy_returns[i])
            equity_curve[i + 1] = equity

            prev_signal = sig

        return strategy_returns, equity_curve, trades

    @njit(cache=True, fastmath=True)
    def _numba_compute_metrics_impl(
        returns: np.ndarray,
        equity_curve: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """Numba-compiled metrics computation."""
        n = len(returns)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Total return
        total_return = equity_curve[-1] / equity_curve[0] - 1

        # Annual return
        n_years = n / 252.0
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # Mean and std
        mean_ret = 0.0
        for i in range(n):
            mean_ret += returns[i]
        mean_ret /= n

        variance = 0.0
        for i in range(n):
            variance += (returns[i] - mean_ret) ** 2
        variance /= n
        std_ret = variance ** 0.5

        # Sharpe ratio
        sharpe = mean_ret / max(std_ret, 1e-10) * (252 ** 0.5)

        # Downside std for Sortino
        downside_sum = 0.0
        downside_count = 0
        for i in range(n):
            if returns[i] < 0:
                downside_sum += returns[i] ** 2
                downside_count += 1

        downside_std = 1e-10
        if downside_count > 0:
            downside_std = (downside_sum / downside_count) ** 0.5

        sortino = mean_ret / max(downside_std, 1e-10) * (252 ** 0.5)

        # Max drawdown
        running_max = equity_curve[0]
        max_dd = 0.0
        for i in range(len(equity_curve)):
            if equity_curve[i] > running_max:
                running_max = equity_curve[i]
            dd = equity_curve[i] / running_max - 1
            if dd < max_dd:
                max_dd = dd

        # Calmar
        calmar = annual_return / max(abs(max_dd), 1e-10)

        # Win rate and profit factor
        wins = 0
        gains = 0.0
        losses = 0.0
        for i in range(n):
            if returns[i] > 0:
                wins += 1
                gains += returns[i]
            else:
                losses += abs(returns[i])

        win_rate = wins / n
        profit_factor = gains / max(losses, 1e-10)

        return total_return, annual_return, sharpe, sortino, max_dd, calmar, win_rate, profit_factor

    @njit(cache=True, fastmath=True, parallel=True)
    def _numba_batch_backtest_impl(
        prices: np.ndarray,
        signal_batches: np.ndarray,
        slippage_bps: float,
        commission_pct: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-compiled parallel batch backtest."""
        n_combos = signal_batches.shape[0]
        total_returns = np.empty(n_combos)
        sharpe_ratios = np.empty(n_combos)

        for i in prange(n_combos):
            returns, equity, _ = _numba_compute_returns_impl(
                prices, signal_batches[i], slippage_bps, commission_pct
            )
            metrics = _numba_compute_metrics_impl(returns, equity)
            total_returns[i] = metrics[0]  # total_return
            sharpe_ratios[i] = metrics[2]  # sharpe

        return total_returns, sharpe_ratios

    def numba_compute_returns(
        prices: np.ndarray,
        signals: np.ndarray,
        slippage_bps: float = 5.0,
        commission_pct: float = 0.001,
    ) -> BacktestArrays:
        """Numba-accelerated return computation."""
        returns, equity, trades = _numba_compute_returns_impl(
            prices.astype(np.float64),
            signals.astype(np.float64),
            slippage_bps,
            commission_pct,
        )

        return BacktestArrays(
            returns=returns,
            equity_curve=equity,
            signals=signals[:-1] if len(signals) == len(prices) else signals,
            trades=trades,
            costs=np.abs(trades) * (slippage_bps / 10000 + commission_pct),
        )

    def numba_compute_metrics(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        periods_per_year: int = 252,
    ) -> MetricsResult:
        """Numba-accelerated metrics computation."""
        metrics = _numba_compute_metrics_impl(
            returns.astype(np.float64),
            equity_curve.astype(np.float64),
        )

        return MetricsResult(
            total_return=metrics[0],
            annual_return=metrics[1],
            sharpe_ratio=metrics[2],
            sortino_ratio=metrics[3],
            max_drawdown=metrics[4],
            calmar_ratio=metrics[5],
            win_rate=metrics[6],
            profit_factor=metrics[7],
            volatility=float(np.std(returns) * np.sqrt(periods_per_year)),
        )

    def numba_batch_backtest(
        prices: np.ndarray,
        signal_batches: np.ndarray,
        slippage_bps: float = 5.0,
        commission_pct: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated parallel batch backtest."""
        return _numba_batch_backtest_impl(
            prices.astype(np.float64),
            signal_batches.astype(np.float64),
            slippage_bps,
            commission_pct,
        )

    _numba_compute_returns = numba_compute_returns
    _numba_compute_metrics = numba_compute_metrics

except ImportError:
    pass


# ============================================================================
# Backend Selection
# ============================================================================

def get_compute_backend(backend: str = "auto") -> str:
    """
    Get the best available compute backend.

    Args:
        backend: Requested backend ("auto", "jax", "numba", "numpy")

    Returns:
        Selected backend name
    """
    if backend == "auto":
        if _jax_available:
            return "jax"
        elif _numba_available:
            return "numba"
        else:
            return "numpy"

    if backend == "jax" and _jax_available:
        return "jax"
    elif backend == "numba" and _numba_available:
        return "numba"
    else:
        return "numpy"


def compute_returns(
    prices: np.ndarray,
    signals: np.ndarray,
    slippage_bps: float = 5.0,
    commission_pct: float = 0.001,
    backend: str = "auto",
) -> BacktestArrays:
    """
    Compute strategy returns using the best available backend.

    Args:
        prices: Close price array
        signals: Position signals (-1 to 1)
        slippage_bps: Slippage in basis points
        commission_pct: Commission as percentage
        backend: Compute backend ("auto", "jax", "numba", "numpy")

    Returns:
        BacktestArrays with computed values
    """
    selected = get_compute_backend(backend)

    if selected == "jax" and _jax_compute_returns is not None:
        return _jax_compute_returns(prices, signals, slippage_bps, commission_pct)
    elif selected == "numba" and _numba_compute_returns is not None:
        return _numba_compute_returns(prices, signals, slippage_bps, commission_pct)
    else:
        return numpy_compute_returns(prices, signals, slippage_bps, commission_pct)


def compute_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = 252,
    backend: str = "auto",
) -> MetricsResult:
    """
    Compute performance metrics using the best available backend.

    Args:
        returns: Strategy return array
        equity_curve: Equity curve array
        periods_per_year: Trading periods per year
        backend: Compute backend

    Returns:
        MetricsResult with computed metrics
    """
    selected = get_compute_backend(backend)

    if selected == "jax" and _jax_compute_metrics is not None:
        return _jax_compute_metrics(returns, equity_curve, periods_per_year)
    elif selected == "numba" and _numba_compute_metrics is not None:
        return _numba_compute_metrics(returns, equity_curve, periods_per_year)
    else:
        return numpy_compute_metrics(returns, equity_curve, periods_per_year)


def batch_backtest(
    prices: np.ndarray,
    signal_batches: np.ndarray,
    slippage_bps: float = 5.0,
    commission_pct: float = 0.001,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run batch backtests using the best available backend.

    Uses JAX vmap for GPU parallelization or Numba prange for CPU parallelization.

    Args:
        prices: Close price array (n_periods,)
        signal_batches: Signal arrays (n_combos, n_periods)
        slippage_bps: Slippage in basis points
        commission_pct: Commission percentage
        backend: Compute backend

    Returns:
        Tuple of (total_returns, sharpe_ratios) arrays
    """
    selected = get_compute_backend(backend)

    if selected == "jax" and _jax_batch_backtest is not None:
        return _jax_batch_backtest(prices, signal_batches, slippage_bps, commission_pct)
    elif selected == "numba" and _numba_available:
        return _numba_batch_backtest_impl(
            prices.astype(np.float64),
            signal_batches.astype(np.float64),
            slippage_bps,
            commission_pct,
        )
    else:
        return numpy_batch_backtest(prices, signal_batches, slippage_bps, commission_pct)


def is_jax_available() -> bool:
    """Check if JAX is available."""
    return _jax_available


def is_numba_available() -> bool:
    """Check if Numba is available."""
    return _numba_available
