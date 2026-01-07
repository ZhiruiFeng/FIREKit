"""
Hybrid Runner

Automatically selects between vectorized and event-driven modes
based on strategy requirements and optimization goals.

v0.2.0 Features:
- Intelligent strategy analysis
- Complexity estimation
- Adaptive execution mode switching
- Result discrepancy detection
"""

from __future__ import annotations

import inspect
import ast
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vectorforge.engine.base import BacktestEngine, BacktestResult
from vectorforge.engine.vectorized import VectorizedBacktester
from vectorforge.engine.event_driven import EventDrivenBacktester

if TYPE_CHECKING:
    from vectorforge.strategy.base import BaseStrategy
    from vectorforge.config import VectorForgeConfig


class StrategyComplexity(Enum):
    """Strategy complexity levels."""
    SIMPLE = "simple"           # Pure signal generation, no state
    MODERATE = "moderate"       # Some state, but vectorizable
    COMPLEX = "complex"         # Requires event-driven execution
    UNKNOWN = "unknown"


@dataclass
class StrategyAnalysis:
    """Results of strategy capability analysis."""
    supports_vectorized: bool
    supports_event_driven: bool
    complexity: StrategyComplexity
    estimated_operations: int
    requires_order_logic: bool
    has_state: bool
    uses_position_sizing: bool
    recommended_mode: str
    confidence: float  # 0-1 confidence in recommendation
    notes: list[str]


class HybridRunner(BacktestEngine):
    """
    Hybrid backtesting runner with intelligent mode selection.

    Uses vectorized mode for:
    - Initial parameter sweeps and optimization
    - Strategies that support array-based signal generation
    - When speed is prioritized over execution accuracy

    Uses event-driven mode for:
    - Final validation before live deployment
    - Strategies requiring order-level logic
    - When realistic execution modeling is required

    v0.2.0 Features:
    - Automatic strategy analysis and complexity detection
    - Adaptive execution that starts vectorized and validates with event-driven
    - Discrepancy detection between modes
    - Performance optimization recommendations

    Example:
        >>> runner = HybridRunner()
        >>> # Analyze strategy capabilities
        >>> analysis = runner.analyze_strategy(my_strategy)
        >>> print(f"Recommended mode: {analysis.recommended_mode}")
        >>>
        >>> # Auto-selects optimal mode for parameter sweep
        >>> results = runner.run_batch(
        ...     strategy_class=MomentumStrategy,
        ...     param_grid={"lookback": range(10, 50)},
        ...     data=data
        ... )
        >>> # Adaptive run with automatic validation
        >>> result = runner.run_adaptive(strategy, data)
    """

    def __init__(self, config: VectorForgeConfig | None = None):
        super().__init__(config)
        self.vectorized = VectorizedBacktester(config)
        self.event_driven = EventDrivenBacktester(config)
        self._strategy_cache: dict[type, StrategyAnalysis] = {}
        self._discrepancy_threshold = 0.1  # 10% return difference triggers warning

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
        mode: str | None = None,
    ) -> BacktestResult:
        """
        Run backtest with automatic or specified mode.

        Args:
            strategy: Trading strategy
            data: OHLCV DataFrame
            initial_capital: Starting capital
            mode: Force "vectorized" or "event_driven", or None for auto

        Returns:
            BacktestResult
        """
        if mode is None:
            mode = self._select_mode(strategy)

        if mode == "vectorized":
            return self.vectorized.run(strategy, data, initial_capital)
        else:
            return self.event_driven.run(strategy, data, initial_capital)

    def run_batch(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> list[BacktestResult]:
        """
        Run parameter sweep using vectorized mode for speed.

        Args:
            strategy_class: Strategy class to test
            param_grid: Parameter combinations
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            List of BacktestResult
        """
        # Use vectorized for batch operations
        return self.vectorized.run_batch(
            strategy_class, param_grid, data, initial_capital
        )

    def validate(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        """
        Validate strategy using event-driven mode.

        Use this for final validation before live deployment.

        Args:
            strategy: Strategy to validate
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            BacktestResult with realistic execution modeling
        """
        return self.event_driven.run(strategy, data, initial_capital)

    def compare_modes(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Run strategy in both modes and compare results.

        Useful for understanding the gap between idealized
        vectorized results and realistic event-driven execution.

        Args:
            strategy: Strategy to test
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            Dict with "vectorized" and "event_driven" results
        """
        vectorized_result = self.vectorized.run(strategy, data, initial_capital)
        event_driven_result = self.event_driven.run(strategy, data, initial_capital)

        return {
            "vectorized": vectorized_result,
            "event_driven": event_driven_result,
            "sharpe_gap": vectorized_result.sharpe_ratio - event_driven_result.sharpe_ratio,
            "return_gap": vectorized_result.total_return - event_driven_result.total_return,
        }

    def run_adaptive(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
        validate: bool = True,
    ) -> BacktestResult:
        """
        Run adaptive backtest with automatic mode selection and validation.

        Starts with vectorized mode for speed, then optionally validates
        with event-driven mode to check for discrepancies.

        Args:
            strategy: Trading strategy
            data: OHLCV DataFrame
            initial_capital: Starting capital
            validate: If True, also runs event-driven validation

        Returns:
            BacktestResult (from event-driven if validate=True)
        """
        analysis = self.analyze_strategy(strategy)

        if analysis.recommended_mode == "event_driven" or not analysis.supports_vectorized:
            # Must use event-driven
            return self.event_driven.run(strategy, data, initial_capital)

        # Start with vectorized
        vec_result = self.vectorized.run(strategy, data, initial_capital)

        if not validate:
            return vec_result

        # Validate with event-driven
        ed_result = self.event_driven.run(strategy, data, initial_capital)

        # Check for discrepancies
        return_gap = abs(vec_result.total_return - ed_result.total_return)
        if return_gap > self._discrepancy_threshold:
            # Log warning about significant discrepancy
            self._last_discrepancy = {
                "vectorized_return": vec_result.total_return,
                "event_driven_return": ed_result.total_return,
                "gap": return_gap,
            }

        # Return the more realistic event-driven result
        return ed_result

    def analyze_strategy(self, strategy: BaseStrategy) -> StrategyAnalysis:
        """
        Analyze strategy to determine capabilities and recommended mode.

        Examines:
        - Available methods (generate_signals, on_bar, on_fill)
        - Source code for state usage and order logic
        - Method signatures and complexity

        Args:
            strategy: Strategy instance to analyze

        Returns:
            StrategyAnalysis with recommendations
        """
        strategy_class = type(strategy)

        # Check cache
        if strategy_class in self._strategy_cache:
            return self._strategy_cache[strategy_class]

        notes: list[str] = []

        # Check method availability
        has_generate_signals = hasattr(strategy, "generate_signals") and callable(
            getattr(strategy, "generate_signals")
        )
        has_on_bar = hasattr(strategy, "on_bar") and callable(
            getattr(strategy, "on_bar")
        )
        has_on_fill = hasattr(strategy, "on_fill") and callable(
            getattr(strategy, "on_fill")
        )

        supports_vectorized = has_generate_signals
        supports_event_driven = has_on_bar

        if has_generate_signals:
            notes.append("Has generate_signals method - vectorizable")
        if has_on_bar:
            notes.append("Has on_bar method - event-driven compatible")
        if has_on_fill:
            notes.append("Has on_fill handler - may need event-driven for fill logic")

        # Analyze source code for complexity indicators
        requires_order_logic = False
        has_state = False
        uses_position_sizing = False
        estimated_operations = 0

        try:
            source = inspect.getsource(strategy_class)
            tree = ast.parse(source)

            for node in ast.walk(tree):
                # Count operations for complexity estimation
                if isinstance(node, (ast.BinOp, ast.Compare, ast.BoolOp)):
                    estimated_operations += 1

                # Check for state usage (self.xxx assignments)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == "self":
                                has_state = True

                # Check for order-related keywords
                if isinstance(node, ast.Name):
                    name = node.id.lower()
                    if name in ("order", "limit", "stop", "trailing"):
                        requires_order_logic = True
                    if name in ("position", "size", "quantity", "shares"):
                        uses_position_sizing = True

                # Check for string literals mentioning order types
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value.lower()
                    if any(kw in val for kw in ("limit", "stop", "bracket", "oco")):
                        requires_order_logic = True

        except (OSError, TypeError):
            notes.append("Could not analyze source code")

        # Determine complexity
        if requires_order_logic or (has_on_fill and has_state):
            complexity = StrategyComplexity.COMPLEX
        elif has_state or uses_position_sizing:
            complexity = StrategyComplexity.MODERATE
        elif supports_vectorized:
            complexity = StrategyComplexity.SIMPLE
        else:
            complexity = StrategyComplexity.UNKNOWN

        # Determine recommended mode
        confidence = 0.5

        if requires_order_logic:
            recommended_mode = "event_driven"
            confidence = 0.9
            notes.append("Requires event-driven for order logic")
        elif supports_vectorized and not has_state:
            recommended_mode = "vectorized"
            confidence = 0.9
            notes.append("Simple vectorizable strategy")
        elif supports_vectorized and has_state:
            recommended_mode = "vectorized"
            confidence = 0.6
            notes.append("Has state but may work vectorized - validate with event-driven")
        elif supports_event_driven:
            recommended_mode = "event_driven"
            confidence = 0.8
            notes.append("Event-driven only strategy")
        else:
            recommended_mode = "vectorized"
            confidence = 0.3
            notes.append("Unknown capabilities - defaulting to vectorized")

        analysis = StrategyAnalysis(
            supports_vectorized=supports_vectorized,
            supports_event_driven=supports_event_driven,
            complexity=complexity,
            estimated_operations=estimated_operations,
            requires_order_logic=requires_order_logic,
            has_state=has_state,
            uses_position_sizing=uses_position_sizing,
            recommended_mode=recommended_mode,
            confidence=confidence,
            notes=notes,
        )

        # Cache result
        self._strategy_cache[strategy_class] = analysis

        return analysis

    def _select_mode(self, strategy: BaseStrategy) -> str:
        """
        Automatically select execution mode based on strategy analysis.

        Returns:
            "vectorized" or "event_driven"
        """
        analysis = self.analyze_strategy(strategy)
        return analysis.recommended_mode

    def get_discrepancy_report(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate detailed comparison report between execution modes.

        Useful for understanding how execution modeling affects results.

        Args:
            strategy: Strategy to test
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            Detailed comparison report
        """
        analysis = self.analyze_strategy(strategy)

        if not analysis.supports_vectorized:
            return {
                "error": "Strategy does not support vectorized mode",
                "analysis": analysis,
            }

        vec_result = self.vectorized.run(strategy, data, initial_capital)
        ed_result = self.event_driven.run(strategy, data, initial_capital)

        # Compute discrepancy metrics
        return_gap = vec_result.total_return - ed_result.total_return
        sharpe_gap = vec_result.sharpe_ratio - ed_result.sharpe_ratio
        dd_gap = vec_result.max_drawdown - ed_result.max_drawdown

        # Compute correlation of equity curves
        vec_equity = vec_result.equity_curve.values
        ed_equity = ed_result.equity_curve.values
        min_len = min(len(vec_equity), len(ed_equity))
        correlation = float(np.corrcoef(vec_equity[:min_len], ed_equity[:min_len])[0, 1])

        return {
            "strategy_analysis": analysis,
            "vectorized_result": vec_result,
            "event_driven_result": ed_result,
            "discrepancies": {
                "total_return_gap": return_gap,
                "total_return_gap_pct": return_gap / max(abs(ed_result.total_return), 1e-10) * 100,
                "sharpe_gap": sharpe_gap,
                "max_drawdown_gap": dd_gap,
                "equity_curve_correlation": correlation,
            },
            "recommendations": self._get_discrepancy_recommendations(
                return_gap, sharpe_gap, correlation, analysis
            ),
        }

    def _get_discrepancy_recommendations(
        self,
        return_gap: float,
        sharpe_gap: float,
        correlation: float,
        analysis: StrategyAnalysis,
    ) -> list[str]:
        """Generate recommendations based on discrepancy analysis."""
        recommendations = []

        if abs(return_gap) > 0.1:
            recommendations.append(
                "Large return gap detected - vectorized may be too optimistic. "
                "Use event-driven for realistic estimates."
            )

        if abs(sharpe_gap) > 0.5:
            recommendations.append(
                "Significant Sharpe ratio difference - execution costs may be "
                "underestimated in vectorized mode."
            )

        if correlation < 0.95:
            recommendations.append(
                "Low equity curve correlation - strategy may have path-dependent "
                "behavior that vectorized mode cannot capture."
            )

        if analysis.has_state:
            recommendations.append(
                "Strategy uses internal state - event-driven mode recommended "
                "for accurate simulation."
            )

        if not recommendations:
            recommendations.append(
                "Results are consistent between modes - vectorized is safe to use "
                "for optimization, but always validate final strategy with event-driven."
            )

        return recommendations
