"""
Tests for VectorForge optimization tools.
"""

import numpy as np
import pandas as pd
import pytest

from vectorforge import VectorizedBacktester
from vectorforge.strategy.base import MomentumStrategy
from vectorforge.optimization.grid_search import GridSearch
from vectorforge.optimization.walk_forward import WalkForwardOptimizer
from vectorforge.optimization.cross_validation import PurgedKFold, CombinatorialPurgedKFold


class TestGridSearch:
    """Tests for GridSearch optimizer."""

    @pytest.fixture
    def grid_search(self, config):
        """Create GridSearch instance."""
        engine = VectorizedBacktester(config)
        return GridSearch(engine)

    def test_basic_search(self, grid_search, sample_ohlcv_data):
        """Test basic grid search."""
        results = grid_search.search(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20, 30]},
            data=sample_ohlcv_data,
            metric="sharpe_ratio",
        )

        assert len(results) == 3
        assert all("params" in r for r in results)
        assert all("result" in r for r in results)
        assert all("sharpe_ratio" in r for r in results)

    def test_sorted_by_metric(self, grid_search, sample_ohlcv_data):
        """Test results are sorted by metric."""
        results = grid_search.search(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20, 30, 40, 50]},
            data=sample_ohlcv_data,
            metric="sharpe_ratio",
            ascending=False,
        )

        sharpes = [r["sharpe_ratio"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_with_filter(self, grid_search, sample_ohlcv_data):
        """Test search with filter function."""
        results = grid_search.search_with_filter(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20, 30, 40, 50]},
            data=sample_ohlcv_data,
            filter_func=lambda r: r.total_trades > 0,
        )

        # All results should pass filter
        assert all(r["result"].total_trades > 0 for r in results)


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    @pytest.fixture
    def wfo(self, config):
        """Create WalkForwardOptimizer instance."""
        engine = VectorizedBacktester(config)
        return WalkForwardOptimizer(
            engine=engine,
            train_period=100,
            test_period=30,
        )

    def test_basic_run(self, wfo, sample_ohlcv_data):
        """Test basic walk-forward optimization."""
        results = wfo.run(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20]},
            data=sample_ohlcv_data,
        )

        # Should have at least one period
        assert len(results) >= 1

        # Check result structure
        for r in results:
            assert hasattr(r, "train_sharpe")
            assert hasattr(r, "test_sharpe")
            assert hasattr(r, "degradation")
            assert hasattr(r, "best_params")

    def test_summary(self, wfo, sample_ohlcv_data):
        """Test summary generation."""
        results = wfo.run(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20]},
            data=sample_ohlcv_data,
        )

        summary = wfo.summary(results)

        if results:
            assert "avg_test_sharpe" in summary
            assert "consistency" in summary
            assert "efficiency_ratio" in summary


class TestPurgedKFold:
    """Tests for PurgedKFold cross-validation."""

    def test_basic_split(self, sample_ohlcv_data):
        """Test basic K-fold split."""
        cv = PurgedKFold(n_splits=5, embargo_period=5)

        splits = list(cv.split(sample_ohlcv_data))

        assert len(splits) == 5

        for train_idx, test_idx in splits:
            # Train should come before test
            assert train_idx.max() < test_idx.min()

            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_embargo_applied(self, sample_ohlcv_data):
        """Test that embargo is applied."""
        embargo = 10
        cv = PurgedKFold(n_splits=5, embargo_period=embargo)

        for train_idx, test_idx in cv.split(sample_ohlcv_data):
            gap = test_idx.min() - train_idx.max()
            assert gap >= embargo

    def test_n_splits(self):
        """Test n_splits property."""
        cv = PurgedKFold(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_invalid_n_splits(self):
        """Test that n_splits < 2 raises error."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)


class TestCombinatorialPurgedKFold:
    """Tests for CombinatorialPurgedKFold."""

    def test_basic_split(self, sample_ohlcv_data):
        """Test combinatorial split."""
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)

        splits = list(cv.split(sample_ohlcv_data))

        # Should have C(5,2) = 10 combinations
        assert len(splits) == 10

        for train_idx, test_idx in splits:
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_n_splits(self):
        """Test get_n_splits returns correct number of combinations."""
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)
        assert cv.get_n_splits() == 10  # C(5,2)

        cv2 = CombinatorialPurgedKFold(n_splits=6, n_test_splits=3)
        assert cv2.get_n_splits() == 20  # C(6,3)
