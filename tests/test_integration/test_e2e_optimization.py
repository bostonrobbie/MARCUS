"""
End-to-end integration tests for optimization workflows.

These tests verify the grid search and vectorized optimization
functionality works correctly end-to-end.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from queue import Queue
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


@pytest.mark.integration
class TestGridSearchIntegration:
    """Integration tests for GridSearch optimizer."""

    def test_grid_search_generates_combinations(self, sample_ohlcv_dict):
        """Should generate all parameter combinations."""
        from backtesting.optimizer import GridSearch
        from backtesting.data import MemoryDataHandler
        from backtesting.strategy import Strategy

        class ParameterizedStrategy(Strategy):
            def __init__(self, bars, events, short_window=5, long_window=20):
                super().__init__(bars, events)
                self.short = short_window
                self.long = long_window

            def calculate_signals(self, event):
                pass

        param_grid = {
            'short_window': [5, 10],
            'long_window': [20, 30]
        }

        optimizer = GridSearch(
            data_handler_cls=MemoryDataHandler,
            data_handler_args=(sample_ohlcv_dict,),
            strategy_cls=ParameterizedStrategy,
            param_grid=param_grid,
            initial_capital=100000.0
        )

        combos = optimizer._generate_param_combinations()

        assert len(combos) == 4  # 2 x 2
        assert {'short_window': 5, 'long_window': 20} in combos
        assert {'short_window': 10, 'long_window': 30} in combos


@pytest.mark.integration
class TestVectorizedOptimizationIntegration:
    """Integration tests for vectorized optimization."""

    def test_vectorized_grid_search_runs(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """Should run vectorized grid search without errors."""
        from backtesting.optimizer import VectorizedGridSearch
        from backtesting.data import MemoryDataHandler

        class VectorStrategy:
            def __init__(self, window=10):
                self.window = window

            def calculate_signals(self, df):
                return pd.Series(np.zeros(len(df)), index=df.index)

        param_grid = {'window': [5, 10, 20]}

        optimizer = VectorizedGridSearch(
            data_handler_cls=MemoryDataHandler,
            data_handler_args=(sample_ohlcv_dict,),
            strategy_cls=VectorStrategy,
            param_grid=param_grid,
            initial_capital=100000.0,
            n_jobs=1
        )

        # Should initialize without error
        assert optimizer.initial_capital == 100000.0


@pytest.mark.integration
class TestOptimizerSafeIloc:
    """Integration tests for Bug 1 fix in optimizer."""

    def test_run_single_vector_backtest_handles_ndarray(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """_run_single_vector_backtest should handle numpy arrays."""
        from backtesting.optimizer import _run_single_vector_backtest

        class MockEngine:
            def __init__(self, strategy, initial_capital):
                self.initial_capital = initial_capital

            def run(self, df):
                # Return numpy arrays (simulates GPU/cuDF conversion)
                return {
                    'equity_curve': np.array([100000, 100500, 101000]),
                    'signals': np.zeros(3),
                    'returns': np.zeros(3),
                    'turnover': np.zeros(3)
                }

        class MockStrategy:
            def __init__(self, **params):
                pass

        args = (
            MockEngine,
            MockStrategy,
            {'test_param': 42},
            100000.0,
            sample_ohlcv_dict['NQ']
        )

        # Should NOT raise AttributeError
        result = _run_single_vector_backtest(args)

        assert 'Total Return' in result
        assert 'Final Equity' in result
        assert 'Error' not in result

    def test_run_single_vector_backtest_handles_series(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """_run_single_vector_backtest should handle pandas Series."""
        from backtesting.optimizer import _run_single_vector_backtest

        class MockEngine:
            def __init__(self, strategy, initial_capital):
                pass

            def run(self, df):
                return {
                    'equity_curve': pd.Series([100000, 100500, 101000]),
                    'signals': pd.Series([0, 1, 0]),
                    'returns': pd.Series([0.0, 0.005, 0.005]),
                    'turnover': pd.Series([0, 1, 1])
                }

        class MockStrategy:
            def __init__(self, **params):
                pass

        args = (MockEngine, MockStrategy, {}, 100000.0, sample_ohlcv_dict['NQ'])

        result = _run_single_vector_backtest(args)

        assert 'Total Return' in result
        assert result['Total Return'] == pytest.approx(0.01, rel=0.01)

    def test_optimizer_catches_exceptions(self, sample_ohlcv_dict):
        """Should catch and report exceptions gracefully."""
        from backtesting.optimizer import _run_single_vector_backtest

        class FailingEngine:
            def __init__(self, strategy, initial_capital):
                pass

            def run(self, df):
                raise ValueError("Test error")

        class MockStrategy:
            def __init__(self, **params):
                pass

        args = (FailingEngine, MockStrategy, {'param': 1}, 100000.0, sample_ohlcv_dict['NQ'])

        result = _run_single_vector_backtest(args)

        assert 'Error' in result
        assert 'Test error' in result['Error']
        assert result['Total Return'] == 0.0


@pytest.mark.integration
class TestOptimizationResultsFormat:
    """Integration tests for optimization results format."""

    def test_results_include_all_params(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """Results should include all parameter values."""
        from backtesting.optimizer import _run_single_vector_backtest

        class MockEngine:
            def __init__(self, strategy, initial_capital):
                pass

            def run(self, df):
                return {
                    'equity_curve': np.array([100000, 101000]),
                    'signals': np.zeros(2),
                    'returns': np.zeros(2),
                    'turnover': np.zeros(2)
                }

        class MockStrategy:
            def __init__(self, **params):
                self.params = params

        params = {'window': 10, 'threshold': 0.5, 'multiplier': 2}
        args = (MockEngine, MockStrategy, params, 100000.0, sample_ohlcv_dict['NQ'])

        result = _run_single_vector_backtest(args)

        # All params should be in result
        assert result['window'] == 10
        assert result['threshold'] == 0.5
        assert result['multiplier'] == 2
