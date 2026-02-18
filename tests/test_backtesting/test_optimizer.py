"""
Tests for optimization classes (GridSearch, VectorizedGridSearch, WalkForward).

These tests verify that optimizers correctly handle different array types
and use safe_iloc for consistent behavior.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestRunSingleBacktest:
    """Tests for the _run_single_backtest worker function."""

    def test_returns_dict_with_params(self, sample_ohlcv_dict):
        """Should return dict containing the input parameters."""
        from backtesting.optimizer import _run_single_backtest
        from backtesting.strategy import Strategy

        # Create a mock strategy class
        class MockStrategy(Strategy):
            def __init__(self, data_handler, events, **params):
                super().__init__(data_handler, events)
                self.params = params

            def calculate_signals(self, event):
                pass

        params = {'param1': 10, 'param2': 20}
        args = (sample_ohlcv_dict, MockStrategy, params, 100000.0)

        result = _run_single_backtest(args)

        assert 'param1' in result
        assert result['param1'] == 10
        assert 'param2' in result
        assert result['param2'] == 20

    def test_handles_empty_equity_curve(self, sample_ohlcv_dict):
        """Should handle case where no trades are made."""
        from backtesting.optimizer import _run_single_backtest
        from backtesting.strategy import Strategy

        class NoTradeStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        params = {}
        args = (sample_ohlcv_dict, NoTradeStrategy, params, 100000.0)

        result = _run_single_backtest(args)

        # Should have default values, not crash
        assert 'Total Return' in result
        assert 'Final Equity' in result


class TestRunSingleVectorBacktest:
    """Tests for _run_single_vector_backtest worker function."""

    def test_handles_ndarray_equity_curve(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """Should handle numpy ndarray equity curves without AttributeError."""
        from backtesting.optimizer import _run_single_vector_backtest

        # Mock engine that returns ndarray
        class MockEngine:
            def __init__(self, strategy, initial_capital):
                self.initial_capital = initial_capital

            def run(self, df):
                return {
                    'equity_curve': np.array([100000, 100500, 101000]),
                    'signals': np.zeros(3),
                    'returns': np.zeros(3),
                    'turnover': np.zeros(3)
                }

        class MockStrategy:
            def __init__(self, **params):
                self.params = params

        args = (
            MockEngine,
            MockStrategy,
            {'test_param': 42},
            100000.0,
            sample_ohlcv_dict['NQ']
        )

        # Should NOT raise AttributeError: 'ndarray' object has no attribute 'iloc'
        result = _run_single_vector_backtest(args)

        assert 'Total Return' in result
        assert 'Final Equity' in result
        assert 'Error' not in result

    def test_handles_pandas_series_equity_curve(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """Should handle pandas Series equity curves."""
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

    def test_catches_exceptions(self, sample_ohlcv_dict):
        """Should catch exceptions and return error info."""
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
        assert result['Final Equity'] == 100000.0


class TestGridSearch:
    """Tests for GridSearch optimizer."""

    def test_param_combinations_generated(self):
        """Should generate all parameter combinations."""
        from backtesting.optimizer import GridSearch
        from backtesting.data import MemoryDataHandler
        from backtesting.strategy import Strategy

        class DummyStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        param_grid = {
            'short_window': [5, 10],
            'long_window': [20, 30]
        }

        optimizer = GridSearch(
            data_handler_cls=MemoryDataHandler,
            data_handler_args=({'NQ': pd.DataFrame()},),
            strategy_cls=DummyStrategy,
            param_grid=param_grid,
            initial_capital=100000.0
        )

        combos = optimizer._generate_param_combinations()

        assert len(combos) == 4  # 2 x 2
        assert {'short_window': 5, 'long_window': 20} in combos
        assert {'short_window': 10, 'long_window': 30} in combos


class TestVectorizedGridSearch:
    """Tests for VectorizedGridSearch optimizer."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        from backtesting.optimizer import VectorizedGridSearch
        from backtesting.data import MemoryDataHandler
        from backtesting.strategy import Strategy

        class DummyStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        param_grid = {'window': [10, 20, 30]}

        optimizer = VectorizedGridSearch(
            data_handler_cls=MemoryDataHandler,
            data_handler_args=({'NQ': pd.DataFrame()},),
            strategy_cls=DummyStrategy,
            param_grid=param_grid,
            initial_capital=50000.0,
            n_jobs=2
        )

        assert optimizer.initial_capital == 50000.0
        assert optimizer.n_jobs == 2
        assert optimizer.param_grid == param_grid
