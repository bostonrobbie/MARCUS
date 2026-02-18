"""
Tests for VectorEngine and vectorized strategies.

These tests verify that the vectorized backtesting engine returns
consistent pandas types regardless of GPU acceleration status.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestVectorEngine:
    """Tests for VectorEngine class."""

    def test_run_returns_pandas_types(self, sample_ohlcv_data, mock_gpu_unavailable):
        """VectorEngine.run() should always return pandas types."""
        from backtesting.vector_engine import VectorEngine, VectorizedMA

        strategy = VectorizedMA(short_window=5, long_window=20)
        engine = VectorEngine(strategy, initial_capital=100000.0)

        result = engine.run(sample_ohlcv_data)

        assert isinstance(result['equity_curve'], pd.Series), "equity_curve should be pandas Series"
        assert isinstance(result['signals'], pd.Series), "signals should be pandas Series"
        assert isinstance(result['returns'], pd.Series), "returns should be pandas Series"
        assert isinstance(result['turnover'], pd.Series), "turnover should be pandas Series"

    def test_equity_curve_has_iloc(self, sample_ohlcv_data, mock_gpu_unavailable):
        """Equity curve should support .iloc accessor (Bug 1 regression test)."""
        from backtesting.vector_engine import VectorEngine, VectorizedMA

        strategy = VectorizedMA(short_window=5, long_window=20)
        engine = VectorEngine(strategy, initial_capital=100000.0)

        result = engine.run(sample_ohlcv_data)

        # This should NOT raise AttributeError
        final_equity = result['equity_curve'].iloc[-1]
        assert isinstance(final_equity, (int, float, np.number))

    def test_initial_capital_preserved(self, sample_ohlcv_data, mock_gpu_unavailable):
        """Initial capital should be used as starting point."""
        from backtesting.vector_engine import VectorEngine, VectorizedMA

        initial_capital = 50000.0
        strategy = VectorizedMA(short_window=5, long_window=20)
        engine = VectorEngine(strategy, initial_capital=initial_capital)

        result = engine.run(sample_ohlcv_data)

        # First equity value should be close to initial capital
        first_equity = result['equity_curve'].iloc[0]
        # Allow small deviation due to first-bar returns
        assert abs(first_equity - initial_capital) / initial_capital < 0.1

    def test_cost_modeling(self, sample_ohlcv_data, mock_gpu_unavailable):
        """Transaction costs should be applied correctly."""
        from backtesting.vector_engine import VectorEngine, VectorizedMA

        strategy = VectorizedMA(short_window=5, long_window=20)

        # With high costs
        engine_with_costs = VectorEngine(
            strategy,
            initial_capital=100000.0,
            commission=10.0,
            slippage=5.0,
            volatility_factor=0.05
        )
        result_with_costs = engine_with_costs.run(sample_ohlcv_data)

        # Without costs
        engine_no_costs = VectorEngine(
            strategy,
            initial_capital=100000.0,
            commission=0.0,
            slippage=0.0,
            volatility_factor=0.0
        )
        result_no_costs = engine_no_costs.run(sample_ohlcv_data)

        # Equity with costs should be lower (or equal if no trades)
        final_with = result_with_costs['equity_curve'].iloc[-1]
        final_without = result_no_costs['equity_curve'].iloc[-1]

        assert final_with <= final_without

    def test_signals_match_data_length(self, sample_ohlcv_data, mock_gpu_unavailable):
        """Signals should have same length as input data."""
        from backtesting.vector_engine import VectorEngine, VectorizedMA

        strategy = VectorizedMA(short_window=5, long_window=20)
        engine = VectorEngine(strategy, initial_capital=100000.0)

        result = engine.run(sample_ohlcv_data)

        assert len(result['signals']) == len(sample_ohlcv_data)
        assert len(result['equity_curve']) == len(sample_ohlcv_data)
        assert len(result['returns']) == len(sample_ohlcv_data)


class TestVectorizedMA:
    """Tests for VectorizedMA strategy."""

    def test_generate_signals_returns_series(self, sample_ohlcv_data, mock_gpu_unavailable):
        """generate_signals should return a pandas Series."""
        from backtesting.vector_engine import VectorizedMA

        strategy = VectorizedMA(short_window=5, long_window=20)
        signals = strategy.generate_signals(sample_ohlcv_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_data)

    def test_signals_are_valid_values(self, sample_ohlcv_data, mock_gpu_unavailable):
        """Signals should only be 0 or 1 for MA crossover."""
        from backtesting.vector_engine import VectorizedMA

        strategy = VectorizedMA(short_window=5, long_window=20)
        signals = strategy.generate_signals(sample_ohlcv_data)

        unique_values = set(signals.unique())
        valid_values = {0, 1}  # MA strategy uses 0 and 1

        assert unique_values.issubset(valid_values)

    def test_parameter_storage(self):
        """Strategy should store parameters correctly."""
        from backtesting.vector_engine import VectorizedMA

        strategy = VectorizedMA(short_window=10, long_window=50)

        assert strategy.short_window == 10
        assert strategy.long_window == 50
        assert strategy.params['short_window'] == 10
        assert strategy.params['long_window'] == 50


class TestVectorizedNQORB:
    """Tests for VectorizedNQORB strategy."""

    def test_generate_signals_shape(self, multi_day_ohlcv_data, mock_gpu_unavailable):
        """Signals should match input data length."""
        from backtesting.vector_engine import VectorizedNQORB

        strategy = VectorizedNQORB(
            orb_start="09:30",
            orb_end="09:45",
            ema_filter=20,
            atr_filter=14
        )

        signals = strategy.generate_signals(multi_day_ohlcv_data)

        assert len(signals) == len(multi_day_ohlcv_data)
        assert isinstance(signals, pd.Series)

    def test_signals_are_valid_values(self, multi_day_ohlcv_data, mock_gpu_unavailable):
        """Signals should only be -1, 0, or 1."""
        from backtesting.vector_engine import VectorizedNQORB

        strategy = VectorizedNQORB()
        signals = strategy.generate_signals(multi_day_ohlcv_data)

        unique_values = set(signals.unique())
        valid_values = {-1, 0, 1}

        assert unique_values.issubset(valid_values)

    def test_orb_parameters(self):
        """ORB strategy should store parameters correctly."""
        from backtesting.vector_engine import VectorizedNQORB

        strategy = VectorizedNQORB(
            orb_start="09:30",
            orb_end="10:00",
            ema_filter=100,
            sl_atr_mult=1.5,
            tp_atr_mult=3.0
        )

        assert strategy.orb_start == "09:30"
        assert strategy.orb_end == "10:00"
        assert strategy.ema_filter == 100
        assert strategy.sl_atr_mult == 1.5
        assert strategy.tp_atr_mult == 3.0
