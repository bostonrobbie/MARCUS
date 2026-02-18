"""
Tests for NQ Opening Range Breakout (ORB) strategy.

These tests verify the ORB strategy logic including range calculation,
signal generation, and risk management.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from queue import Queue
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'strategies'))


class TestNqOrbInitialization:
    """Tests for NqOrb strategy initialization."""

    def test_default_parameters(self, multi_day_ohlcv_data):
        """Should initialize with default parameters."""
        from backtesting.data import MemoryDataHandler
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        strategy = NqOrb(data_handler, events)

        assert strategy.orb_start == time(9, 30)
        assert strategy.orb_end == time(10, 0)
        assert strategy.exit_time == time(15, 55)
        assert strategy.stop_loss == 75.0
        assert strategy.take_profit == 100.0

    def test_custom_parameters(self, multi_day_ohlcv_data):
        """Should accept custom parameters."""
        from backtesting.data import MemoryDataHandler
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        strategy = NqOrb(
            data_handler, events,
            orb_start_time=time(9, 45),
            orb_end_time=time(10, 15),
            stop_loss=50.0,
            take_profit=150.0
        )

        assert strategy.orb_start == time(9, 45)
        assert strategy.orb_end == time(10, 15)
        assert strategy.stop_loss == 50.0
        assert strategy.take_profit == 150.0

    def test_initial_state(self, multi_day_ohlcv_data):
        """Should initialize state tracking variables."""
        from backtesting.data import MemoryDataHandler
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        strategy = NqOrb(data_handler, events)

        assert strategy.orb_high is None
        assert strategy.orb_low is None
        assert strategy.current_date is None
        assert strategy.traded_today is False
        assert strategy.in_long is False
        assert strategy.in_short is False


class TestNqOrbRangeCalculation:
    """Tests for ORB range calculation."""

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_range_builds_during_orb_period(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should track high and low during ORB period."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        # Load enough bars for indicators
        for _ in range(100):
            data_handler.update_bars()

        strategy = NqOrb(data_handler, events)

        # Simulate bars during ORB period
        ts = datetime(2024, 1, 15, 9, 30, 0)
        bar1 = Bar(symbol='NQ', timestamp=ts, open=17000, high=17050, low=16990, close=17025, volume=100)
        strategy.calculate_signals(bar1)

        ts2 = datetime(2024, 1, 15, 9, 45, 0)
        bar2 = Bar(symbol='NQ', timestamp=ts2, open=17025, high=17080, low=16970, close=17060, volume=100)
        strategy.calculate_signals(bar2)

        assert strategy.orb_high == 17080
        assert strategy.orb_low == 16970

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_range_resets_each_day(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should reset range for new trading day."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        strategy = NqOrb(data_handler, events)

        # Day 1
        bar1 = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 9, 30),
                   open=17000, high=17100, low=16950, close=17050, volume=100)
        strategy.calculate_signals(bar1)

        assert strategy.current_date == datetime(2024, 1, 15).date()

        # Day 2
        bar2 = Bar(symbol='NQ', timestamp=datetime(2024, 1, 16, 9, 30),
                   open=17200, high=17250, low=17180, close=17220, volume=100)
        strategy.calculate_signals(bar2)

        assert strategy.current_date == datetime(2024, 1, 16).date()
        assert strategy.orb_high == 17250
        assert strategy.orb_low == 17180
        assert strategy.traded_today is False


class TestNqOrbSignalGeneration:
    """Tests for ORB signal generation logic."""

    def test_no_signal_before_orb_end(self, multi_day_ohlcv_data):
        """Should not generate signals during ORB period."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        strategy = NqOrb(data_handler, events)

        # Bar during ORB period
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 9, 45),
                  open=17000, high=17100, low=16950, close=17050, volume=100)
        strategy.calculate_signals(bar)

        # Should have no signals
        assert events.empty()

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_long_signal_on_breakout(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should generate long signal when price breaks above range."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar, SignalType
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        # Mock indicators
        mock_ema.return_value = pd.Series([17020.0] * 60)  # EMA below breakout
        mock_atr.return_value = pd.Series([50.0] * 60)  # ATR for filter

        strategy = NqOrb(data_handler, events)

        # Setup range
        strategy.orb_high = 17050
        strategy.orb_low = 16950
        strategy.current_date = datetime(2024, 1, 15).date()

        # Breakout bar (close > orb_high and close > ema)
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 10, 30),
                  open=17040, high=17080, low=17030, close=17070, volume=100)
        strategy.calculate_signals(bar)

        # Should have long signal
        assert not events.empty()
        signal = events.get()
        assert signal.signal_type == SignalType.LONG

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_short_signal_on_breakdown(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should generate short signal when price breaks below range."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar, SignalType
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        # Mock indicators (EMA above breakdown)
        mock_ema.return_value = pd.Series([17000.0] * 60)
        mock_atr.return_value = pd.Series([50.0] * 60)

        strategy = NqOrb(data_handler, events)

        # Setup range
        strategy.orb_high = 17050
        strategy.orb_low = 16950
        strategy.current_date = datetime(2024, 1, 15).date()

        # Breakdown bar (close < orb_low and close < ema)
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 10, 30),
                  open=16960, high=16970, low=16900, close=16920, volume=100)
        strategy.calculate_signals(bar)

        # Should have short signal
        assert not events.empty()
        signal = events.get()
        assert signal.signal_type == SignalType.SHORT


class TestNqOrbRiskManagement:
    """Tests for stop loss and take profit."""

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_stop_loss_long(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should exit long position on stop loss."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar, SignalType
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        mock_ema.return_value = pd.Series([17000.0] * 60)
        mock_atr.return_value = pd.Series([50.0] * 60)

        strategy = NqOrb(data_handler, events, stop_loss=50.0)

        # Simulate being in a long position
        strategy.orb_high = 17050
        strategy.orb_low = 16950
        strategy.current_date = datetime(2024, 1, 15).date()
        strategy.traded_today = True
        strategy.in_long = True
        strategy.entry_price = 17060

        # Bar that triggers stop loss (low < entry - stop_loss)
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 11, 0),
                  open=17020, high=17030, low=17000, close=17010, volume=100)
        strategy.calculate_signals(bar)

        # Should have exit signal
        assert not events.empty()
        signal = events.get()
        assert signal.signal_type == SignalType.EXIT
        assert strategy.in_long is False

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_take_profit_long(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should exit long position on take profit."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar, SignalType
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        mock_ema.return_value = pd.Series([17000.0] * 60)
        mock_atr.return_value = pd.Series([50.0] * 60)

        strategy = NqOrb(data_handler, events, take_profit=50.0)

        # Simulate being in a long position
        strategy.orb_high = 17050
        strategy.orb_low = 16950
        strategy.current_date = datetime(2024, 1, 15).date()
        strategy.traded_today = True
        strategy.in_long = True
        strategy.entry_price = 17060

        # Bar that triggers take profit (high > entry + take_profit)
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 11, 0),
                  open=17100, high=17120, low=17090, close=17110, volume=100)
        strategy.calculate_signals(bar)

        # Should have exit signal
        assert not events.empty()
        signal = events.get()
        assert signal.signal_type == SignalType.EXIT


class TestNqOrbSessionManagement:
    """Tests for session end liquidation."""

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_session_end_liquidation(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should liquidate position at session end."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar, SignalType
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        mock_ema.return_value = pd.Series([17000.0] * 60)
        mock_atr.return_value = pd.Series([50.0] * 60)

        strategy = NqOrb(data_handler, events, exit_time=time(15, 55))

        # Simulate having position
        strategy.current_date = datetime(2024, 1, 15).date()
        strategy.orb_high = 17050
        strategy.orb_low = 16950
        strategy.traded_today = True
        strategy.in_long = True
        strategy.entry_price = 17060

        # Bar at session end
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 15, 55),
                  open=17080, high=17090, low=17070, close=17085, volume=100)
        strategy.calculate_signals(bar)

        # Should have exit signal
        assert not events.empty()
        signal = events.get()
        assert signal.signal_type == SignalType.EXIT

    @patch('pandas_ta.ema')
    @patch('pandas_ta.atr')
    def test_one_trade_per_day(self, mock_atr, mock_ema, multi_day_ohlcv_data):
        """Should only allow one trade per day."""
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import Bar
        from nqorb import NqOrb

        events = Queue()
        data_handler = MemoryDataHandler(multi_day_ohlcv_data)

        for _ in range(100):
            data_handler.update_bars()

        mock_ema.return_value = pd.Series([17020.0] * 60)
        mock_atr.return_value = pd.Series([50.0] * 60)

        strategy = NqOrb(data_handler, events)

        # Setup range and mark as already traded
        strategy.orb_high = 17050
        strategy.orb_low = 16950
        strategy.current_date = datetime(2024, 1, 15).date()
        strategy.traded_today = True  # Already traded

        # Breakout bar that would trigger signal
        bar = Bar(symbol='NQ', timestamp=datetime(2024, 1, 15, 11, 0),
                  open=17060, high=17080, low=17055, close=17075, volume=100)
        strategy.calculate_signals(bar)

        # Should NOT have signal (already traded today)
        assert events.empty()
