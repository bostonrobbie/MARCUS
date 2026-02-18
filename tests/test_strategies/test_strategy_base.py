"""
Tests for Strategy abstract base class.

These tests verify the strategy interface and helper methods.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from queue import Queue
from unittest.mock import Mock, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestStrategyInterface:
    """Tests for Strategy abstract interface."""

    def test_cannot_instantiate_directly(self, sample_ohlcv_dict):
        """Should not be able to instantiate abstract Strategy."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        with pytest.raises(TypeError):
            Strategy(data_handler, events)

    def test_concrete_implementation(self, sample_ohlcv_dict):
        """Should be able to create concrete strategy."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        class ConcreteStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = ConcreteStrategy(data_handler, events)

        assert strategy.bars is data_handler
        assert strategy.events is events

    def test_symbol_list_inherited(self, sample_ohlcv_dict):
        """Should inherit symbol list from data handler."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        class ConcreteStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = ConcreteStrategy(data_handler, events)

        assert 'NQ' in strategy.symbol_list
        assert 'ES' in strategy.symbol_list


class TestStrategyHelperMethods:
    """Tests for Strategy helper methods."""

    def test_buy_creates_long_signal(self, sample_ohlcv_dict):
        """Should create LONG signal."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)
        strategy.buy('NQ', quantity=10)

        signal = events.get()
        assert isinstance(signal, SignalEvent)
        assert signal.signal_type == SignalType.LONG
        assert signal.symbol == 'NQ'
        assert signal.target_qty == 10

    def test_sell_creates_short_signal(self, sample_ohlcv_dict):
        """Should create SHORT signal."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)
        strategy.sell('NQ', quantity=5)

        signal = events.get()
        assert signal.signal_type == SignalType.SHORT
        assert signal.target_qty == 5

    def test_exit_creates_exit_signal(self, sample_ohlcv_dict):
        """Should create EXIT signal."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)
        strategy.exit('NQ')

        signal = events.get()
        assert signal.signal_type == SignalType.EXIT
        assert signal.symbol == 'NQ'

    def test_buy_with_limit_price(self, sample_ohlcv_dict):
        """Should include limit price in signal."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)
        strategy.buy('NQ', quantity=1, limit_price=17000.0)

        signal = events.get()
        assert signal.target_price == 17000.0

    def test_signal_timestamp_from_bar(self, sample_ohlcv_dict):
        """Should use bar timestamp for signal."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        bar = data_handler.get_latest_bar('NQ')

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)
        strategy.buy('NQ')

        signal = events.get()
        assert signal.timestamp == bar.timestamp


class TestStrategyOnFill:
    """Tests for strategy on_fill callback."""

    def test_on_fill_default_does_nothing(self, sample_ohlcv_dict):
        """Default on_fill should do nothing."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import FillEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)

        fill = FillEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            price=17000.0,
            commission=2.05,
            slippage=0.0,
            side=OrderSide.BUY
        )

        # Should not raise
        strategy.on_fill(fill)

    def test_on_fill_can_be_overridden(self, sample_ohlcv_dict):
        """Strategy can override on_fill."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import FillEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        fills_received = []

        class TrackingStrategy(Strategy):
            def calculate_signals(self, event):
                pass

            def on_fill(self, event):
                fills_received.append(event)

        strategy = TrackingStrategy(data_handler, events)

        fill = FillEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            price=17000.0,
            commission=2.05,
            slippage=0.0,
            side=OrderSide.BUY
        )

        strategy.on_fill(fill)

        assert len(fills_received) == 1
        assert fills_received[0] is fill


class TestStrategyPineExport:
    """Tests for Pine Script export functionality."""

    def test_export_without_template(self, sample_ohlcv_dict):
        """Should return default message without template."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        class NoTemplateStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = NoTemplateStrategy(data_handler, events)
        result = strategy.export_to_pine({})

        assert "No Pine Script Template" in result

    def test_export_with_template(self, sample_ohlcv_dict):
        """Should fill template with parameters."""
        from backtesting.strategy import Strategy
        from backtesting.data import MemoryDataHandler
        from unittest.mock import patch

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        class TemplatedStrategy(Strategy):
            PINE_TEMPLATE = "length = {{length}}"

            def calculate_signals(self, event):
                pass

        strategy = TemplatedStrategy(data_handler, events)

        with patch('backtesting.pine_generator.PineScriptHelper.fill_template') as mock_fill:
            mock_fill.return_value = "length = 14"
            result = strategy.export_to_pine({'length': 14})

        mock_fill.assert_called_once()
