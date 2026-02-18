"""
Tests for BacktestEngine class.

These tests verify the event-driven backtesting engine correctly
processes bars, signals, orders, and fills.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from queue import Queue
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_initialization(self, sample_ohlcv_dict):
        """Should initialize with all components."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        class TestStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = TestStrategy(data_handler, events)

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=execution
        )

        assert engine.data_handler is data_handler
        assert engine.strategy is strategy
        assert engine.portfolio is portfolio
        assert engine.execution_handler is execution
        assert engine.events is portfolio.events

    def test_run_processes_all_bars(self, sample_ohlcv_dict, capsys):
        """Should process all available bars."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        signals_received = []

        class CountingStrategy(Strategy):
            def calculate_signals(self, event):
                signals_received.append(event)

        strategy = CountingStrategy(data_handler, events)

        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have processed all bars (100 bars * 2 symbols = 200)
        assert len(signals_received) == 200

        captured = capsys.readouterr()
        assert "Starting Backtest" in captured.out
        assert "Completed" in captured.out

    def test_run_updates_portfolio_timeindex(self, sample_ohlcv_dict):
        """Should update portfolio equity curve each bar."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        class NoOpStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = NoOpStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have equity curve entries
        assert len(portfolio.equity_curve) > 0

    def test_signal_to_order_flow(self, sample_ohlcv_dict):
        """Should convert signals to orders."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy
        from backtesting.schema import SignalEvent, SignalType

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        signal_count = [0]

        class SingleSignalStrategy(Strategy):
            def calculate_signals(self, event):
                # Only signal on first bar
                if signal_count[0] == 0:
                    signal = SignalEvent(
                        symbol=event.symbol,
                        timestamp=event.timestamp,
                        signal_type=SignalType.LONG,
                        target_qty=10
                    )
                    self.events.put(signal)
                    signal_count[0] += 1

        strategy = SingleSignalStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have at least one trade logged
        assert len(portfolio.trade_log) >= 1

    def test_execution_handler_on_bar_called(self, sample_ohlcv_dict):
        """Should call execution handler's on_bar method."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        # Track on_bar calls
        on_bar_calls = []
        original_on_bar = execution.on_bar
        def tracking_on_bar(event):
            on_bar_calls.append(event)
            return original_on_bar(event)
        execution.on_bar = tracking_on_bar

        class NoOpStrategy(Strategy):
            def calculate_signals(self, event):
                pass

        strategy = NoOpStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have called on_bar for each bar
        assert len(on_bar_calls) == 200  # 100 bars * 2 symbols


class TestBacktestEngineIntegration:
    """Integration tests for the full backtest flow."""

    def test_full_trade_cycle(self):
        """Should execute complete buy-sell cycle."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy
        from backtesting.schema import SignalEvent, SignalType

        # Create price data with known movement
        data = pd.DataFrame({
            'Open': [100, 100, 102, 104, 106],
            'High': [101, 102, 104, 106, 108],
            'Low': [99, 99, 101, 103, 105],
            'Close': [100, 102, 104, 106, 108],
            'Volume': [1000] * 5
        }, index=pd.date_range('2024-01-01', periods=5, freq='1min'))

        events = Queue()
        data_handler = MemoryDataHandler({'TEST': data})
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        bar_count = [0]

        class BuySellStrategy(Strategy):
            def calculate_signals(self, event):
                bar_count[0] += 1
                if bar_count[0] == 1:
                    # Buy on first bar
                    self.events.put(SignalEvent(
                        symbol='TEST',
                        timestamp=event.timestamp,
                        signal_type=SignalType.LONG,
                        target_qty=100
                    ))
                elif bar_count[0] == 4:
                    # Exit on fourth bar
                    self.events.put(SignalEvent(
                        symbol='TEST',
                        timestamp=event.timestamp,
                        signal_type=SignalType.EXIT
                    ))

        strategy = BuySellStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have 2 trades (buy and sell)
        assert len(portfolio.trade_log) == 2

        # First trade is buy
        assert portfolio.trade_log[0]['side'] == 'BUY'

        # Second trade is sell
        assert portfolio.trade_log[1]['side'] == 'SELL'

    def test_no_look_ahead_bias(self):
        """Orders should fill on next bar, not current bar."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy
        from backtesting.schema import SignalEvent, SignalType

        # Create specific price pattern
        data = pd.DataFrame({
            'Open': [100, 110, 120],  # Note the jumps
            'High': [105, 115, 125],
            'Low': [95, 105, 115],
            'Close': [102, 112, 122],
            'Volume': [1000] * 3
        }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))

        events = Queue()
        data_handler = MemoryDataHandler({'TEST': data})
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler, mode='OPTIMISTIC')

        bar_count = [0]

        class FirstBarBuyStrategy(Strategy):
            def calculate_signals(self, event):
                bar_count[0] += 1
                if bar_count[0] == 1:
                    self.events.put(SignalEvent(
                        symbol='TEST',
                        timestamp=event.timestamp,
                        signal_type=SignalType.LONG,
                        target_qty=10
                    ))

        strategy = FirstBarBuyStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should fill at second bar's open (110), not first bar's close (102)
        if portfolio.trade_log:
            fill_price = portfolio.trade_log[0]['price']
            assert fill_price == 110  # Next bar open
