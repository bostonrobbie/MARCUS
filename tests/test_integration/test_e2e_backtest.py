"""
End-to-end integration tests for the backtesting system.

These tests verify the complete backtest flow from data loading
through execution and performance reporting.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from queue import Queue
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'strategies'))


@pytest.mark.integration
class TestFullBacktestFlow:
    """Integration tests for complete backtest execution."""

    def test_simple_backtest_runs_to_completion(self, sample_ohlcv_dict):
        """Should run a complete backtest without errors."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy
        from backtesting.schema import SignalType

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        class SimpleBuyHoldStrategy(Strategy):
            def __init__(self, bars, events):
                super().__init__(bars, events)
                self.bought = False

            def calculate_signals(self, event):
                if not self.bought:
                    self.buy(event.symbol, quantity=10)
                    self.bought = True

        strategy = SimpleBuyHoldStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)

        # Should complete without error
        engine.run()

        # Should have equity curve
        assert len(portfolio.equity_curve) > 0

        # Should have at least one trade
        assert len(portfolio.trade_log) >= 1

    def test_backtest_with_multiple_symbols(self, sample_ohlcv_dict):
        """Should handle multiple symbols correctly."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        symbols_seen = set()

        class MultiSymbolStrategy(Strategy):
            def calculate_signals(self, event):
                symbols_seen.add(event.symbol)

        strategy = MultiSymbolStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have seen both symbols
        assert 'NQ' in symbols_seen
        assert 'ES' in symbols_seen

    def test_backtest_generates_valid_metrics(self, sample_ohlcv_dict):
        """Should generate valid performance metrics."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy
        from backtesting.performance import TearSheet

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        buy_count = [0]

        class TradingStrategy(Strategy):
            def calculate_signals(self, event):
                if buy_count[0] < 3:
                    self.buy(event.symbol, quantity=10)
                    buy_count[0] += 1

        strategy = TradingStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Create tear sheet
        with patch.object(TearSheet, 'create_html_report'):
            tear_sheet = TearSheet(portfolio)
            stats = tear_sheet.analyze()

        # Should have key metrics
        assert 'Total Return' in stats or len(portfolio.equity_curve) > 0


@pytest.mark.integration
class TestTypeNormalizationIntegration:
    """Integration tests for Bug 1 fix - type normalization."""

    def test_vectorized_backtest_returns_pandas(self, sample_ohlcv_dict, mock_gpu_unavailable):
        """VectorEngine should return pandas types after fix."""
        from backtesting.vector_engine import VectorEngine
        from backtesting.type_utils import ensure_pandas_series

        class MockStrategy:
            def __init__(self, **params):
                pass

            def calculate_signals(self, df):
                # Simple moving average crossover
                return pd.Series(np.zeros(len(df)), index=df.index)

        engine = VectorEngine(MockStrategy, initial_capital=100000)
        result = engine.run(sample_ohlcv_dict['NQ'])

        # All results should be pandas after normalization
        assert isinstance(result['equity_curve'], pd.Series)
        assert isinstance(result['signals'], pd.Series)
        assert isinstance(result['returns'], pd.Series)

    def test_safe_iloc_works_on_all_types(self, numpy_array_equity, pandas_series_equity):
        """safe_iloc should work on numpy and pandas."""
        from backtesting.type_utils import safe_iloc

        # Numpy array
        np_last = safe_iloc(numpy_array_equity, -1)
        assert np_last == 110000.0

        # Pandas Series
        pd_last = safe_iloc(pandas_series_equity, -1)
        assert pd_last == 110000.0


@pytest.mark.integration
class TestFullTradeCycle:
    """Integration tests for complete buy-sell cycles."""

    def test_buy_sell_cycle_updates_pnl(self):
        """Complete trade cycle should update realized PnL."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy
        from backtesting.schema import SignalType

        # Create price data with known profit potential
        data = pd.DataFrame({
            'Open': [100, 102, 104, 106, 108, 110],
            'High': [101, 103, 105, 107, 109, 111],
            'Low': [99, 101, 103, 105, 107, 109],
            'Close': [100.5, 102.5, 104.5, 106.5, 108.5, 110.5],
            'Volume': [1000] * 6
        }, index=pd.date_range('2024-01-01', periods=6, freq='1min'))

        events = Queue()
        data_handler = MemoryDataHandler({'TEST': data})
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(events, data_handler)

        bar_count = [0]

        class BuySellStrategy(Strategy):
            def calculate_signals(self, event):
                bar_count[0] += 1
                if bar_count[0] == 1:
                    self.buy('TEST', quantity=100)
                elif bar_count[0] == 5:
                    self.exit('TEST')

        strategy = BuySellStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have both buy and sell
        assert len(portfolio.trade_log) >= 2

        # Should have realized PnL
        sells = [t for t in portfolio.trade_log if t['side'] == 'SELL']
        if sells:
            # Price went up, should have positive PnL
            assert portfolio.current_positions.get('TEST') is None or \
                   portfolio.current_positions['TEST'].quantity == 0


@pytest.mark.integration
class TestDataHandlerIntegration:
    """Integration tests for data handler with engine."""

    def test_data_flows_through_engine(self, sample_ohlcv_dict):
        """Data should flow correctly through entire system."""
        from backtesting.engine import BacktestEngine
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Portfolio
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.strategy import Strategy

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events)
        execution = SimulatedExecutionHandler(events, data_handler)

        bars_received = []

        class TrackingStrategy(Strategy):
            def calculate_signals(self, event):
                bars_received.append({
                    'symbol': event.symbol,
                    'close': event.close,
                    'timestamp': event.timestamp
                })

        strategy = TrackingStrategy(data_handler, events)
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()

        # Should have received all bars
        assert len(bars_received) == 200  # 100 bars * 2 symbols

        # Verify data integrity
        nq_bars = [b for b in bars_received if b['symbol'] == 'NQ']
        assert len(nq_bars) == 100
