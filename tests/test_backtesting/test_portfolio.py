"""
Tests for Portfolio and Position classes.

These tests verify position tracking, PnL calculation, and order generation.
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


class TestPosition:
    """Tests for Position class."""

    def test_initialization(self):
        """Should initialize with default values."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ')

        assert pos.symbol == 'NQ'
        assert pos.multiplier == 1.0
        assert pos.quantity == 0
        assert pos.avg_price == 0.0
        assert pos.market_value == 0.0
        assert pos.unrealized_pnl == 0.0
        assert pos.realized_pnl == 0.0

    def test_initialization_with_values(self):
        """Should initialize with provided values."""
        from backtesting.portfolio import Position

        pos = Position(symbol='ES', multiplier=50.0, quantity=2, avg_price=4500.0)

        assert pos.symbol == 'ES'
        assert pos.multiplier == 50.0
        assert pos.quantity == 2
        assert pos.avg_price == 4500.0

    def test_update_market_value_long(self):
        """Should calculate market value for long position."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ', multiplier=20.0, quantity=2, avg_price=17000.0)
        pos.update_market_value(17100.0)

        # MV = 2 * 17100 * 20 = 684000
        assert pos.market_value == 684000.0

        # Unrealized PnL = MV - Cost = 684000 - (2 * 17000 * 20) = 684000 - 680000 = 4000
        assert pos.unrealized_pnl == 4000.0

    def test_update_market_value_short(self):
        """Should calculate market value for short position."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ', multiplier=20.0, quantity=-2, avg_price=17000.0)
        pos.update_market_value(16900.0)

        # MV = -2 * 16900 * 20 = -676000
        assert pos.market_value == -676000.0

        # Unrealized PnL = MV - Cost = -676000 - (-2 * 17000 * 20) = -676000 + 680000 = 4000
        assert pos.unrealized_pnl == 4000.0

    def test_update_fill_open_long(self):
        """Should open long position correctly."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ')
        pos.update_fill(quantity=2, price=17000.0, commission=4.10)

        assert pos.quantity == 2
        assert pos.avg_price == 17000.0

    def test_update_fill_open_short(self):
        """Should open short position correctly."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ')
        pos.update_fill(quantity=-2, price=17000.0, commission=4.10)

        assert pos.quantity == -2
        assert pos.avg_price == 17000.0

    def test_update_fill_add_to_long(self):
        """Should add to existing long position with weighted average."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ', quantity=2, avg_price=17000.0)
        pos.update_fill(quantity=2, price=17100.0, commission=4.10)

        assert pos.quantity == 4
        # Weighted avg: (2*17000 + 2*17100) / 4 = 68200 / 4 = 17050
        assert pos.avg_price == 17050.0

    def test_update_fill_reduce_long(self):
        """Should reduce long position and realize PnL."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ', multiplier=20.0, quantity=2, avg_price=17000.0)
        pos.update_fill(quantity=-1, price=17100.0, commission=2.05)

        assert pos.quantity == 1
        # Realized PnL = (17100 - 17000) * 1 * 20 = 2000
        assert pos.realized_pnl == 2000.0

    def test_update_fill_close_long(self):
        """Should close position completely and reset avg price."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ', multiplier=20.0, quantity=2, avg_price=17000.0)
        pos.update_fill(quantity=-2, price=17200.0, commission=4.10)

        assert pos.quantity == 0
        assert pos.avg_price == 0.0
        # Realized PnL = (17200 - 17000) * 2 * 20 = 8000
        assert pos.realized_pnl == 8000.0

    def test_update_fill_reduce_short(self):
        """Should reduce short position and realize PnL."""
        from backtesting.portfolio import Position

        pos = Position(symbol='NQ', multiplier=20.0, quantity=-2, avg_price=17000.0)
        pos.update_fill(quantity=1, price=16900.0, commission=2.05)

        assert pos.quantity == -1
        # Short PnL = (17000 - 16900) * 1 * 20 = 2000 (profit when price goes down)
        # But formula: (price - avg_price) * (-qty) * mult = (16900 - 17000) * (-(-1)) * 20 = -100 * 1 * 20 = -2000
        # Wait, let's trace through: trade_pnl = (price - avg) * (-qty) * mult
        # = (16900 - 17000) * (-(1)) * 20 = -100 * -1 * 20 = 2000
        assert pos.realized_pnl == 2000.0


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_initialization(self, sample_ohlcv_dict):
        """Should initialize with correct values."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        assert portfolio.initial_capital == 100000.0
        assert portfolio.current_cash == 100000.0
        assert len(portfolio.current_positions) == 0
        assert len(portfolio.equity_curve) == 0
        assert len(portfolio.trade_log) == 0

    def test_update_signal_long(self, sample_ohlcv_dict):
        """Should create buy order from long signal."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType, OrderEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        signal = SignalEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            signal_type=SignalType.LONG,
            target_qty=10
        )

        portfolio.update_signal(signal)

        # Should have order in queue
        assert not events.empty()
        order = events.get()
        assert isinstance(order, OrderEvent)
        assert order.symbol == 'NQ'
        assert order.side == OrderSide.BUY
        assert order.quantity == 10

    def test_update_signal_short(self, sample_ohlcv_dict):
        """Should create sell order from short signal."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType, OrderEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        signal = SignalEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            signal_type=SignalType.SHORT,
            target_qty=5
        )

        portfolio.update_signal(signal)

        order = events.get()
        assert order.side == OrderSide.SELL
        assert order.quantity == 5

    def test_update_signal_exit_with_position(self, sample_ohlcv_dict):
        """Should create liquidation order from exit signal."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType, OrderEvent, OrderSide, OrderType
        from backtesting.portfolio import Position

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        # Create existing position
        portfolio.current_positions['NQ'] = Position(symbol='NQ', quantity=5, avg_price=17000.0)

        signal = SignalEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            signal_type=SignalType.EXIT
        )

        portfolio.update_signal(signal)

        # Should have cancel order first, then liquidation order
        cancel_order = events.get()
        assert cancel_order.order_type == OrderType.CANCEL_ALL

        liquidate_order = events.get()
        assert liquidate_order.side == OrderSide.SELL
        assert liquidate_order.quantity == 5

    def test_update_signal_exit_no_position(self, sample_ohlcv_dict):
        """Should not create order if no position to exit."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import SignalEvent, SignalType

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        signal = SignalEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            signal_type=SignalType.EXIT
        )

        portfolio.update_signal(signal)

        # Cancel order for unknown symbol is still created, but no liquidation
        # Actually looking at code, it only puts cancel if there's a position
        # Let me check again...
        # The code first puts CANCEL_ALL, then checks position
        # So we should only have cancel order, then return
        cancel_order = events.get()
        assert events.empty()  # No liquidation order

    def test_update_fill_creates_position(self, sample_ohlcv_dict):
        """Should create position from fill event."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import FillEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        fill = FillEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=2,
            price=17000.0,
            commission=4.10,
            slippage=0.50,
            side=OrderSide.BUY
        )

        portfolio.update_fill(fill)

        assert 'NQ' in portfolio.current_positions
        assert portfolio.current_positions['NQ'].quantity == 2

    def test_update_fill_adjusts_cash(self, sample_ohlcv_dict):
        """Should deduct cash for buy fills."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import FillEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        fill = FillEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=2,
            price=100.0,
            commission=4.10,
            slippage=0.0,
            side=OrderSide.BUY
        )

        portfolio.update_fill(fill)

        # Cash = 100000 - (2 * 100 * 1.0) - 4.10 = 100000 - 200 - 4.10 = 99795.90
        assert portfolio.current_cash == pytest.approx(99795.90, abs=0.01)

    def test_update_fill_logs_trade(self, sample_ohlcv_dict):
        """Should add entry to trade log."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import FillEvent, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        fill = FillEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=2,
            price=17000.0,
            commission=4.10,
            slippage=0.50,
            side=OrderSide.BUY
        )

        portfolio.update_fill(fill)

        assert len(portfolio.trade_log) == 1
        trade = portfolio.trade_log[0]
        assert trade['symbol'] == 'NQ'
        assert trade['side'] == 'BUY'
        assert trade['quantity'] == 2
        assert trade['price'] == 17000.0

    def test_update_timeindex(self, sample_ohlcv_dict):
        """Should append equity curve entry."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()  # Load first bar

        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
        portfolio.update_timeindex()

        assert len(portfolio.equity_curve) == 1
        entry = portfolio.equity_curve[0]
        assert 'datetime' in entry
        assert 'cash' in entry
        assert 'equity' in entry
        assert entry['equity'] == 100000.0  # No positions

    def test_update_timeindex_with_position(self, sample_ohlcv_dict):
        """Should include position market value in equity."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler
        from backtesting.portfolio import Position

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        # Create position
        portfolio.current_positions['NQ'] = Position(symbol='NQ', quantity=10, avg_price=100.0)
        portfolio.current_cash = 99000.0  # After buying 10 @ 100

        portfolio.update_timeindex()

        entry = portfolio.equity_curve[0]
        # Equity = cash + market_value
        # Market value depends on latest bar's close
        assert entry['equity'] > 99000.0  # Should have some market value

    def test_multiplier_from_instrument_config(self, sample_ohlcv_dict):
        """Should use multiplier from instrument config."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(
            data_handler, events,
            initial_capital=100000.0,
            instruments={'NQ': {'multiplier': 20.0}}
        )

        mult = portfolio._get_multiplier('NQ')
        assert mult == 20.0

    def test_default_multiplier(self, sample_ohlcv_dict):
        """Should use default multiplier of 1.0."""
        from backtesting.portfolio import Portfolio
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        portfolio = Portfolio(data_handler, events, initial_capital=100000.0)

        mult = portfolio._get_multiplier('UNKNOWN')
        assert mult == 1.0
