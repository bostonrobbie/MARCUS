"""
Tests for execution handlers and cost models.

These tests verify order execution, slippage, and commission calculations.
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


class TestFixedCommission:
    """Tests for FixedCommission model."""

    def test_per_trade_commission(self):
        """Should charge flat per-trade commission."""
        from backtesting.execution import FixedCommission

        model = FixedCommission(commission_per_trade=5.0)
        commission = model.calculate_commission(quantity=100, price=50.0)

        assert commission == 5.0

    def test_per_share_commission(self):
        """Should charge per-share commission."""
        from backtesting.execution import FixedCommission

        model = FixedCommission(commission_per_share=0.01)
        commission = model.calculate_commission(quantity=100, price=50.0)

        assert commission == 1.0  # 100 * 0.01

    def test_combined_commission(self):
        """Should combine per-trade and per-share."""
        from backtesting.execution import FixedCommission

        model = FixedCommission(commission_per_trade=5.0, commission_per_share=0.01)
        commission = model.calculate_commission(quantity=100, price=50.0)

        assert commission == 6.0  # 5.0 + 1.0

    def test_handles_negative_quantity(self):
        """Should use absolute quantity."""
        from backtesting.execution import FixedCommission

        model = FixedCommission(commission_per_share=0.01)
        commission = model.calculate_commission(quantity=-100, price=50.0)

        assert commission == 1.0


class TestAssetAwareCommissionModel:
    """Tests for AssetAwareCommissionModel."""

    def test_future_commission(self):
        """Should apply future commission rate."""
        from backtesting.execution import AssetAwareCommissionModel

        specs = {'NQ': {'type': 'FUTURE', 'commission': 2.05}}
        model = AssetAwareCommissionModel(instrument_specs=specs)

        commission = model.calculate_commission(quantity=2, price=17000.0, symbol='NQ')

        assert commission == 4.10  # 2 * 2.05

    def test_equity_commission(self):
        """Should apply equity commission rate."""
        from backtesting.execution import AssetAwareCommissionModel

        specs = {'AAPL': {'type': 'EQUITY', 'commission': 0.005}}
        model = AssetAwareCommissionModel(instrument_specs=specs)

        commission = model.calculate_commission(quantity=100, price=150.0, symbol='AAPL')

        assert commission == 1.0  # max(1.0, 100 * 0.005) = max(1.0, 0.5) = 1.0

    def test_default_future_commission(self):
        """Should use default future commission."""
        from backtesting.execution import AssetAwareCommissionModel

        specs = {'ES': {'type': 'FUTURE'}}
        model = AssetAwareCommissionModel(instrument_specs=specs)

        commission = model.calculate_commission(quantity=1, price=4500.0, symbol='ES')

        assert commission == 2.05  # default

    def test_unknown_symbol_defaults_to_equity(self):
        """Should default to equity for unknown symbols."""
        from backtesting.execution import AssetAwareCommissionModel

        model = AssetAwareCommissionModel()
        commission = model.calculate_commission(quantity=100, price=50.0, symbol='UNKNOWN')

        assert commission == 1.0  # min $1


class TestFixedSlippage:
    """Tests for FixedSlippage model."""

    def test_slippage_calculation(self):
        """Should calculate total slippage."""
        from backtesting.execution import FixedSlippage

        model = FixedSlippage(slippage_per_share=0.05)
        slippage = model.calculate_slippage(quantity=100, price=50.0)

        assert slippage == 5.0  # 100 * 0.05

    def test_handles_negative_quantity(self):
        """Should use absolute quantity."""
        from backtesting.execution import FixedSlippage

        model = FixedSlippage(slippage_per_share=0.05)
        slippage = model.calculate_slippage(quantity=-100, price=50.0)

        assert slippage == 5.0

    def test_zero_slippage(self):
        """Should return zero with zero rate."""
        from backtesting.execution import FixedSlippage

        model = FixedSlippage(slippage_per_share=0.0)
        slippage = model.calculate_slippage(quantity=100, price=50.0)

        assert slippage == 0.0


class TestVolatilitySlippageModel:
    """Tests for VolatilitySlippageModel."""

    def test_slippage_based_on_bar_range(self, sample_ohlcv_dict):
        """Should calculate slippage from bar high-low range."""
        from backtesting.execution import VolatilitySlippageModel
        from backtesting.data import MemoryDataHandler

        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        data_handler.update_bars()

        model = VolatilitySlippageModel(data_handler, factor=0.1)
        slippage = model.calculate_slippage(quantity=10, price=100.0, symbol='NQ')

        # Should be > 0 based on bar range
        assert slippage > 0

    def test_returns_zero_without_bar(self, sample_ohlcv_dict):
        """Should return 0 if no bar available."""
        from backtesting.execution import VolatilitySlippageModel
        from backtesting.data import MemoryDataHandler

        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        # Don't call update_bars

        model = VolatilitySlippageModel(data_handler, factor=0.1)
        slippage = model.calculate_slippage(quantity=10, price=100.0, symbol='NQ')

        assert slippage == 0.0


class TestSimulatedExecutionHandler:
    """Tests for SimulatedExecutionHandler."""

    def test_initialization(self, sample_ohlcv_dict):
        """Should initialize with default models."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)

        handler = SimulatedExecutionHandler(events, data_handler)

        assert handler.events is events
        assert handler.bars is data_handler
        assert len(handler.pending_orders) == 0

    def test_execute_order_queues_order(self, sample_ohlcv_dict):
        """Should queue order for next bar execution."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        handler = SimulatedExecutionHandler(events, data_handler)

        order = OrderEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        handler.execute_order(order)

        assert len(handler.pending_orders) == 1

    def test_execute_order_cancel_all(self, sample_ohlcv_dict):
        """Should cancel all pending orders for symbol."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        handler = SimulatedExecutionHandler(events, data_handler)

        # Add pending orders
        order1 = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        order2 = OrderEvent(
            symbol='ES', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        handler.pending_orders = [order1, order2]

        # Cancel NQ orders
        cancel = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=0, side=OrderSide.BUY, order_type=OrderType.CANCEL_ALL
        )
        handler.execute_order(cancel)

        assert len(handler.pending_orders) == 1
        assert handler.pending_orders[0].symbol == 'ES'

    def test_on_bar_fills_market_order(self, sample_ohlcv_dict):
        """Should fill market order on next bar."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide, FillEvent, Bar

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        handler = SimulatedExecutionHandler(events, data_handler)

        order = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        handler.pending_orders = [order]

        bar = Bar(
            symbol='NQ',
            timestamp=datetime.now(),
            open=17000.0, high=17050.0, low=16950.0, close=17025.0,
            volume=1000
        )

        handler.on_bar(bar)

        # Should have fill in queue
        fill = events.get()
        assert isinstance(fill, FillEvent)
        assert fill.symbol == 'NQ'
        assert fill.quantity == 1
        assert fill.price == 17000.0  # Fill at open (optimistic)

        # Pending order should be removed
        assert len(handler.pending_orders) == 0

    def test_on_bar_pessimistic_fill(self, sample_ohlcv_dict):
        """Should fill at worst price in pessimistic mode."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide, Bar

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        handler = SimulatedExecutionHandler(events, data_handler, mode='PESSIMISTIC')

        order = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        handler.pending_orders = [order]

        bar = Bar(
            symbol='NQ', timestamp=datetime.now(),
            open=17000.0, high=17050.0, low=16950.0, close=17025.0,
            volume=1000
        )

        handler.on_bar(bar)

        fill = events.get()
        assert fill.price == 17050.0  # Fill at high for buy (worst price)

    def test_on_bar_limit_order_filled(self, sample_ohlcv_dict):
        """Should fill limit order when price crosses limit."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide, Bar

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        handler = SimulatedExecutionHandler(events, data_handler)

        # Buy limit below current price
        order = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.LIMIT,
            limit_price=16975.0
        )
        handler.pending_orders = [order]

        bar = Bar(
            symbol='NQ', timestamp=datetime.now(),
            open=17000.0, high=17050.0, low=16950.0, close=17025.0,
            volume=1000
        )

        handler.on_bar(bar)

        fill = events.get()
        assert fill.price == 16975.0  # Fill at limit price

    def test_on_bar_limit_order_not_filled(self, sample_ohlcv_dict):
        """Should not fill limit order if price doesn't reach limit."""
        from backtesting.execution import SimulatedExecutionHandler
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide, Bar

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        handler = SimulatedExecutionHandler(events, data_handler)

        # Buy limit way below current price
        order = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.LIMIT,
            limit_price=16000.0
        )
        handler.pending_orders = [order]

        bar = Bar(
            symbol='NQ', timestamp=datetime.now(),
            open=17000.0, high=17050.0, low=16950.0, close=17025.0,
            volume=1000
        )

        handler.on_bar(bar)

        # No fill
        assert events.empty()
        # Order still pending
        assert len(handler.pending_orders) == 1

    def test_slippage_affects_fill_price(self, sample_ohlcv_dict):
        """Should add slippage to fill price for buys."""
        from backtesting.execution import SimulatedExecutionHandler, FixedSlippage
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide, Bar

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        slippage = FixedSlippage(slippage_per_share=1.0)
        handler = SimulatedExecutionHandler(events, data_handler, slippage_model=slippage)

        order = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        handler.pending_orders = [order]

        bar = Bar(
            symbol='NQ', timestamp=datetime.now(),
            open=17000.0, high=17050.0, low=16950.0, close=17025.0,
            volume=1000
        )

        handler.on_bar(bar)

        fill = events.get()
        # Fill price = open + slippage/qty = 17000 + 1.0/1 = 17001
        assert fill.price == 17001.0

    def test_commission_in_fill(self, sample_ohlcv_dict):
        """Should include commission in fill event."""
        from backtesting.execution import SimulatedExecutionHandler, FixedCommission
        from backtesting.data import MemoryDataHandler
        from backtesting.schema import OrderEvent, OrderType, OrderSide, Bar

        events = Queue()
        data_handler = MemoryDataHandler(sample_ohlcv_dict)
        commission = FixedCommission(commission_per_trade=2.05)
        handler = SimulatedExecutionHandler(events, data_handler, commission_model=commission)

        order = OrderEvent(
            symbol='NQ', timestamp=datetime.now(),
            quantity=1, side=OrderSide.BUY, order_type=OrderType.MARKET
        )
        handler.pending_orders = [order]

        bar = Bar(
            symbol='NQ', timestamp=datetime.now(),
            open=17000.0, high=17050.0, low=16950.0, close=17025.0,
            volume=1000
        )

        handler.on_bar(bar)

        fill = events.get()
        assert fill.commission == 2.05
