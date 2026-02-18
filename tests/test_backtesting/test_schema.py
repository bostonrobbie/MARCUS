"""
Tests for schema classes (Bar, SignalEvent, OrderEvent, FillEvent, enums).

These tests verify the data classes and enums are correctly defined
and function as expected.
"""
import pytest
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestEnums:
    """Tests for schema enums."""

    def test_signal_type_values(self):
        """Should have LONG, SHORT, EXIT signal types."""
        from backtesting.schema import SignalType

        assert hasattr(SignalType, 'LONG')
        assert hasattr(SignalType, 'SHORT')
        assert hasattr(SignalType, 'EXIT')

    def test_order_type_values(self):
        """Should have MARKET, LIMIT, STOP, CANCEL_ALL order types."""
        from backtesting.schema import OrderType

        assert hasattr(OrderType, 'MARKET')
        assert hasattr(OrderType, 'LIMIT')
        assert hasattr(OrderType, 'STOP')
        assert hasattr(OrderType, 'CANCEL_ALL')

    def test_order_side_values(self):
        """Should have BUY and SELL sides."""
        from backtesting.schema import OrderSide

        assert hasattr(OrderSide, 'BUY')
        assert hasattr(OrderSide, 'SELL')

    def test_order_status_values(self):
        """Should have all order status values."""
        from backtesting.schema import OrderStatus

        assert hasattr(OrderStatus, 'CREATED')
        assert hasattr(OrderStatus, 'SUBMITTED')
        assert hasattr(OrderStatus, 'PARTIAL')
        assert hasattr(OrderStatus, 'FILLED')
        assert hasattr(OrderStatus, 'CANCELED')
        assert hasattr(OrderStatus, 'REJECTED')


class TestInstrument:
    """Tests for Instrument dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        from backtesting.schema import Instrument

        inst = Instrument(symbol='NQ')

        assert inst.symbol == 'NQ'
        assert inst.multiplier == 1.0
        assert inst.tick_size == 0.01
        assert inst.margin_req == 0.0

    def test_custom_values(self):
        """Should accept custom values."""
        from backtesting.schema import Instrument

        inst = Instrument(
            symbol='ES',
            multiplier=50.0,
            tick_size=0.25,
            margin_req=12000.0
        )

        assert inst.symbol == 'ES'
        assert inst.multiplier == 50.0
        assert inst.tick_size == 0.25
        assert inst.margin_req == 12000.0

    def test_immutable(self):
        """Should be frozen (immutable)."""
        from backtesting.schema import Instrument

        inst = Instrument(symbol='NQ')

        with pytest.raises(Exception):  # FrozenInstanceError
            inst.symbol = 'ES'


class TestBar:
    """Tests for Bar dataclass."""

    def test_creation(self):
        """Should create Bar with all fields."""
        from backtesting.schema import Bar

        ts = datetime(2024, 1, 15, 9, 30, 0)
        bar = Bar(
            symbol='NQ',
            timestamp=ts,
            open=17000.0,
            high=17050.0,
            low=16950.0,
            close=17025.0,
            volume=1000
        )

        assert bar.symbol == 'NQ'
        assert bar.timestamp == ts
        assert bar.open == 17000.0
        assert bar.high == 17050.0
        assert bar.low == 16950.0
        assert bar.close == 17025.0
        assert bar.volume == 1000

    def test_immutable(self):
        """Should be frozen (immutable)."""
        from backtesting.schema import Bar

        bar = Bar(
            symbol='NQ',
            timestamp=datetime.now(),
            open=100, high=101, low=99, close=100.5, volume=100
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            bar.close = 200


class TestSignalEvent:
    """Tests for SignalEvent dataclass."""

    def test_creation_minimal(self):
        """Should create with minimal required fields."""
        from backtesting.schema import SignalEvent, SignalType

        ts = datetime.now()
        signal = SignalEvent(
            symbol='NQ',
            timestamp=ts,
            signal_type=SignalType.LONG
        )

        assert signal.symbol == 'NQ'
        assert signal.timestamp == ts
        assert signal.signal_type == SignalType.LONG
        assert signal.target_price is None
        assert signal.target_qty is None
        assert signal.strength == 1.0

    def test_creation_full(self):
        """Should create with all optional fields."""
        from backtesting.schema import SignalEvent, SignalType

        signal = SignalEvent(
            symbol='ES',
            timestamp=datetime.now(),
            signal_type=SignalType.SHORT,
            target_price=4500.0,
            target_qty=2,
            strength=0.75
        )

        assert signal.target_price == 4500.0
        assert signal.target_qty == 2
        assert signal.strength == 0.75

    def test_signal_types(self):
        """Should work with all signal types."""
        from backtesting.schema import SignalEvent, SignalType

        ts = datetime.now()

        long_signal = SignalEvent(symbol='NQ', timestamp=ts, signal_type=SignalType.LONG)
        short_signal = SignalEvent(symbol='NQ', timestamp=ts, signal_type=SignalType.SHORT)
        exit_signal = SignalEvent(symbol='NQ', timestamp=ts, signal_type=SignalType.EXIT)

        assert long_signal.signal_type == SignalType.LONG
        assert short_signal.signal_type == SignalType.SHORT
        assert exit_signal.signal_type == SignalType.EXIT


class TestOrderEvent:
    """Tests for OrderEvent dataclass."""

    def test_market_order(self):
        """Should create market order correctly."""
        from backtesting.schema import OrderEvent, OrderType, OrderSide, OrderStatus

        order = OrderEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        assert order.quantity == 1
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.CREATED
        assert order.limit_price is None

    def test_limit_order(self):
        """Should create limit order with price."""
        from backtesting.schema import OrderEvent, OrderType, OrderSide

        order = OrderEvent(
            symbol='ES',
            timestamp=datetime.now(),
            quantity=2,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=4550.0
        )

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 4550.0

    def test_stop_order(self):
        """Should create stop order with stop price."""
        from backtesting.schema import OrderEvent, OrderType, OrderSide

        order = OrderEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=16900.0
        )

        assert order.order_type == OrderType.STOP
        assert order.stop_price == 16900.0

    def test_mutable_status(self):
        """OrderEvent should be mutable (not frozen)."""
        from backtesting.schema import OrderEvent, OrderType, OrderSide, OrderStatus

        order = OrderEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        # Should be able to update status
        order.status = OrderStatus.FILLED
        assert order.status == OrderStatus.FILLED


class TestFillEvent:
    """Tests for FillEvent dataclass."""

    def test_creation(self):
        """Should create fill with all required fields."""
        from backtesting.schema import FillEvent, OrderSide

        ts = datetime.now()
        fill = FillEvent(
            symbol='NQ',
            timestamp=ts,
            quantity=1,
            price=17000.0,
            commission=2.05,
            slippage=0.50,
            side=OrderSide.BUY
        )

        assert fill.symbol == 'NQ'
        assert fill.timestamp == ts
        assert fill.quantity == 1
        assert fill.price == 17000.0
        assert fill.commission == 2.05
        assert fill.slippage == 0.50
        assert fill.side == OrderSide.BUY
        assert fill.exchange_id is None

    def test_with_exchange_id(self):
        """Should accept optional exchange_id."""
        from backtesting.schema import FillEvent, OrderSide

        fill = FillEvent(
            symbol='ES',
            timestamp=datetime.now(),
            quantity=2,
            price=4500.0,
            commission=4.10,
            slippage=1.0,
            side=OrderSide.SELL,
            exchange_id='SIM-12345'
        )

        assert fill.exchange_id == 'SIM-12345'

    def test_immutable(self):
        """Should be frozen (immutable)."""
        from backtesting.schema import FillEvent, OrderSide

        fill = FillEvent(
            symbol='NQ',
            timestamp=datetime.now(),
            quantity=1,
            price=17000.0,
            commission=2.05,
            slippage=0.50,
            side=OrderSide.BUY
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            fill.price = 18000.0
