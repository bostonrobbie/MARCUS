import pandas as pd
from queue import Queue
from typing import Dict, List, Optional
from datetime import datetime
from .schema import SignalEvent, OrderEvent, FillEvent, Bar, SignalType, OrderType, OrderSide, OrderStatus
from .data import DataHandler

class Position:
    def __init__(self, symbol: str, multiplier: float = 1.0, quantity: int = 0, avg_price: float = 0.0):
        self.symbol = symbol
        self.multiplier = multiplier
        self.quantity = quantity
        self.avg_price = avg_price
        self.market_value = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

    def update_market_value(self, price: float):
        # Market Value only makes sense for Equity? For Futures, it's about PnL and Margin.
        # But commonly: MV = Qty * Price * Multiplier
        self.market_value = self.quantity * price * self.multiplier
        cost_basis = self.quantity * self.avg_price * self.multiplier
        self.unrealized_pnl = self.market_value - cost_basis

    def update_fill(self, quantity: int, price: float, commission: float):
        if self.quantity == 0:
            # Opening new position
            self.quantity = quantity
            self.avg_price = price
        else:
            # Adjusting
            if (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
                # Adding to position
                # Weighted Average Price
                current_cost = self.quantity * self.avg_price
                new_cost = quantity * price
                total_qty = self.quantity + quantity
                self.avg_price = (current_cost + new_cost) / total_qty
                self.quantity = total_qty
            else:
                # Closing / Reducing
                # PnL = (ExitPrice - EntryPrice) * Qty * Multiplier
                trade_pnl = (price - self.avg_price) * (-quantity) * self.multiplier
                self.realized_pnl += trade_pnl
                
                self.quantity += quantity
                if self.quantity == 0:
                    self.avg_price = 0.0

class Portfolio:
    def __init__(self, bars: DataHandler, events: Queue, initial_capital: float = 100000.0, 
                 instruments: Dict[str, dict] = None):
        """
        instruments: dict of symbol -> {'multiplier': float, 'margin': float}
        """
        self.bars = bars
        self.events = events
        self.initial_capital = initial_capital
        
        # Default config if generic
        self.instrument_config = instruments if instruments else {} 
        
        self.current_cash = initial_capital
        self.current_positions: Dict[str, Position] = {} 
        
        # History
        self.equity_curve: List[Dict] = []
        self.trade_log: List[Dict] = [] # New: Track every fill/trade

    def _get_multiplier(self, symbol):
        return self.instrument_config.get(symbol, {}).get('multiplier', 1.0)
        
    def update_signal(self, event: SignalEvent):
        """
        Acts on a SignalEvent to generate an OrderEvent.
        """
        # Simple sizing: 100 shares if not specified
        # FIX: Use 'is not None' to correctly handle target_qty=0 (explicit zero)
        qty = event.target_qty if event.target_qty is not None else 100
        side = OrderSide.BUY if event.signal_type == SignalType.LONG else OrderSide.SELL
        
        if event.signal_type == SignalType.EXIT:
            # First, cancel all pending orders for this symbol
            self.events.put(OrderEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                quantity=0,
                side=OrderSide.BUY, # Dummy side
                order_type=OrderType.CANCEL_ALL
            ))

            # Liquidate logic
            if event.symbol in self.current_positions:
                pos = self.current_positions[event.symbol]
                if pos.quantity > 0:
                    side = OrderSide.SELL
                    qty = pos.quantity
                elif pos.quantity < 0:
                    side = OrderSide.BUY
                    qty = abs(pos.quantity)
                else:
                    return # No position
            else:
                return # No position

        # Create Order
        # Convention: Order Qty is always Positive? Or Signed?
        # Schema says 'quantity: int'. Usually Order events specify Side and positive quantity.
        # But Position.update_fill usually easier with signed.
        # Let's keep Order Quantity POSITIVE and use SIDE.
        
        if event.target_price:
            order_type = OrderType.LIMIT
            limit_price = event.target_price
        else:
            order_type = OrderType.MARKET
            limit_price = None

        order = OrderEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            quantity=abs(qty),
            side=side,
            order_type=order_type,
            limit_price=limit_price
        )
        self.events.put(order)

    def update_fill(self, event: FillEvent):
        """
        Updates portfolio current positions from a FillEvent.
        """
        # Determine signed quantity for position update
        fill_qty = event.quantity if event.side == OrderSide.BUY else -event.quantity
        multiplier = self._get_multiplier(event.symbol)

        trade_pnl = 0.0
        if event.symbol in self.current_positions:
            pos = self.current_positions[event.symbol]
            if (pos.quantity > 0 and fill_qty < 0) or (pos.quantity < 0 and fill_qty > 0):
                # Reduction: Calculate PnL for this specific fill
                # Use min of absolute quantities to get the part of the position being closed
                qty_closed = min(abs(pos.quantity), abs(fill_qty))
                trade_pnl = (event.price - pos.avg_price) * (qty_closed if pos.quantity > 0 else -qty_closed) * multiplier
            
        if event.symbol not in self.current_positions:
            self.current_positions[event.symbol] = Position(event.symbol, multiplier=multiplier)
            
        self.current_positions[event.symbol].update_fill(fill_qty, event.price, event.commission)
        
        # Cash Flow Logic:
        # For futures (multiplier > 1), use margin-based accounting:
        # Only commission impacts cash (PnL flows through position mark-to-market)
        # For equities (multiplier == 1), use full notional cost
        is_futures = self.instrument_config.get(event.symbol, {}).get('type', 'EQUITY') == 'FUTURE'
        if is_futures:
            # Futures: only commission deducted from cash (margin not modeled as cost)
            self.current_cash -= event.commission
        else:
            cost = fill_qty * event.price * multiplier
            self.current_cash -= (cost + event.commission)
        
        # Log Trade
        self.trade_log.append({
            'datetime': event.timestamp,
            'symbol': event.symbol,
            'side': event.side.name,
            'quantity': event.quantity,
            'price': event.price,
            'commission': event.commission,
            'multiplier': multiplier,
            'realized_pnl': trade_pnl
        })

    def update_timeindex(self):
        """
        Calculates current equity and appends to history.
        Should be called at the end of every bar processing cycle.
        """
        total_market_value = 0.0
        
        # Update MTM for all positions
        for symbol, pos in self.current_positions.items():
            if pos.quantity != 0:
                bar = self.bars.get_latest_bar(symbol)
                if bar:
                    pos.update_market_value(bar.close)
                    total_market_value += pos.market_value
        
        total_equity = self.current_cash + total_market_value
        
        # Determine strict timestamp (from one of the bars?)
        # We'll validly assume the engine guarantees we are at 'current_time'
        # But here we just take the last update? 
        # Ideally passing 'timestamp' to this function is better.
        timestamp = datetime.now() # Placeholder, ideally pass current backtest time
        # Try to guess from data
        if self.bars.symbol_list:
             # Just use the first symbol's latest bar time
             bar = self.bars.get_latest_bar(self.bars.symbol_list[0])
             if bar:
                 timestamp = bar.timestamp
        
        self.equity_curve.append({
            'datetime': timestamp,
            'cash': self.current_cash,
            'equity': total_equity,
            'total_market_value': total_market_value
        })
