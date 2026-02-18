from abc import ABC, abstractmethod
from queue import Queue
from datetime import datetime
from typing import Optional, List
from .schema import OrderEvent, FillEvent, OrderSide, OrderStatus, Bar, OrderType
from .data import DataHandler

class TransactionCostModel(ABC):
    @abstractmethod
    def calculate_commission(self, quantity: int, price: float) -> float:
        raise NotImplementedError

class FixedCommission(TransactionCostModel):
    def __init__(self, commission_per_trade: float = 0.0, commission_per_share: float = 0.0):
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share

    def calculate_commission(self, quantity: int, price: float) -> float:
        return self.commission_per_trade + (self.commission_per_share * abs(quantity))

class AssetAwareCommissionModel(TransactionCostModel):
    """
    Calculates commission based on Asset Type (Futures vs Equity).
    Reads specs from a provided config dict or defaults.
    """
    def __init__(self, instrument_specs: Optional[dict] = None):
        self.specs = instrument_specs if instrument_specs else {}
        self.default_future_comm = 2.05
        self.default_equity_comm = 0.005

    def calculate_commission(self, quantity: int, price: float, symbol: str = "") -> float:
        qty = abs(quantity)
        spec = self.specs.get(symbol, {})
        asset_type = spec.get('type', 'EQUITY')
        
        if asset_type == 'FUTURE':
            rate = spec.get('commission', self.default_future_comm)
            return rate * qty
        else:
            rate = spec.get('commission', self.default_equity_comm)
            return max(1.0, rate * qty)

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, event: OrderEvent):
        raise NotImplementedError

class SlippageModel(ABC):
    @abstractmethod
    def calculate_slippage(self, quantity: int, price: float) -> float:
        raise NotImplementedError

class FixedSlippage(SlippageModel):
    def __init__(self, slippage_per_share: float = 0.0):
        self.slippage_per_share = slippage_per_share

    def calculate_slippage(self, quantity: int, price: float) -> float:
        # Total Slippage Dollar Amount = qty * per_share
        return abs(quantity) * self.slippage_per_share

class VolatilitySlippageModel(SlippageModel):
    """
    Slippage moves dynamically with market volatility.
    Slippage = (High - Low) * factor
    """
    def __init__(self, data_handler: DataHandler, factor: float = 0.05, min_ticks: int = 1):
        self.data_handler = data_handler
        self.factor = factor
        self.min_ticks = min_ticks

    def calculate_slippage(self, quantity: int, price: float, symbol: str = "") -> float:
        bar = self.data_handler.get_latest_bar(symbol)
        if not bar:
            return 0.0
            
        bar_range = bar.high - bar.low
        estimated_slip = bar_range * self.factor
        min_slip = 0.01 
        return max(min_slip, estimated_slip) * abs(quantity)

class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulates order execution.
    Supports:
    - OPTIMISTIC: Fill at Next Open.
    - PESSIMISTIC: Fill at Next Bar's Worst Price (High for Buy, Low for Sell).
    
    Why Worst Price? To simulate slippage and hostile market conditions.
    """
    
    def __init__(self, events: Queue, bars: DataHandler, 
                 commission_model: Optional[TransactionCostModel] = None,
                 slippage_model: Optional[SlippageModel] = None,
                 fill_on_next_open: bool = True,
                 mode: str = 'OPTIMISTIC'): # 'OPTIMISTIC' or 'PESSIMISTIC'
        
        self.events = events
        self.bars = bars
        self.commission_model = commission_model if commission_model else FixedCommission()
        self.slippage_model = slippage_model if slippage_model else FixedSlippage(slippage_per_share=0.0)
        self.fill_on_next_open = True # Alway True for realism in this engine
        self.mode = mode
        
        self.pending_orders: List[OrderEvent] = []

    def on_bar(self, bar: Bar):
        """
        Called when a new Bar is received. 
        Checks strictly for pending orders that need to fill.
        """
        remaining_orders = []
        
        # Process pending orders
        for order in list(self.pending_orders):
            if order.symbol == bar.symbol:
                filled = False
                fill_price = 0.0
                
                # --- TRADINGVIEW EMULATOR LOGIC (INTRABAR) ---
                if self.mode == 'TV_BROKER_EMULATOR':
                     # Limit Orders: Check if Low <= Limit <= High
                     if order.order_type == OrderType.LIMIT:
                         if order.side == OrderSide.BUY:
                             if bar.low <= order.limit_price:
                                 # TV assumes fill at Limit, or Open if gap down
                                 fill_price = min(order.limit_price, bar.open) if bar.open < order.limit_price else order.limit_price
                                 filled = True
                         else: # SELL
                             if bar.high >= order.limit_price:
                                 fill_price = max(order.limit_price, bar.open) if bar.open > order.limit_price else order.limit_price
                                 filled = True
                                 
                     # Stop Orders: Check if Low <= Stop <= High
                     elif order.order_type == OrderType.STOP:
                         if order.side == OrderSide.BUY: # Stop Limit / Stop Market (Buy Stop)
                             if bar.high >= order.stop_price:
                                 fill_price = max(order.stop_price, bar.open)
                                 filled = True
                         else: # Sell Stop
                             if bar.low <= order.stop_price:
                                 fill_price = min(order.stop_price, bar.open)
                                 filled = True
                                 
                     # Market: Always fill
                     elif order.order_type == OrderType.MARKET:
                         fill_price = bar.open
                         filled = True
                
                # --- STANDARD (NEXT BAR) LOGIC ---
                elif True: # Fallback or explicit check
                    if order.order_type == OrderType.MARKET:
                        # Market Order Logic
                        fill_price = bar.open # Default Optimistic
                        if self.mode == 'PESSIMISTIC':
                            if order.side == OrderSide.BUY:
                                fill_price = bar.high
                            else:
                                fill_price = bar.low
                        filled = True

                    elif order.order_type == OrderType.LIMIT:
                        # Limit Order Logic
                        limit = order.limit_price
                        if order.side == OrderSide.BUY:
                            # Buy Limit: Low must be <= Limit
                            if bar.low <= limit:
                                fill_price = min(limit, bar.open)
                                filled = True
                        else:
                            # Sell Limit: High must be >= Limit
                            if bar.high >= limit:
                                fill_price = max(limit, bar.open)
                                filled = True

                    elif order.order_type == OrderType.STOP:
                        # Stop Order Logic (was missing - STOP orders never filled)
                        stop = order.stop_price
                        if order.side == OrderSide.BUY:
                            # Buy Stop: triggers when price rises to stop level
                            if bar.high >= stop:
                                fill_price = max(stop, bar.open)
                                filled = True
                        else:
                            # Sell Stop: triggers when price falls to stop level
                            if bar.low <= stop:
                                fill_price = min(stop, bar.open)
                                filled = True
                            
                if filled:
                    self._fill_order(order, price=fill_price, timestamp=bar.timestamp)
                else:
                    remaining_orders.append(order)
            else:
                remaining_orders.append(order)
        
        self.pending_orders = remaining_orders

    def execute_order(self, event: OrderEvent):
        if event.status == OrderStatus.CANCELED:
            return
        
        if event.order_type == OrderType.CANCEL_ALL:
            # Remove all pending pending orders for this symbol
            self.pending_orders = [o for o in self.pending_orders if o.symbol != event.symbol]
            return

        # Always queue for next bar to avoid look-ahead bias
        self.pending_orders.append(event)

    def _fill_order(self, event: OrderEvent, price: float, timestamp: datetime):
        # We need to support old models (calculate_commission(qty, price)) AND new ones (..., symbol=...)
        # Inspect signatures or just try/except? 
        # Actually, python ignores extra kwargs if we aren't careful, so let's check or assume we updated the base interface?
        # The base interface in my previous edit didn't change the abstract method signature (oops).
        # But python is dynamic. Let's just update the abstract signature too or allow kwargs.
        # For now, let's assume I'll update the abstract methods in a second.
        
        # ACTUALLY, I should update the abstract methods first to be safe.
        # But wait, I can just try call with symbol, fallback? No, simpler to just update the calls and rely on updated models.
        # For legacy models (FixedCommission), they might not accept 'symbol'.
        # Solution: Inspect method.
        
        import inspect
        
        # Commission
        comm_sig = inspect.signature(self.commission_model.calculate_commission)
        if 'symbol' in comm_sig.parameters:
             commission = self.commission_model.calculate_commission(event.quantity, price, symbol=event.symbol)
        else:
             commission = self.commission_model.calculate_commission(event.quantity, price)

        # Slippage
        slip_sig = inspect.signature(self.slippage_model.calculate_slippage)
        if 'symbol' in slip_sig.parameters:
             slippage = self.slippage_model.calculate_slippage(event.quantity, price, symbol=event.symbol)
        else:
             slippage = self.slippage_model.calculate_slippage(event.quantity, price)
        
        # Adjust fill price for slippage
        # Buy: Price goes UP (worse)
        # Sell: Price goes DOWN (worse)
        final_price = price
        price_impact = slippage / abs(event.quantity) if event.quantity != 0 else 0
        
        if event.side == OrderSide.BUY:
            final_price = price + price_impact
        else:
            final_price = price - price_impact

        fill = FillEvent(
            symbol=event.symbol,
            timestamp=timestamp,
            quantity=event.quantity,
            price=final_price,
            commission=commission,
            slippage=slippage,
            side=event.side
        )
        self.events.put(fill)
