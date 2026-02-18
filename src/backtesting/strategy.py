from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from queue import Queue
from datetime import datetime
from .schema import SignalEvent, SignalType, Bar
from .data import DataHandler

try:
    import pandas_ta as ta
except ImportError:
    ta = None
    # Provide a warning but don't crash usually, unless it's used.
    # print("Warning: pandas_ta not found. Indicators will not be available.")

class Strategy(ABC):
    """
    Abstract Base Class for Strategies.
    Strategies accept updates (Bars) and generate SignalEvents.
    """
    
    def __init__(self, bars: DataHandler, events: Queue):
        self.bars = bars       # Access to historical data
        self.events = events   # Queue to push signals to behavior
        # FIX: Safely get symbol_list with fallback for any DataHandler that lacks it
        self.symbol_list = getattr(bars, 'symbol_list', list(getattr(bars, 'symbol_data', {}).keys()))

    @abstractmethod
    def calculate_signals(self, event: Bar):
        """
        Calculates signals based on the received Bar event.
        Must be implemented by the user.
        """
        raise NotImplementedError

    def export_to_pine(self, params: Dict[str, Any]) -> str:
        """
        Exports the strategy logic back to Pine Script with optimized parameters.
        Must be implemented by subclasses or use a PINE_TEMPLATE class attribute.
        """
        if hasattr(self, 'PINE_TEMPLATE'):
            from .pine_generator import PineScriptHelper
            return PineScriptHelper.fill_template(self.PINE_TEMPLATE, params)
        return "// No Pine Script Template defined for this strategy."

    def on_fill(self, event):
        """
        Updates strategy state upon order execution.
        Optional override.
        """
        pass

    def _create_signal(self, symbol: str, signal_type: SignalType, strength: float = 1.0, 
                      target_qty: Optional[int] = None, target_price: Optional[float] = None):
        """
        Internal helper to create and push a signal event.
        """
        # Get latest timestamp from data handler to ensure alignment
        latest_bar = self.bars.get_latest_bar(symbol)
        if latest_bar:
            timestamp = latest_bar.timestamp
        else:
            timestamp = datetime.now() # Fallback, should rarely happen in backtest
            
        signal = SignalEvent(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            target_price=target_price,
            target_qty=target_qty,
            strength=strength
        )
        self.events.put(signal)

    def buy(self, symbol: str, quantity: Optional[int] = None, limit_price: Optional[float] = None):
        """
        Generates a LONG signal.
        """
        self._create_signal(symbol, SignalType.LONG, target_qty=quantity, target_price=limit_price)

    def sell(self, symbol: str, quantity: Optional[int] = None, limit_price: Optional[float] = None):
        """
        Generates a SHORT signal.
        """
        self._create_signal(symbol, SignalType.SHORT, target_qty=quantity, target_price=limit_price)

    def exit(self, symbol: str):
        """
        Generates an EXIT signal (close all positions for symbol).
        """
        self._create_signal(symbol, SignalType.EXIT)
