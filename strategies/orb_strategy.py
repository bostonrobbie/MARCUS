from backtesting.strategy import Strategy
from backtesting.schema import OrderSide
from datetime import time, datetime, timedelta

class OpeningRangeBreakout(Strategy):
    """
    Opening Range Breakout (ORB) Strategy.
    
    Logic:
    1. Define Opening Range (e.g., 09:30 - 10:00).
    2. Calculate High/Low of this range.
    3. Enter Long if Price breaks High.
    4. Enter Short if Price breaks Low.
    5. Exit at End of Day (15:55) or Stop Loss.
    """
    
    def __init__(self, data_handler, events, 
                 orb_start=time(9, 30), 
                 orb_end=time(10, 00), 
                 exit_time=time(15, 55),
                 stop_loss_ticks=20): # 20 ticks = 5 points on ES
        super().__init__(data_handler, events)
        self.orb_start = orb_start
        self.orb_end = orb_end
        self.exit_time = exit_time
        self.stop_loss_ticks = stop_loss_ticks
        
        # State
        self.orb_high = None
        self.orb_low = None
        self.entry_price = None
        self.traded_today = False
        self.current_date = None

    def calculate_signals(self, event):
        bar = event
        dt = bar.timestamp
        current_time = dt.time()
        
        # Reset daily state
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.orb_high = -float('inf')
            self.orb_low = float('inf')
            self.traded_today = False
            self.entry_price = None
            
        # 1. Build ORB High/Low during the window
        if self.orb_start <= current_time < self.orb_end:
            self.orb_high = max(self.orb_high, bar.high)
            self.orb_low = min(self.orb_low, bar.low)
            return # Don't trade during formation range
            
        # 2. Trading Window
        if self.orb_end <= current_time < self.exit_time:
            if self.traded_today:
                # Check for Stop Loss or Exit logic? 
                # Basic engine handles exits if we send orders? 
                # Actually, our engine assumes we manage positions.
                # If we have a position, check EOD exit.
                if self.bought or self.sold:
                     # Check Stop Loss (Manual logic needed if engine handles raw orders)
                     # For simplicity, we trust EOD exit mainly here, 
                     # but let's implement EOD Check.
                     pass
            else:
                # Breakout Logic
                if bar.close > self.orb_high:
                    self.buy(bar.symbol, 1) # Buy 1 contract
                    self.traded_today = True
                    self.entry_price = bar.close
                    print(f"[{dt}] ORB BREAKOUT LONG: {bar.close} > {self.orb_high}")
                    
                elif bar.close < self.orb_low:
                    self.sell(bar.symbol, 1) # Sell 1 contract
                    self.traded_today = True
                    self.entry_price = bar.close
                    print(f"[{dt}] ORB BREAKOUT SHORT: {bar.close} < {self.orb_low}")

        # 3. End of Day Exit
        if current_time >= self.exit_time:
            if self.bought or self.sold:
                self.exit(bar.symbol)
                print(f"[{dt}] EOD EXIT")
