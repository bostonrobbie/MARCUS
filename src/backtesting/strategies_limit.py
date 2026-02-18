
from datetime import datetime
import pandas as pd
from .strategy import Strategy
from .schema import Bar, OrderType
from .data import DataHandler
from queue import Queue

class NqPullbackLimit(Strategy):
    """
    Long-Term Robust Strategy Concept: "Pullback Liquidity"
    Instead of buying the ORB breakout at market, we place a Limit Order 
    at the ORB High (Retest) after the breakout is confirmed.
    
    Logic:
    1. Identify Breakout (Price > ORB High).
    2. Place LIMIT Buy at ORB High.
    3. Target: Volatility adjusted (2R).
    4. Stop: Below ORB Low.
    """
    def __init__(self, 
                 bars: DataHandler, 
                 events: Queue, 
                 orb_start="09:30", 
                 orb_end="09:45",
                 ema_filter=50):
        super().__init__(bars, events)
        self.orb_start_time = datetime.strptime(orb_start, "%H:%M").time()
        self.orb_end_time = datetime.strptime(orb_end, "%H:%M").time()
        self.exit_time = datetime.strptime("15:45", "%H:%M").time()
        self.ema_filter = int(ema_filter)
        
        # State
        self.orb_high = -1.0
        self.orb_low = 1e9
        self.breakout_confirmed = False
        self.order_placed = False
        self.current_date = None

    def calculate_signals(self, event: Bar):
        bar_date = event.timestamp.date()
        bar_time = event.timestamp.time()
        
        # Reset Day
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.orb_high = -1.0
            self.orb_low = 1e9
            self.breakout_confirmed = False
            self.order_placed = False
            
        # 1. Undefined Phase (Pre-Market)
        if bar_time < self.orb_start_time:
            return

        # 2. ORB Phase
        if self.orb_start_time <= bar_time < self.orb_end_time:
            if self.orb_high == -1.0:
                self.orb_high = event.high
                self.orb_low = event.low
            else:
                self.orb_high = max(self.orb_high, event.high)
                self.orb_low = min(self.orb_low, event.low)

        # 3. Trading Phase
        elif self.orb_end_time <= bar_time < self.exit_time:
            if not self.order_placed and self.orb_high != -1.0:
                # Check for Breakout Logic
                # Using Close > High for confirmation
                if event.close > self.orb_high:
                    # Breakout Confirmed!
                    # Do we buy now? NO. We wait for pullback.
                    # Place Limit @ ORB High
                    
                    limit_price = self.orb_high
                    
                    # Filter: Only if above EMA (Trend)
                    history = self.bars.get_latest_bars(event.symbol, N=self.ema_filter)
                    if len(history) >= self.ema_filter:
                        closes = pd.Series([b.close for b in history])
                        ema = closes.ewm(span=self.ema_filter, adjust=False).mean().iloc[-1]
                        
                        if event.close > ema:
                            # Execute Limit Buy
                            # Using 'limit_price=...' which our Strategy.buy must support
                            self.buy(event.symbol, quantity=1, limit_price=limit_price)
                            self.order_placed = True
                            print(f"[{event.timestamp}] LIMIT BUY placed at {limit_price}")

            # Exit Logic (Simple EOD for now, SL/TP should be bracket)
             
        # 4. EOD Exit
        elif bar_time >= self.exit_time:
            if self.order_placed:
                self.exit(event.symbol)
                self.order_placed = False
