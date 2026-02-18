
from datetime import datetime
import pandas as pd
import numpy as np
from .strategy import Strategy
from .schema import Bar
from .data import DataHandler
from queue import Queue

class NqOrbVwap(Strategy):
    """
    ORB Strategy with VWAP Trend Filter.
    - Long: Breakout > ORB High AND Price > VWAP.
    - Short: Breakout < ORB Low AND Price < VWAP.
    """
    def __init__(self, 
                 bars: DataHandler, 
                 events: Queue, 
                 orb_start="09:30", 
                 orb_end="09:45",
                 use_vwap=True):
        super().__init__(bars, events)
        self.orb_start_time = datetime.strptime(orb_start, "%H:%M").time()
        self.orb_end_time = datetime.strptime(orb_end, "%H:%M").time()
        self.exit_time = datetime.strptime("15:45", "%H:%M").time()
        
        # VWAP State
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.current_date = None
        
        # ORB State
        self.orb_high = -1.0
        self.orb_low = 1e9
        self.traded_today = False
        self.in_trade = False

    def calculate_signals(self, event: Bar):
        bar_date = event.timestamp.date()
        bar_time = event.timestamp.time()
        
        # 1. Reset Daily Logic
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.orb_high = -1.0
            self.orb_low = 1e9
            self.traded_today = False
            self.in_trade = False
            self.vwap_num = 0.0
            self.vwap_den = 0.0
            
        # 2. Update VWAP (Typical Price * Vol)
        typical_price = (event.high + event.low + event.close) / 3.0
        self.vwap_num += (typical_price * event.volume)
        self.vwap_den += event.volume
        current_vwap = self.vwap_num / self.vwap_den if self.vwap_den > 0 else event.close
        
        # 3. ORB Window
        if self.orb_start_time <= bar_time < self.orb_end_time:
            if self.orb_high == -1.0:
                self.orb_high = event.high
                self.orb_low = event.low
            else:
                self.orb_high = max(self.orb_high, event.high)
                self.orb_low = min(self.orb_low, event.low)
                
        # 4. Trading Window
        elif self.orb_end_time <= bar_time < self.exit_time:
            if not self.traded_today and not self.in_trade and self.orb_high != -1.0:
                
                # Logic
                if event.close > self.orb_high and event.close > current_vwap:
                    self.buy(event.symbol)
                    self.traded_today = True
                    self.in_trade = True
                    
                elif event.close < self.orb_low and event.close < current_vwap:
                    self.sell(event.symbol)
                    self.traded_today = True
                    self.in_trade = True
                    
        # 5. Exit
        elif bar_time >= self.exit_time:
            if self.in_trade:
                self.exit(event.symbol)
                self.in_trade = False

class NqOrbMomentum(Strategy):
    """
    ORB Strategy with RSI Momentum Filter.
    - Long: Breakout > ORB High AND RSI > 50 (Momentum builds).
    - Short: Breakout < ORB Low AND RSI < 50.
    """
    def __init__(self, 
                 bars: DataHandler, 
                 events: Queue, 
                 orb_start="09:30", 
                 orb_end="09:45",
                 rsi_period=14):
        super().__init__(bars, events)
        self.orb_start_time = datetime.strptime(orb_start, "%H:%M").time()
        self.orb_end_time = datetime.strptime(orb_end, "%H:%M").time()
        self.exit_time = datetime.strptime("15:45", "%H:%M").time()
        self.rsi_period = int(rsi_period)
        
        self.orb_high = -1.0
        self.orb_low = 1e9
        self.traded_today = False
        self.in_trade = False
        self.current_date = None

    def calculate_signals(self, event: Bar):
        bar_date = event.timestamp.date()
        bar_time = event.timestamp.time()
        
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.orb_high = -1.0
            self.orb_low = 1e9
            self.traded_today = False
            self.in_trade = False
            
        history = self.bars.get_latest_bars(event.symbol, N=self.rsi_period + 5)
        if len(history) < self.rsi_period + 1:
            return

        if self.orb_start_time <= bar_time < self.orb_end_time:
            if self.orb_high == -1.0:
                self.orb_high = event.high
                self.orb_low = event.low
            else:
                self.orb_high = max(self.orb_high, event.high)
                self.orb_low = min(self.orb_low, event.low)
                
        elif self.orb_end_time <= bar_time < self.exit_time:
            if not self.traded_today and not self.in_trade and self.orb_high != -1.0:
                
                # Calc RSI
                closes = pd.Series([b.close for b in history])
                delta = closes.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                if event.close > self.orb_high and rsi > 50:
                    self.buy(event.symbol)
                    self.traded_today = True
                    self.in_trade = True
                
                elif event.close < self.orb_low and rsi < 50:
                    self.sell(event.symbol)
                    self.traded_today = True
                    self.in_trade = True
                    
        elif bar_time >= self.exit_time:
            if self.in_trade:
                self.exit(event.symbol)
                self.in_trade = False
