
from datetime import datetime
import pandas as pd
from .strategy import Strategy
from .schema import Bar
from .data import DataHandler
from queue import Queue

class NqOrb15m(Strategy):
    """
    Event-driven implementation of the NQ 15m Opening Range Breakout strategy.
    Designed to match VectorizedNQORB for parity validation.
    """
    def __init__(self, 
                 bars: DataHandler, 
                 events: Queue, 
                 orb_start="09:30", 
                 orb_end="09:45", 
                 ema_filter=50, 
                 atr_filter=14, 
                 sl_atr_mult=2.0, 
                 tp_atr_mult=4.0, 
                 atr_max_mult=2.5,
                 # Phase 2 Args (accept them to be compatible with vector params, even if not fully implemented yet)
                 use_htf=False, htf_ma=200, 
                 use_rvol=False, rvol_thresh=1.5,
                 use_hurst=False, hurst_thresh=0.5,
                 use_adx=False, adx_thresh=20,
                 use_trailing_stop=False, ts_atr_mult=3.0):
        
        super().__init__(bars, events)
        
        self.orb_start_time = datetime.strptime(orb_start, "%H:%M").time()
        self.orb_end_time = datetime.strptime(orb_end, "%H:%M").time()
        self.exit_time = datetime.strptime("15:45", "%H:%M").time()
        
        self.ema_period = int(ema_filter)
        self.atr_period = int(atr_filter)
        self.sl_mult = float(sl_atr_mult)
        self.tp_mult = float(tp_atr_mult)
        self.atr_max_mult = float(atr_max_mult)
        
        # State
        self.orb_high = -1.0
        self.orb_low = 1e9
        self.traded_today = False
        self.current_date = None
        
        # Position State (implied, handled by Portfolio usually, but we track logic state for entry)
        self.in_trade = False 
        
    def calculate_signals(self, event: Bar):
        # 1. Day Management
        bar_date = event.timestamp.date()
        bar_time = event.timestamp.time()
        
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.orb_high = -1.0
            self.orb_low = 1e9
            self.traded_today = False
            self.in_trade = False
            
        # 2. Get History for Indicators
        # We need enough bars for EMA(50)
        history = self.bars.get_latest_bars(event.symbol, N=self.ema_period + 5)
        if len(history) < self.ema_period:
            return
            
        # 3. ORB Window Logic
        if self.orb_start_time <= bar_time < self.orb_end_time:
            # Update High/Low
            if self.orb_high == -1.0:
                self.orb_high = event.high
                self.orb_low = event.low
            else:
                self.orb_high = max(self.orb_high, event.high)
                self.orb_low = min(self.orb_low, event.low)
                
        # 4. Trading Window
        elif self.orb_end_time <= bar_time < self.exit_time:
            if self.in_trade:
                # Exits handled by SL/TP orders ideally, or logic here.
                # For this simple verification, we let Portfolio/Execution handle SL/TP fills.
                # But we must issue EXIT signal at EOD.
                pass
            
            elif not self.traded_today and self.orb_high != -1.0:
                # Check Entry Conditions
                
                # Retrieve Indicators
                # Note: In a real system, we'd use an incremental indicator calculator for speed.
                # Here we re-calc using pandas for simplicity/parity.
                df = pd.DataFrame([vars(b) for b in history])
                closes = df['close']
                highs = df['high']
                lows = df['low']
                
                # Calc ATR
                # Simple Manual ATR for last bar
                tr = pd.concat([
                    highs - lows,
                    (highs - closes.shift(1)).abs(),
                    (lows - closes.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
                
                # Calc EMA
                ema = closes.ewm(span=self.ema_period, adjust=False).mean().iloc[-1]
                
                current_close = event.close
                orb_range = self.orb_high - self.orb_low
                
                # Filter Checks
                entry_signal = None
                
                if orb_range <= (atr * self.atr_max_mult):
                    # Long
                    if current_close > self.orb_high and current_close > ema:
                        entry_signal = 'LONG'
                    # Short
                    elif current_close < self.orb_low and current_close < ema:
                        entry_signal = 'SHORT'
                        
                if entry_signal:
                    sl_price = 0.0
                    tp_price = 0.0
                    
                    if entry_signal == 'LONG':
                        sl_price = current_close - (atr * self.sl_mult)
                        tp_price = current_close + (atr * self.tp_mult)
                        # self.buy(event.symbol, limit_price=tp_price) 
                        self.buy(event.symbol) 
                        self.traded_today = True
                    else:
                        sl_price = current_close + (atr * self.sl_mult)
                        tp_price = current_close - (atr * self.tp_mult)
                        self.sell(event.symbol)
                        self.traded_today = True
                        
                    self.traded_today = True
                    self.in_trade = True
                    
        # 5. EOD Exit
        elif bar_time >= self.exit_time:
            if self.in_trade:
                self.exit(event.symbol)
                self.in_trade = False
