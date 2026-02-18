import pandas as pd
import numpy as np
import sys
import os
import queue
from datetime import time, datetime

# Ensure we can import the backtesting package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.strategy import Strategy
from backtesting.schema import Bar, SignalType
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet

class NqOrb(Strategy):
    """
    Professional Opening Range Breakout (ORB) for NQ Futures.
    
    Hypothesis:
    Large institutions often define the daily trend within the first 30 minutes 
    of the US Cash Open. A breakout of this range with momentum often leads 
    to a trending day.
    """
    def __init__(self, bars, events, 
                 orb_start_time=time(9, 30), 
                 orb_end_time=time(10, 0),
                 exit_time=time(15, 55),
                 stop_loss=75.0,
                 take_profit=100.0,
                 ema_filter=50,
                 atr_filter=14,
                 atr_max_mult=3.0):
        super().__init__(bars, events)
        self.orb_start = orb_start_time
        self.orb_end = orb_end_time
        self.exit_time = exit_time
        
        # State tracking
        self.orb_high = None
        self.orb_low = None
        self.current_date = None
        self.traded_today = False
        self.entry_price = None
        self.in_long = False
        self.in_short = False
        
        # Strategy Parameters
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)
        self.ema_filter = int(ema_filter)
        self.atr_filter = int(atr_filter)
        self.atr_max_mult = float(atr_max_mult)

    def calculate_signals(self, event: Bar):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # 1. Daily State Management
        if self.current_date != ts.date():
            self.current_date = ts.date()
            self.orb_high = -float('inf')
            self.orb_low = float('inf')
            self.traded_today = False
            self.entry_price = None
            self.in_long = False
            self.in_short = False

        # 2. Define the Range (09:30 - 10:00)
        if self.orb_start <= current_time < self.orb_end:
            self.orb_high = max(self.orb_high, event.high)
            self.orb_low = min(self.orb_low, event.low)
            return

        # 3. Indicator Calculation
        lookback = max(self.ema_filter, self.atr_filter) + 5
        bars = self.bars.get_latest_bars(symbol, N=lookback)
        if len(bars) < lookback:
            return
            
        df = pd.DataFrame([{
            'close': b.close, 'high': b.high, 'low': b.low, 'open': b.open
        } for b in bars])
        
        try:
            import pandas_ta as ta
        except ImportError:
            return

        ema = ta.ema(df['close'], length=self.ema_filter).iloc[-1]
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_filter).iloc[-1]

        # 4. Execution Window (10:00 - 15:55)
        if self.orb_end <= current_time < self.exit_time:
            if not self.traded_today:
                # Entry Logic with Filters
                range_size = self.orb_high - self.orb_low
                
                # Long: Price > Range High AND Close > EMA AND Range is not too wide
                if event.close > self.orb_high and event.close > ema:
                    if range_size <= (atr * self.atr_max_mult):
                        print(f"[{ts}] NQ ORB LONG ENTRY @ {event.close} | EMA: {ema:.2f}, Range: {range_size:.1f}")
                        self.buy(symbol, quantity=1)
                        self.traded_today = True
                        self.in_long = True
                        self.entry_price = event.close
                
                # Short: Price < Range Low AND Close < EMA AND Range is not too wide
                elif event.close < self.orb_low and event.close < ema:
                    if range_size <= (atr * self.atr_max_mult):
                        print(f"[{ts}] NQ ORB SHORT ENTRY @ {event.close} | EMA: {ema:.2f}, Range: {range_size:.1f}")
                        self.sell(symbol, quantity=1)
                        self.traded_today = True
                        self.in_short = True
                        self.entry_price = event.close
            
            elif self.in_long or self.in_short:
                # 5. Risk Management
                if self.in_long:
                    if event.low < (self.entry_price - self.stop_loss):
                        print(f"[{ts}] NQ STOP LOSS (LONG) @ {event.low}")
                        self.exit(symbol)
                        self.in_long = False
                    elif event.high > (self.entry_price + self.take_profit):
                        print(f"[{ts}] NQ TAKE PROFIT (LONG) @ {event.high}")
                        self.exit(symbol)
                        self.in_long = False
                elif self.in_short:
                    if event.high > (self.entry_price + self.stop_loss):
                        print(f"[{ts}] NQ STOP LOSS (SHORT) @ {event.high}")
                        self.exit(symbol)
                        self.in_short = False
                    elif event.low < (self.entry_price - self.take_profit):
                        print(f"[{ts}] NQ TAKE PROFIT (SHORT) @ {event.low}")
                        self.exit(symbol)
                        self.in_short = False

        # 5. End of Session Flatten (15:55)
        if current_time >= self.exit_time:
            if self.in_long or self.in_short:
                print(f"[{ts}] SESSION END: Liquidating Position.")
                self.exit(symbol)
                self.in_long = False
                self.in_short = False

if __name__ == "__main__":
    # Settings for NQ Intraday
    symbol_list = ['NQ']
    interval = '1m' # Minute bars for precision
    data_dir = r"C:\Users\User\Desktop\Portfolio\OHLC\Intra OHLC"
    
    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=[data_dir], interval=interval)
    
    # NQ Multiplier is $20 per point
    instruments = {'NQ': {'multiplier': 20.0}}
    portfolio = Portfolio(data, events, initial_capital=250000.0, instruments=instruments)
    
    strategy = NqOrb(data, events)
    execution = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(1.0))
    
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    
    tearsheet = TearSheet(portfolio)
    tearsheet.analyze()
