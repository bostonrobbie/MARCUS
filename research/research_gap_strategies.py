
import sys
import os
from datetime import time, timedelta

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

import pandas as pd
import numpy as np
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

# ==========================================
# 1. Gap Fill Strategy
# ==========================================
class GapFillStrategy(Strategy):
    def __init__(self, bars, events, min_gap_pct=0.15, stop_pct=0.5, verbose=False):
        super().__init__(bars, events)
        self.min_gap_pct = min_gap_pct
        self.stop_pct = stop_pct
        self.verbose = verbose
        
        self.prev_close = None
        self.current_pos = 0
        self.entry_price = 0.0
        self.target_price = 0.0
        self.stop_price = 0.0
        self.daily_trade_complete = False
        self.last_date = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date = ts.date()
        
        # Reset daily state
        if self.last_date != current_date:
            self.daily_trade_complete = False
            self.last_date = current_date
            
        # Capture Previous Close (16:00 bar or 15:55 bar depending on data)
        # We assume the data stream is continuous. We need the close of the LAST session.
        # Ideally, we look for the last bar before 17:00 of the previous day.
        # Simplified: We treat the first bar of the day (9:30) as the "Gap" check moment.
        # We need to know what the previous close was.
        
        # In this event loop, we see bars sequentially.
        # If time is 15:55 or 16:00, store close as prev_close.
        if current_time == time(15, 55) or current_time == time(16, 0): 
            self.prev_close = event.close
            
        # Exit EOD
        if current_time >= time(15, 55) and self.current_pos != 0:
            self.exit(symbol)
            self.current_pos = 0
            return

        # Trade Logic at Open (9:30)
        if current_time == time(9, 30):
            if self.prev_close is None: return # No history
            if self.daily_trade_complete: return
            
            open_price = event.open 
            # Note: event.close is the close of the 9:30 5m bar (9:35).
            # Realistically we trade the OPEN of this bar, but we only receive the event at the end.
            # So we are trading at 9:35 based on the gap at 9:30.
            # Wait, standard backtesting constraint: We get the bar AFTER it closes.
            # So at 9:30 timestamp (which usually means 9:30->9:35 bar), we know the Open.
            # Gap = Event.Open - Prev_Close.
            # Signal: We enter at Close of this bar (Market) or Next Open.
            
            gap_pct = (open_price - self.prev_close) / self.prev_close * 100
            
            if abs(gap_pct) >= self.min_gap_pct:
                # REJECTION FILTER: Only Enter if First Bar confirms Reversion
                # Gap Up -> Short -> Need Red Candle (Close < Open)
                # Gap Down -> Long -> Need Green Candle (Close > Open)
                
                if gap_pct > 0:
                    # Gap Up -> Short to Fill
                    if event.close < event.open: # Rejection confirmed
                        if self.verbose: print(f"[{ts}] GAP UP {gap_pct:.2f}% (Red Bar). Selling.")
                        self.sell(symbol, 1)
                        self.current_pos = -1
                        self.entry_price = event.close 
                        self.target_price = self.prev_close
                        self.stop_price = self.entry_price * (1 + self.stop_pct/100)
                        self.daily_trade_complete = True
                    else:
                        if self.verbose: print(f"[{ts}] GAP UP {gap_pct:.2f}% (Green Bar). SKIPPED.")
                    
                else:
                    # Gap Down -> Buy to Fill
                    if event.close > event.open: # Support confirmed
                        if self.verbose: print(f"[{ts}] GAP DOWN {gap_pct:.2f}% (Green Bar). Buying.")
                        self.buy(symbol, 1)
                        self.current_pos = 1
                        self.entry_price = event.close
                        self.target_price = self.prev_close
                        self.stop_price = self.entry_price * (1 - self.stop_pct/100)
                        self.daily_trade_complete = True
                    else:
                        if self.verbose: print(f"[{ts}] GAP DOWN {gap_pct:.2f}% (Red Bar). SKIPPED.")
                    
        # Manage Trade
        if self.current_pos != 0:
            if self.current_pos > 0:
                if event.high >= self.target_price:
                    self.exit(symbol) # Fill reached
                    self.current_pos = 0
                elif event.low <= self.stop_price:
                    self.exit(symbol) # Stop hit
                    self.current_pos = 0
            elif self.current_pos < 0:
                if event.low <= self.target_price:
                    self.exit(symbol) # Fill reached
                    self.current_pos = 0
                elif event.high >= self.stop_price:
                    self.exit(symbol) # Stop hit
                    self.current_pos = 0

# ==========================================
# 2. Gap Support Strategy (Trend)
# ==========================================
class GapSupportStrategy(Strategy):
    def __init__(self, bars, events, min_gap_pct=0.25, verbose=False):
        super().__init__(bars, events)
        self.min_gap_pct = min_gap_pct
        self.verbose = verbose
        
        self.prev_close = None
        self.current_pos = 0
        self.entry_price = 0.0
        self.daily_trade_complete = False
        self.last_date = None
        
        # Logic State
        self.gap_direction = 0 # 1 Up, -1 Down
        self.gap_zone_start = 0.0
        self.gap_zone_end = 0.0
        self.waiting_for_pullback = False

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date = ts.date()
        
        if self.last_date != current_date:
            self.daily_trade_complete = False
            self.last_date = current_date
            self.waiting_for_pullback = False
            self.gap_direction = 0
            
        if current_time == time(15, 55) or current_time == time(16, 0): 
            self.prev_close = event.close
            
        if current_time >= time(15, 45) and self.current_pos != 0:
            self.exit(symbol)
            self.current_pos = 0
            return

        # Check Gap at 9:30
        if current_time == time(9, 30):
            if self.prev_close is None: return
            
            gap_pct = (event.open - self.prev_close) / self.prev_close * 100
            
            if abs(gap_pct) >= self.min_gap_pct:
                if gap_pct > 0:
                    # Gap Up
                    self.gap_direction = 1
                    # Support Zone: Between Midpoint and Prev Close
                    midpoint = (event.open + self.prev_close) / 2
                    self.gap_zone_start = event.open # Top of zone
                    self.gap_zone_end = midpoint     # Bottom of acceptable support
                    self.waiting_for_pullback = True
                else:
                    # Gap Down
                    self.gap_direction = -1
                    midpoint = (event.open + self.prev_close) / 2
                    self.gap_zone_start = event.open # Bottom of zone
                    self.gap_zone_end = midpoint     # Top of acceptable resistance
                    self.waiting_for_pullback = True
        
        # Monitor for Pullback Entry
        if self.waiting_for_pullback and not self.daily_trade_complete:
            # We want price to enter zone and bounce.
            # Simplified: If Low < ZoneStart and Close > ZoneStart (rejection of value lower) -> Buy
            
            if self.gap_direction == 1: # Long
                # Zone is below current price.
                # If we dip into zone?
                if event.low < self.gap_zone_start and event.close > self.gap_zone_end: # Valid bounce area?
                    # Check for "Green Candle" Bounce
                    if event.close > event.open:
                         if self.verbose: print(f"[{ts}] Gap Support Buy: Dip to {event.low} held.")
                         self.buy(symbol, 1)
                         self.current_pos = 1
                         self.waiting_for_pullback = False
                         self.daily_trade_complete = True
                # Fail condition: Filled gap (dropped below prev close)
                if event.close < self.prev_close:
                    self.waiting_for_pullback = False # Setup failed
                    
            elif self.gap_direction == -1: # Short
                if event.high > self.gap_zone_start and event.close < self.gap_zone_end:
                    # Check for "Red Candle" Rejection
                    if event.close < event.open:
                        if self.verbose: print(f"[{ts}] Gap Resist Sell: Rally to {event.high} rejected.")
                        self.sell(symbol, 1)
                        self.current_pos = -1
                        self.waiting_for_pullback = False
                        self.daily_trade_complete = True
                if event.close > self.prev_close:
                     self.waiting_for_pullback = False

        # Stop Loss for Support Trades
        if self.current_pos != 0:
            # If we bought gap support, stop is Gap Fill (Prev Close)
            if self.current_pos > 0:
                if event.close < self.prev_close:
                    self.exit(symbol)
                    self.current_pos = 0
            elif self.current_pos < 0:
                if event.close > self.prev_close:
                    self.exit(symbol)
                    self.current_pos = 0

# ==========================================
# Runner
# ==========================================
def run_gap_research(start_date, end_date):
    print(f"Running Gap Research: {start_date} to {end_date}")
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd(), 'examples'), os.path.join(os.getcwd())]
    symbol_list = ['NQ']

    # --- Strat 1: Gap Fill ---
    print("\n>>> Testing Gap Fill (Fade)...")
    events_1 = queue.Queue()
    data1 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port1 = Portfolio(data1, events_1, initial_capital=100000.0)
    strat1 = GapFillStrategy(data1, events_1, verbose=False)
    exec1 = SimulatedExecutionHandler(events_1, data1, commission_model=FixedCommission(2.05))
    engine1 = BacktestEngine(data1, strat1, port1, exec1)
    engine1.run()
    
    ts1 = TearSheet(port1)
    stats1 = ts1.analyze()
    print(f"Gap Fill Return: {stats1.get('Total Return', 0):.2%}")
    print(f"Gap Fill MaxDD:  {stats1.get('Max Drawdown', 0):.2%}")
    
    # --- Strat 2: Gap Support ---
    print("\n>>> Testing Gap Support (Trend)...")
    events_2 = queue.Queue()
    data2 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port2 = Portfolio(data2, events_2, initial_capital=100000.0)
    strat2 = GapSupportStrategy(data2, events_2, verbose=False)
    exec2 = SimulatedExecutionHandler(events_2, data2, commission_model=FixedCommission(2.05))
    engine2 = BacktestEngine(data2, strat2, port2, exec2)
    engine2.run()
    
    ts2 = TearSheet(port2)
    stats2 = ts2.analyze()
    print(f"Gap Support Return: {stats2.get('Total Return', 0):.2%}")
    print(f"Gap Support MaxDD:  {stats2.get('Max Drawdown', 0):.2%}")
    
    # Winner
    ret1 = stats1.get('Total Return', 0)
    ret2 = stats2.get('Total Return', 0)
    if ret1 > ret2 and ret1 > 0:
        print("\nWinner: Gap Fill")
    elif ret2 > ret1 and ret2 > 0:
        print("\nWinner: Gap Support")
    else:
        print("\nBoth Failed or Negative")

if __name__ == "__main__":
    run_gap_research("2015-01-01", "2024-12-31")
