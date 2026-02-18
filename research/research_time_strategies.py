
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
# 1. Overnight Hold (Globex Drift)
# ==========================================
class OvernightHoldStrategy(Strategy):
    def __init__(self, bars, events, verbose=False):
        super().__init__(bars, events)
        self.verbose = verbose
        self.current_pos = 0
        self.entry_price = 0.0

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # Logic:
        # Buy @ 15:55 (Close of RTH)
        # Sell @ 9:30 (Open of Next RTH)
        
        # Note on 5m bars:
        # 15:55 timestamp usually represents bar 15:55-16:00. Close of this bar is ~16:00 close.
        # 9:30 timestamp usually represents bar 9:30-9:35. Open of this bar is ~9:30 open.
        
        if current_time == time(15, 55):
            if self.current_pos == 0:
                if self.verbose: print(f"[{ts}] Overnight Buy (MOC)")
                self.buy(symbol, 1)
                self.entry_price = event.close
                self.current_pos = 1
                
        elif current_time == time(9, 30):
            if self.current_pos != 0:
                if self.verbose: print(f"[{ts}] Overnight Sell (Open)")
                self.exit(symbol)
                self.current_pos = 0

# ==========================================
# 2. 10am Reversal Strategy
# ==========================================
class Reversal10amStrategy(Strategy):
    def __init__(self, bars, events, fade_threshold_pts=20, sl_pts=30, tp_pts=60, verbose=False):
        super().__init__(bars, events)
        self.fade_threshold_pts = fade_threshold_pts
        self.sl_pts = sl_pts
        self.tp_pts = tp_pts 
        self.verbose = verbose
        
        self.open_price_930 = None
        self.current_pos = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.last_date = None
        self.daily_trade_complete = False

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date = ts.date()
        
        if self.last_date != current_date:
            self.daily_trade_complete = False
            self.last_date = current_date
            self.open_price_930 = None
            
        # Hard Exit EOD
        if current_time >= time(15, 55) and self.current_pos != 0:
             self.exit(symbol)
             self.current_pos = 0
             return

        # Capture 9:30 Open
        if current_time == time(9, 30):
            self.open_price_930 = event.open
            
        # At 10:00, check trend and fade
        if current_time == time(10, 0):
            if self.open_price_930 is None: return
            if self.daily_trade_complete: return
            
            # Trend = Close(10:00) - Open(9:30)
            trend_pts = event.close - self.open_price_930
            
            # Fade Logic
            if trend_pts > self.fade_threshold_pts:
                # Big Up Move -> Sell
                if self.verbose: print(f"[{ts}] 10am Fade Sell. Trend +{trend_pts:.1f}")
                self.sell(symbol, 1)
                self.current_pos = -1
                self.entry_price = event.close
                self.stop_price = self.entry_price + self.sl_pts
                self.target_price = self.entry_price - self.tp_pts
                self.daily_trade_complete = True
                
            elif trend_pts < -self.fade_threshold_pts:
                # Big Down Move -> Buy
                if self.verbose: print(f"[{ts}] 10am Fade Buy. Trend {trend_pts:.1f}")
                self.buy(symbol, 1)
                self.current_pos = 1
                self.entry_price = event.close
                self.stop_price = self.entry_price - self.sl_pts
                self.target_price = self.entry_price + self.tp_pts
                self.daily_trade_complete = True
        
        # Manage Trade
        if self.current_pos != 0:
            if self.current_pos > 0:
                if event.high >= self.target_price:
                    self.exit(symbol)
                    self.current_pos = 0
                elif event.low <= self.stop_price:
                    self.exit(symbol)
                    self.current_pos = 0
            elif self.current_pos < 0:
                if event.low <= self.target_price:
                    self.exit(symbol)
                    self.current_pos = 0
                elif event.high >= self.stop_price:
                    self.exit(symbol)
                    self.current_pos = 0

# ==========================================
# Runner
# ==========================================
def run_time_research(start_date, end_date):
    print(f"Running Time-Based Research: {start_date} to {end_date}")
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd(), 'examples'), os.path.join(os.getcwd())]
    symbol_list = ['NQ']

    # --- Strat 1: Overnight ---
    print("\n>>> Testing Overnight Hold (Long)...")
    events_1 = queue.Queue()
    data1 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port1 = Portfolio(data1, events_1, initial_capital=100000.0)
    strat1 = OvernightHoldStrategy(data1, events_1, verbose=False)
    exec1 = SimulatedExecutionHandler(events_1, data1, commission_model=FixedCommission(2.05))
    engine1 = BacktestEngine(data1, strat1, port1, exec1)
    engine1.run()
    
    ts1 = TearSheet(port1)
    stats1 = ts1.analyze()
    print(f"Overnight Return: {stats1.get('Total Return', 0):.2%}")
    print(f"Overnight MaxDD:  {stats1.get('Max Drawdown', 0):.2%}")
    
    # --- Strat 2: 10am Reversal ---
    print("\n>>> Testing 10am Reversal (Fade)...")
    events_2 = queue.Queue()
    data2 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port2 = Portfolio(data2, events_2, initial_capital=100000.0)
    strat2 = Reversal10amStrategy(data2, events_2, verbose=False)
    exec2 = SimulatedExecutionHandler(events_2, data2, commission_model=FixedCommission(2.05))
    engine2 = BacktestEngine(data2, strat2, port2, exec2)
    engine2.run()
    
    ts2 = TearSheet(port2)
    stats2 = ts2.analyze()
    print(f"10am Reversal Return: {stats2.get('Total Return', 0):.2%}")
    print(f"10am Reversal MaxDD:  {stats2.get('Max Drawdown', 0):.2%}")

if __name__ == "__main__":
    run_time_research("2015-01-01", "2024-12-31")
