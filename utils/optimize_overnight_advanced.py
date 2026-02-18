
import sys
import os
import site

# Add Src to Path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(project_root, 'StrategyPipeline'))

# Explicitly add user site-packages
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import pandas as pd
import numpy as np
from datetime import time
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

class OvernightAdvancedStrategy(Strategy):
    def __init__(self, bars, events, 
                 close_loc_thresh=0.0, # 0.0 = Off. 0.5 = Upper Half. 0.8 = Upper 20%.
                 require_pm_mom=False, # Require 16:00 close > 13:00 close
                 require_green_day=False, # Require Close > Open
                 verbose=False):
        super().__init__(bars, events)
        self.close_loc_thresh = close_loc_thresh
        self.require_pm_mom = require_pm_mom
        self.require_green_day = require_green_day
        self.verbose = verbose
        
        self.current_pos = 0
        self.current_date = None
        self.daily_high = -1.0
        self.daily_low = 999999.0
        self.day_open = None
        self.pm_open = None # Open at 13:00 (approx)

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date_val = ts.date()
        
        # 1. Update Daily Stats
        if self.current_date != current_date_val:
            self.current_date = current_date_val
            self.daily_high = event.high
            self.daily_low = event.low
            self.day_open = event.open 
            self.pm_open = None
        else:
            if event.high > self.daily_high: self.daily_high = event.high
            if event.low < self.daily_low: self.daily_low = event.low
            
        # Capture ~13:00 Price for PM Momentum
        if current_time >= time(13, 0) and self.pm_open is None:
            self.pm_open = event.open

        # 2. Exit Logic (Market Open Next Day)
        # Standard Overnight: Exit at 09:30 or 09:35 next day. Or 09:00?
        # Let's say we hold until 09:30 for standard overlap.
        # Actually drift is usually until Open. 
        if current_time >= time(9, 30) and current_time < time(10, 0):
             if self.current_pos != 0:
                self.exit(symbol)
                self.current_pos = 0
             return

        # 3. Entry Logic (MOC)
        # Enter at 15:55
        if current_time >= time(15, 55) and current_time < time(16, 0):
            if self.current_pos == 0:
                
                # CHECKS
                
                # A. Close Location
                if self.close_loc_thresh > 0:
                    rng = self.daily_high - self.daily_low
                    if rng > 0:
                        loc = (event.close - self.daily_low) / rng
                        if loc < self.close_loc_thresh:
                            return # Filtered
                            
                # B. PM Momentum
                if self.require_pm_mom:
                    if self.pm_open and event.close < self.pm_open:
                        return # Filtered (PM Session was red)
                        
                # C. Daily Candle Color
                if self.require_green_day:
                    if self.day_open and event.close < self.day_open:
                        return # Filtered (Red Day)
                
                # ENTRY
                self.buy(symbol, 1) # Fixed 1 contract
                self.current_pos = 1

def run_optimization():
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    print(f"\nOVERNIGHT ADVANCED OPTIMIZATION: {start_date} to {end_date}")
    
    # Baseline
    print("\n--- Baseline (No Filters) ---")
    _run_test(symbol_list, search_dirs, start_date, end_date, {})

    # 1. Close Location
    print("\n--- 1. Close Location Filter ---")
    locs = [0.5, 0.7, 0.8, 0.9]
    for l in locs:
        print(f"Testing Close Loc > {l}...")
        _run_test(symbol_list, search_dirs, start_date, end_date, {'close_loc_thresh': l})
        
    # 2. PM Momentum
    print("\n--- 2. PM Momentum (13:00-16:00 Green) ---")
    _run_test(symbol_list, search_dirs, start_date, end_date, {'require_pm_mom': True})
    
    # 3. Green Day
    print("\n--- 3. Green Day (Close > Open) ---")
    _run_test(symbol_list, search_dirs, start_date, end_date, {'require_green_day': True})

def _run_test(symbol_list, search_dirs, start_date, end_date, params):
    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port = Portfolio(data, events, initial_capital=100000.0)
    strat = OvernightAdvancedStrategy(data, events, **params)
    exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
    engine = BacktestEngine(data, strat, port, exec_h)
    engine.run()
    
    stats = TearSheet(port).analyze()
    print(f"Params: {params} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

if __name__ == "__main__":
    run_optimization()
