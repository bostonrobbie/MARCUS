
import sys
import os
import site

# Debug Paths
print(f"Python Executable: {sys.executable}")
# print(f"Sys Path: {sys.path}")

# Explicitly add user site-packages if not present (Workaround)
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import pandas as pd
import numpy as np
from datetime import time


# Add Src to Path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(project_root, 'StrategyPipeline'))

from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(project_root, 'StrategyPipeline'))

class EODFadeStrategy(Strategy):
    def __init__(self, bars, events, mode='VWAP_Ext', vwap_mult=2.0, sl_pct=0.0075, tp_pct=0.01, verbose=False):
        super().__init__(bars, events)
        self.mode = mode
        self.vwap_mult = vwap_mult
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0 # 0, 1, -1
        
        # State for Range Mode
        self.rng_high = None
        self.rng_low = None
        self.current_date = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date_val = ts.date()
        
        # Reset Daily State
        if self.current_date != current_date_val:
            self.current_date = current_date_val
            self.rng_high = None
            self.rng_low = None

        # 1. Update Afternoon Range (12:00 - 14:00)
        # Note: If data is 5m bars, 12:00 bar is the start.
        if time(12, 0) <= current_time < time(14, 0):
            if self.rng_high is None or event.high > self.rng_high:
                self.rng_high = event.high
            if self.rng_low is None or event.low < self.rng_low:
                self.rng_low = event.low

        # 2. Hard Exit at 15:55 (MOC)
        if current_time >= time(15, 55):
            if self.current_pos != 0:
                self.exit(symbol)
                self.current_pos = 0
            return

        # 3. Entry Window (14:00 - 15:45)
        if not (time(14, 0) <= current_time <= time(15, 45)):
            # If we are outside entry window, just manage risk, don't enter
            if self.current_pos != 0:
                self.manage_risk(event, symbol)
            return

        # 4. Data Requirements
        # Need enough bars for ATR and VWAP calc approximation
        bars = self.bars.get_latest_bars(symbol, N=100) # Get last 100 bars (~8 hours)
        if len(bars) < 20: return
        
        # 5. Indicators
        # Quick VWAP Approx: Rolling VWAP since market open (9:30)?
        # Since we don't have easy "Daily" access in this event loop without overhead,
        # we will calculate VWAP from the loaded bars that belong to TODAY.
        
        # Filter bars for today
        todays_bars = [b for b in bars if b.timestamp.date() == current_date_val]
        if not todays_bars: return
        
        closes = np.array([b.close for b in todays_bars])
        highs = np.array([b.high for b in todays_bars])
        lows = np.array([b.low for b in todays_bars])
        volumes = np.array([b.volume for b in todays_bars])
        
        # VWAP
        cum_vol = np.cumsum(volumes)
        cum_pv  = np.cumsum(closes * volumes)
        # Avoid div by zero
        vwap = cum_pv / np.where(cum_vol == 0, 1, cum_vol)
        current_vwap = vwap[-1]
        
        # ATR (14 of available)
        if len(todays_bars) > 1:
            tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
            atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        else:
            atr = highs[0] - lows[0]

        current_price = event.close
        
        # 6. Entry Logic
        if self.current_pos == 0:
            
            # --- MODE A: VWAP EXTENSION ---
            if self.mode == 'VWAP_Ext':
                upper_band = current_vwap + (self.vwap_mult * atr)
                lower_band = current_vwap - (self.vwap_mult * atr)
                
                if current_price > upper_band:
                    # Short Fade
                    if self.verbose: print(f"[{ts}] VWAP Ext Short: {current_price} > {upper_band:.2f}")
                    self.sell(symbol, 1)
                    self.entry_price = current_price
                    self.current_pos = -1
                    
                elif current_price < lower_band:
                    # Long Fade
                    if self.verbose: print(f"[{ts}] VWAP Ext Long: {current_price} < {lower_band:.2f}")
                    self.buy(symbol, 1)
                    self.entry_price = current_price
                    self.current_pos = 1

            # --- MODE B: AFTERNOON RANGE FADE ---
            elif self.mode == 'Range_Fade':
                if self.rng_high and self.rng_low:
                    # False Break High: High > RngHigh, Close < RngHigh
                    # Using current bar logic (event is the just completed bar)
                    if event.high > self.rng_high and event.close < self.rng_high:
                         # Short
                         if self.verbose: print(f"[{ts}] False Break High Short")
                         self.sell(symbol, 1)
                         self.entry_price = current_price
                         self.current_pos = -1
                    
                    elif event.low < self.rng_low and event.close > self.rng_low:
                         # Long
                         if self.verbose: print(f"[{ts}] False Break Low Long")
                         self.buy(symbol, 1)
                         self.entry_price = current_price
                         self.current_pos = 1

        else:
            self.manage_risk(event, symbol)

    def manage_risk(self, event, symbol):
        if self.current_pos > 0:
            sl = self.entry_price * (1 - self.sl_pct)
            tp = self.entry_price * (1 + self.tp_pct)
            if event.low < sl:
                self.exit(symbol)
                self.current_pos = 0
            elif event.high > tp:
                self.exit(symbol)
                self.current_pos = 0
        elif self.current_pos < 0:
            sl = self.entry_price * (1 + self.sl_pct)
            tp = self.entry_price * (1 - self.tp_pct)
            if event.high > sl:
                self.exit(symbol)
                self.current_pos = 0
            elif event.low < tp:
                self.exit(symbol)
                self.current_pos = 0

def run_grid_search():
    start_date = "2020-01-01" # Test last 4-5 years
    end_date = "2024-12-31"
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    # Load Data Once
    try:
        ref_data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    except Exception as e:
        print(f"Data Init Error: {e}")
        return

    print(f"Loaded Data: {start_date} -> {end_date}")
    
    # 1. Test VWAP Extension Mode
    print("\n--- Testing VWAP Extension Mode ---")
    multipliers = [1.5, 2.0, 2.5, 3.0]
    best_ret = -999
    best_params = None
    
    for mult in multipliers:
        events = queue.Queue()
        # Cloning data logic or re-init? simplified: reuse logic
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = EODFadeStrategy(data, events, mode='VWAP_Ext', vwap_mult=mult, verbose=False)
        exec_handler = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_handler)
        engine.run()
        
        stats = TearSheet(port).analyze()
        ret = stats.get('Total Return', 0)
        dd = stats.get('Max Drawdown', 0)
        print(f"Mult: {mult} | Return: {ret:.2%} | MaxDD: {dd:.2%}")
        
        if ret > best_ret:
            best_ret = ret
            best_params = mult
            
    print(f"Best VWAP Mult: {best_params} (Ret: {best_ret:.2%})")

    # 2. Test Range Fade Mode
    print("\n--- Testing Afternoon Range Fade Mode ---")
    events2 = queue.Queue()
    data2 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port2 = Portfolio(data2, events2, initial_capital=100000.0)
    strat2 = EODFadeStrategy(data2, events2, mode='Range_Fade', verbose=False)
    exec2 = SimulatedExecutionHandler(events2, data2, commission_model=FixedCommission(2.05))
    engine2 = BacktestEngine(data2, strat2, port2, exec2)
    engine2.run()
    
    stats2 = TearSheet(port2).analyze()
    print(f"Range Fade | Return: {stats2.get('Total Return', 0):.2%} | MaxDD: {stats2.get('Max Drawdown', 0):.2%}")

if __name__ == "__main__":
    run_grid_search()
