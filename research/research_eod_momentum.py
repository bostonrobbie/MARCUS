
import sys
import os
import site

# Add Src to Path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(project_root, 'StrategyPipeline'))

# Explicitly add user site-packages if not present
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

class EODMomentumStrategy(Strategy):
    def __init__(self, bars, events, mode='VWAP_Follow', vwap_mult=2.0, rng_end_hour=14, sl_pct=0.0075, tp_pct=0.015, verbose=False):
        super().__init__(bars, events)
        self.mode = mode
        self.vwap_mult = vwap_mult
        self.rng_end_hour = rng_end_hour
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct # Extended TP for momentum?
        
        self.entry_price = 0.0
        self.current_pos = 0 
        
        self.rng_high = None
        self.rng_low = None
        self.current_date = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date_val = ts.date()
        
        if self.current_date != current_date_val:
            self.current_date = current_date_val
            self.rng_high = None
            self.rng_low = None

        # 1. Range Update (12:00 -> rng_end_hour)
        if time(12, 0) <= current_time < time(self.rng_end_hour, 0):
            if self.rng_high is None or event.high > self.rng_high:
                self.rng_high = event.high
            if self.rng_low is None or event.low < self.rng_low:
                self.rng_low = event.low

        # 2. Hard Exit MOC (15:55)
        if current_time >= time(15, 55):
            if self.current_pos != 0:
                self.exit(symbol)
                self.current_pos = 0
            return

        # 3. Entry Window
        # Starts after range end, ends at 15:45
        if not (time(self.rng_end_hour, 0) <= current_time <= time(15, 45)):
            if self.current_pos != 0:
                self.manage_risk(event, symbol)
            return

        bars = self.bars.get_latest_bars(symbol, N=100)
        todays_bars = [b for b in bars if b.timestamp.date() == current_date_val]
        if len(todays_bars) < 5: return # Need some data
        
        closes = np.array([b.close for b in todays_bars])
        volumes = np.array([b.volume for b in todays_bars])
        highs = np.array([b.high for b in todays_bars])
        lows = np.array([b.low for b in todays_bars])
        
        # VWAP
        cum_vol = np.cumsum(volumes)
        cum_pv  = np.cumsum(closes * volumes)
        vwap = cum_pv / np.where(cum_vol == 0, 1, cum_vol)
        current_vwap = vwap[-1]
        
        # ATR
        if len(todays_bars) > 1:
            tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
            atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        else:
            atr = highs[0] - lows[0]
            
        current_price = event.close

        # ENTRY LOGIC
        if self.current_pos == 0:
            
            # MODE A: VWAP FOLLOW (Momentum)
            if self.mode == 'VWAP_Follow':
                upper = current_vwap + (self.vwap_mult * atr)
                lower = current_vwap - (self.vwap_mult * atr)
                
                # If price breaks ABOVE upper band -> BUY
                if current_price > upper:
                    self.buy(symbol, 1)
                    self.entry_price = current_price
                    self.current_pos = 1
                # If price breaks BELOW lower band -> SELL
                elif current_price < lower:
                    self.sell(symbol, 1)
                    self.entry_price = current_price
                    self.current_pos = -1

            # MODE B: RANGE BREAKOUT
            elif self.mode == 'Range_Breakout':
                if self.rng_high and self.rng_low:
                    # Breakout High -> BUY
                    if current_price > self.rng_high:
                        self.buy(symbol, 1)
                        self.entry_price = current_price
                        self.current_pos = 1
                    # Breakout Low -> SELL
                    elif current_price < self.rng_low:
                        self.sell(symbol, 1)
                        self.entry_price = current_price
                        self.current_pos = -1

        else:
            self.manage_risk(event, symbol)

    def manage_risk(self, event, symbol):
        # Basic Fixed SL/TP for consistency baseline
        # For momentum, usually wider TP or Trailing is better, but let's stick to 2R
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

def run_research():
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    # 1. Test VWAP Follow
    print("\n=== 1. VWAP Follow (Momentum) Variations ===")
    # Variations: Mult
    mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for m in mults:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        # Using rng_end_hour=13 just to allow early entry if using VWAP? 
        # Actually VWAP Follow usually is valid all afternoon. Let's set start 14:00 (default)
        strat = EODMomentumStrategy(data, events, mode='VWAP_Follow', vwap_mult=m, rng_end_hour=14)
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"VWAP Mult: {m} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

    # 2. Test Range Breakout
    print("\n=== 2. Afternoon Range Breakout Variations ===")
    # Variations: Range End Time (when does breakout mode start?)
    range_ends = [13, 14] # 13:00 and 14:00
    
    for r_end in range_ends:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = EODMomentumStrategy(data, events, mode='Range_Breakout', rng_end_hour=r_end)
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"Range End: {r_end}:00 | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

if __name__ == "__main__":
    run_research()
