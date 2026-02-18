
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

class ExhaustionFadeStrategy(Strategy):
    def __init__(self, bars, events, adr_mult=2.0, verbose=False):
        super().__init__(bars, events)
        self.adr_mult = adr_mult
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0
        
        # Daily State
        self.current_date = None
        self.daily_high = -1.0
        self.daily_low = 999999.0
        
    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date_val = ts.date()
        
        # 1. Update Daily High/Low
        if self.current_date != current_date_val:
            self.current_date = current_date_val
            self.daily_high = event.high
            self.daily_low = event.low
        else:
            if event.high > self.daily_high: self.daily_high = event.high
            if event.low < self.daily_low: self.daily_low = event.low
            
        # 2. Hard Exit (MOC)
        if current_time >= time(15, 55):
            if self.current_pos != 0:
                self.exit(symbol)
                self.current_pos = 0
            return
            
        # 3. Entry Window (Late Day Exhaustion? 13:00+)
        if current_time < time(13, 0): return
        if current_time > time(15, 45): return
        
        # 4. Calculate ADR (Approximate)
        bars = self.bars.get_latest_bars(symbol, N=2000) 
        if len(bars) < 100: return
        
        todays_bars = [b for b in bars if b.timestamp.date() == current_date_val]
        
        # Create Daily DF from history
        df = pd.DataFrame([{'dt': b.timestamp, 'high': b.high, 'low': b.low, 'close': b.close} for b in bars])
        df['date'] = df['dt'].dt.date
        daily_df = df.groupby('date').agg({'high':'max', 'low':'min', 'close':'last'})
        
        # Exclude today from ADR calc
        prev_days = daily_df[daily_df.index < current_date_val]
        if len(prev_days) < 10: return
        
        # Range
        prev_days['rng'] = prev_days['high'] - prev_days['low']
        adr = prev_days['rng'].tail(14).mean()
        
        if adr == 0 or np.isnan(adr): return

        # Current Range
        current_rng = self.daily_high - self.daily_low
        
        # 5. Logic
        if current_rng > (adr * self.adr_mult):
            
            # Threshold for "Near" extremes (top/bottom 10% of current range)
            threshold = current_rng * 0.10
            
            if self.current_pos == 0:
                dist_to_high = self.daily_high - event.close
                dist_to_low  = event.close - self.daily_low

                # FADE SHORT (At Highs)
                if dist_to_high < threshold:
                     # if self.verbose: print(f"[{ts}] Exhaustion Short: Rng {current_rng:.2f} > {self.adr_mult}x ADR")
                     self.sell(symbol, 1)
                     self.entry_price = event.close
                     self.current_pos = -1
                
                # FADE LONG (At Lows)
                elif dist_to_low < threshold:
                     # if self.verbose: print(f"[{ts}] Exhaustion Long: Rng {current_rng:.2f} > {self.adr_mult}x ADR")
                     self.buy(symbol, 1)
                     self.entry_price = event.close
                     self.current_pos = 1
                     
        # Risk Management (Fixed % Stop/TP)
        if self.current_pos != 0:
            self.manage_risk(event, symbol)

    def manage_risk(self, event, symbol):
        sl_pct = 0.0075
        tp_pct = 0.015
        
        if self.current_pos > 0:
            sl = self.entry_price * (1 - sl_pct)
            tp = self.entry_price * (1 + tp_pct)
            if event.low < sl:
                self.exit(symbol)
                self.current_pos = 0
            elif event.high > tp:
                self.exit(symbol)
                self.current_pos = 0
        elif self.current_pos < 0:
            sl = self.entry_price * (1 + sl_pct)
            tp = self.entry_price * (1 - tp_pct)
            if event.high > sl:
                self.exit(symbol)
                self.current_pos = 0
            elif event.low < tp:
                self.exit(symbol)
                self.current_pos = 0

def run_exhaustion_research():
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    print(f"\nEXHAUSTION RESEARCH: {start_date} to {end_date}")
    
    # Test aggressively high multiples (Exhaustion)
    multipliers = [1.5, 2.0, 2.5]
    
    for m in multipliers:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = ExhaustionFadeStrategy(data, events, adr_mult=m)
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"ADR Mult: {m} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

if __name__ == "__main__":
    run_exhaustion_research()
