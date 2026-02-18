
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
from datetime import time, timedelta
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

class AlphaSeekerStrategy(Strategy):
    def __init__(self, bars, events, mode='Lunch_Rev', params=None, verbose=False):
        super().__init__(bars, events)
        self.mode = mode
        self.params = params or {}
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0
        self.current_date = None
        
        # Gap State
        self.yest_close = None
        self.today_open = None
        
        # Vol Crunch State
        self.bb_sqz_threshold = self.params.get('bb_sqz', 0.02) # Bandwidth %

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date = ts.date()
        
        # Daily Reset / Gap Logic
        if self.current_date != current_date:
            self.current_date = current_date
            # Assuming event.close from previous bar (last of yesterday) is tough to get cleanly here 
            # without persistent state from previous tick.
            # Simplified: Use history.
            pass
            
        # Common Exit Logic (EOD)
        if current_time >= time(15, 55):
            if self.current_pos != 0:
                self.exit(symbol)
                self.current_pos = 0
            return

        # DATA PREP
        # Get enough bars
        bars = self.bars.get_latest_bars(symbol, N=200)
        if len(bars) < 50: return
        
        todays_bars = [b for b in bars if b.timestamp.date() == current_date]
        
        # ---------------------------
        # STRATEGY 1: LUNCH REVERSION
        # ---------------------------
        if self.mode == 'Lunch_Rev':
            # Window: 11:00 - 13:00
            # Logic: Bollinger Band Mean Reversion? Or RSI?
            # Let's try BB Reversion: Price touches 2.0 StdDev -> Fade to MA
            
            if not (time(11, 0) <= current_time <= time(13, 0)):
                if self.current_pos != 0: self.manage_risk(event, symbol)
                return

            closes = np.array([b.close for b in bars]) # Use longer history for MA
            
            # BB Calc (20, 2.0)
            ma = pd.Series(closes).rolling(20).mean().iloc[-1]
            std = pd.Series(closes).rolling(20).std().iloc[-1]
            upper = ma + (self.params.get('bb_std', 2.0) * std)
            lower = ma - (self.params.get('bb_std', 2.0) * std)
            
            curr = event.close
            
            if self.current_pos == 0:
                if curr > upper:
                    self.sell(symbol, 1)
                    self.entry_price = curr
                    self.current_pos = -1
                elif curr < lower:
                    self.buy(symbol, 1)
                    self.entry_price = curr
                    self.current_pos = 1
            else:
                # Target: Mean Reversion to MA
                if self.current_pos > 0 and curr >= ma:
                    self.exit(symbol)
                    self.current_pos = 0
                elif self.current_pos < 0 and curr <= ma:
                    self.exit(symbol)
                    self.current_pos = 0
                else:
                    self.manage_risk(event, symbol)


        # ---------------------------
        # STRATEGY 2: VOL CRUNCH (Compression)
        # ---------------------------
        elif self.mode == 'Vol_Crunch':
            # Logic: BB Bandwidth is extremely low (< X%). Wait for breakout?
            # Normalized BB Bandwidth = (Upper - Lower) / Middle
            
            closes = np.array([b.close for b in bars])
            ma = pd.Series(closes).rolling(20).mean().iloc[-1]
            std = pd.Series(closes).rolling(20).std().iloc[-1]
            if ma == 0: return
            
            bw = (4 * std) / ma # 2 std up, 2 std down = 4 std width
            
            is_squeezed = bw < self.params.get('sqz_thresh', 0.002) # e.g. 0.2% width extremely tight
            
            if self.current_pos == 0:
                if is_squeezed:
                    # Breakout Logic?
                    # If Close > Upper -> Long?
                    # If Close < Lower -> Short?
                    upper = ma + (2.0 * std)
                    lower = ma - (2.0 * std)
                    
                    if event.close > upper:
                        self.buy(symbol, 1)
                        self.entry_price = event.close
                        self.current_pos = 1
                    elif event.close < lower:
                        self.sell(symbol, 1)
                        self.entry_price = event.close
                        self.current_pos = -1
            else:
                self.manage_risk(event, symbol) # Or Trailing Stop


        # ---------------------------
        # STRATEGY 3: GAP FILL
        # ---------------------------
        elif self.mode == 'Gap_Fill':
            # Logic: At 9:35, define Gap (Today Open - Yest Close).
            # If Gap > Threshold, Fade it (Gap Fill)?
            # Or Follow it?
            
            # Identify "Yesterday's Close"
            # Getting accurate YC from bar stream in backtester is tricky if we don't have Daily resolution.
            # Workaround: bars[-1] at 9:30 timestamp vs bars[-2] at 16:00 prev day?
            # Rely on todays_bars. If len(todays_bars) == 1 (it is 9:30 or 9:35), determine gap.
            
            if time(9, 30) <= current_time <= time(10, 0): # Morning only
                if self.current_pos != 0: 
                    self.manage_risk(event, symbol)
                    return
                
                # Check for Entry ONLY bars are new (e.g. 9:35 bar)
                # Need Yest Close.
                # Find last bar where date < current_date
                prev_day_bars = [b for b in bars if b.timestamp.date() < current_date]
                if not prev_day_bars: return
                
                yest_close = prev_day_bars[-1].close
                today_open = todays_bars[0].open # 9:30 Open
                
                gap_pct = (today_open - yest_close) / yest_close
                
                # Threshold
                thresh = self.params.get('gap_thresh', 0.0025) # 0.25% Gap
                
                if abs(gap_pct) > thresh:
                    # Fade the Gap (Return to Yest Close)
                    if gap_pct > 0:
                        # Gap Up -> Short
                        self.sell(symbol, 1)
                        self.entry_price = event.close
                        self.current_pos = -1
                    else:
                        # Gap Down -> Long
                        self.buy(symbol, 1)
                        self.entry_price = event.close
                        self.current_pos = 1
            else:
                # Target = Yest Close (Gap Fill)
                # Or regular risk manage
                if self.current_pos != 0:
                    # Specialized Exit: Gap Filled?
                    # Recalculate yest close (inefficient but safe)
                    prev_day_bars = [b for b in bars if b.timestamp.date() < current_date]
                    if prev_day_bars:
                        yc = prev_day_bars[-1].close
                        if self.current_pos > 0 and event.high >= yc:
                            self.exit(symbol)
                            self.current_pos = 0
                        elif self.current_pos < 0 and event.low <= yc:
                            self.exit(symbol)
                            self.current_pos = 0
                        else:
                            self.manage_risk(event, symbol)

    def manage_risk(self, event, symbol):
        sl_pct = self.params.get('sl', 0.005)
        tp_pct = self.params.get('tp', 0.01)
        
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

def run_comprehensive_research():
    start_date = "2015-01-01" 
    end_date = "2024-12-31" # Full Dataset ~10 Years
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    print(f"\nCOMPREHENSIVE RESEARCH: {start_date} to {end_date}")
    
    # 1. Lunch Shift Reversion
    print("\n--- 1. Lunch Shift Reversion (11:00-13:00) ---")
    # Vars: BB Std Dev (Mean Reversion Aggressiveness)
    bb_devs = [2.0, 2.5, 3.0]
    for dev in bb_devs:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = AlphaSeekerStrategy(data, events, mode='Lunch_Rev', params={'bb_std': dev, 'sl': 0.005, 'tp': 0.005})
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"BB Dev: {dev} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

    # 2. Vol Crunch Breakout
    print("\n--- 2. Vol Crunch (BB Squeeze) Breakout ---")
    # Vars: Squeeze Threshold (Bandwidth)
    sqz_thresholds = [0.001, 0.002, 0.003] # 0.1%, 0.2%, 0.3%
    for sqz in sqz_thresholds:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = AlphaSeekerStrategy(data, events, mode='Vol_Crunch', params={'sqz_thresh': sqz, 'sl': 0.0075, 'tp': 0.02})
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"Sqz Thresh: {sqz} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

    # 3. Morning Gap Fill (Fade)
    print("\n--- 3. Morning Gap Fill (Fade) ---")
    # Vars: Gap Min Threshold
    gap_threshs = [0.001, 0.0025, 0.005] # 0.1%, 0.25%, 0.5%
    for g in gap_threshs:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = AlphaSeekerStrategy(data, events, mode='Gap_Fill', params={'gap_thresh': g, 'sl': 0.01, 'tp': 0.01})
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"Gap Thresh: {g:.2%} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

if __name__ == "__main__":
    run_comprehensive_research()
