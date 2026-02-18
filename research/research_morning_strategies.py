
import sys
import os
import site

# Add Src to Path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(project_root, 'StrategyPipeline'))

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

class MorningStrategy(Strategy):
    def __init__(self, bars, events, 
                 mode='VWAP_PULLBACK',
                 vwap_dist=0.5,
                 trend_filter='VWAP',
                 rr_ratio=2.0,
                 extreme_pct=1.0,
                 reversal_confirm='NO_NEW_HIGH',
                 entry_timing='IMMEDIATE',
                 verbose=False):
        super().__init__(bars, events)
        self.mode = mode
        self.vwap_dist = vwap_dist
        self.trend_filter = trend_filter
        self.rr_ratio = rr_ratio
        self.extreme_pct = extreme_pct
        self.reversal_confirm = reversal_confirm
        self.entry_timing = entry_timing
        self.verbose = verbose
        
        self.current_pos = 0
        self.entry_price = 0.0
        self.current_date = None
        
        # Daily tracking
        self.open_930 = None
        self.high_1030 = None
        self.low_1030 = None
        self.last_high_time = None
        self.last_low_time = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date_val = ts.date()
        
        # Daily reset
        if self.current_date != current_date_val:
            self.current_date = current_date_val
            self.open_930 = None
            self.high_1030 = None
            self.low_1030 = None
            self.last_high_time = None
            self.last_low_time = None
        
        # Capture 9:30 open
        if current_time >= time(9, 30) and current_time < time(9, 35) and self.open_930 is None:
            self.open_930 = event.open
            
        # Session window: 10:00 - 12:00 ET
        if current_time < time(10, 0) or current_time >= time(12, 0):
            if self.current_pos != 0:
                self.exit(symbol)
                self.current_pos = 0
            return
            
        # Get bars
        bars = self.bars.get_latest_bars(symbol, N=200)
        if len(bars) < 100: return
        
        todays_bars = [b for b in bars if b.timestamp.date() == current_date_val]
        if len(todays_bars) < 10: return
        
        closes = np.array([b.close for b in bars[-100:]])
        highs = np.array([b.high for b in bars[-100:]])
        lows = np.array([b.low for b in bars[-100:]])
        
        # ATR
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
        atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        
        # VWAP (today)
        today_closes = np.array([b.close for b in todays_bars])
        today_volumes = np.array([b.volume for b in todays_bars])
        cum_vol = np.cumsum(today_volumes)
        cum_pv = np.cumsum(today_closes * today_volumes)
        vwap = cum_pv[-1] / cum_vol[-1] if cum_vol[-1] > 0 else event.close
        
        curr_price = event.close
        
        # Track 10:30 high/low for reversal
        if current_time >= time(10, 0) and current_time < time(10, 35):
            if self.high_1030 is None or event.high > self.high_1030:
                self.high_1030 = event.high
                self.last_high_time = ts
            if self.low_1030 is None or event.low < self.low_1030:
                self.low_1030 = event.low
                self.last_low_time = ts
        
        # ==========================================
        # MODE 1: VWAP PULLBACK
        # ==========================================
        if self.mode == 'VWAP_PULLBACK':
            # Trend Filter
            trend_ok = False
            if self.trend_filter == 'VWAP':
                trend_ok = curr_price > vwap
            elif self.trend_filter == 'EMA20':
                ema20 = pd.Series(closes).ewm(span=20).mean().iloc[-1]
                trend_ok = curr_price > ema20
            elif self.trend_filter == 'MORNING_MOM':
                if self.open_930:
                    trend_ok = curr_price > self.open_930
            else:
                trend_ok = True
                
            # Entry: Price near VWAP
            dist_to_vwap = abs(curr_price - vwap)
            near_vwap = dist_to_vwap < (self.vwap_dist * atr)
            
            if self.current_pos == 0 and trend_ok and near_vwap and curr_price > vwap:
                self.buy(symbol, 1)
                self.entry_price = curr_price
                self.current_pos = 1
                
            # Exit: R:R or VWAP cross
            elif self.current_pos > 0:
                target = self.entry_price + (self.rr_ratio * atr)
                stop = self.entry_price - atr
                
                if curr_price >= target or curr_price <= stop or curr_price < vwap:
                    self.exit(symbol)
                    self.current_pos = 0
                    
        # ==========================================
        # MODE 2: FIRST HOUR REVERSAL
        # ==========================================
        elif self.mode == 'FIRST_HOUR_REVERSAL':
            if not self.open_930 or not self.high_1030 or not self.low_1030:
                return
                
            # Check if we're past 10:30
            if current_time < time(10, 30):
                return
                
            # Define extreme move
            move_up = ((self.high_1030 - self.open_930) / self.open_930) * 100
            move_down = ((self.open_930 - self.low_1030) / self.open_930) * 100
            
            is_extreme_up = move_up > self.extreme_pct
            is_extreme_down = move_down > self.extreme_pct
            
            if not (is_extreme_up or is_extreme_down):
                return
                
            # Reversal confirmation
            confirmed = False
            if self.reversal_confirm == 'NO_NEW_HIGH':
                # No new high in last 5 bars (10 mins on 5m)
                if is_extreme_up and self.last_high_time:
                    bars_since = (ts - self.last_high_time).total_seconds() / 60 / 5
                    confirmed = bars_since >= 5
                elif is_extreme_down and self.last_low_time:
                    bars_since = (ts - self.last_low_time).total_seconds() / 60 / 5
                    confirmed = bars_since >= 5
            else:
                confirmed = True
                
            if not confirmed:
                return
                
            # Entry timing
            can_enter = False
            if self.entry_timing == 'IMMEDIATE':
                can_enter = current_time >= time(10, 30)
            elif self.entry_timing == 'VWAP_CROSS':
                if is_extreme_up:
                    can_enter = curr_price < vwap
                elif is_extreme_down:
                    can_enter = curr_price > vwap
                    
            if self.current_pos == 0 and can_enter:
                if is_extreme_up:
                    # Fade the high
                    self.sell(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = -1
                elif is_extreme_down:
                    # Fade the low
                    self.buy(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = 1
                    
            # Exit: Return to open or R:R
            elif self.current_pos != 0:
                if self.current_pos > 0:
                    target = self.open_930
                    stop = self.entry_price - (self.rr_ratio * atr)
                    if curr_price >= target or curr_price <= stop:
                        self.exit(symbol)
                        self.current_pos = 0
                elif self.current_pos < 0:
                    target = self.open_930
                    stop = self.entry_price + (self.rr_ratio * atr)
                    if curr_price <= target or curr_price >= stop:
                        self.exit(symbol)
                        self.current_pos = 0

def run_research():
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    print(f"\nMORNING STRATEGIES RESEARCH: {start_date} to {end_date}")
    
    # ==========================================
    # 1. VWAP PULLBACK VARIATIONS
    # ==========================================
    print("\n" + "="*60)
    print("STRATEGY 1: VWAP PULLBACK (10:00-12:00)")
    print("="*60)
    
    for vwap_dist in [0.25, 0.5, 0.75]:
        for trend_filter in ['VWAP', 'EMA20', 'MORNING_MOM']:
            for rr in [1.0, 2.0, 3.0]:
                events = queue.Queue()
                data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
                port = Portfolio(data, events, initial_capital=100000.0)
                strat = MorningStrategy(data, events, mode='VWAP_PULLBACK', vwap_dist=vwap_dist, trend_filter=trend_filter, rr_ratio=rr)
                exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
                engine = BacktestEngine(data, strat, port, exec_h)
                engine.run()
                
                stats = TearSheet(port).analyze()
                print(f"VWAP Dist: {vwap_dist} ATR | Filter: {trend_filter:12} | R:R: {rr} | Ret: {stats.get('Total Return',0):6.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):6.2%}")

    # ==========================================
    # 2. FIRST HOUR REVERSAL VARIATIONS
    # ==========================================
    print("\n" + "="*60)
    print("STRATEGY 2: FIRST HOUR REVERSAL (Fade 9:30-10:30 Extremes)")
    print("="*60)
    
    for extreme_pct in [0.5, 1.0, 1.5]:
        for confirm in ['NO_NEW_HIGH', 'NONE']:
            for entry in ['IMMEDIATE', 'VWAP_CROSS']:
                events = queue.Queue()
                data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
                port = Portfolio(data, events, initial_capital=100000.0)
                strat = MorningStrategy(data, events, mode='FIRST_HOUR_REVERSAL', extreme_pct=extreme_pct, reversal_confirm=confirm, entry_timing=entry)
                exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
                engine = BacktestEngine(data, strat, port, exec_h)
                engine.run()
                
                stats = TearSheet(port).analyze()
                print(f"Extreme: {extreme_pct}% | Confirm: {confirm:12} | Entry: {entry:12} | Ret: {stats.get('Total Return',0):6.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):6.2%}")

if __name__ == "__main__":
    run_research()
