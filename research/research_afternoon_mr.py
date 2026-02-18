
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

class AfternoonMRStrategy(Strategy):
    def __init__(self, bars, events, 
                 mode='BB_MTF',
                 bb_period=20, bb_std=2.0,
                 vwap_mult=1.5,
                 rsi_period=14, rsi_low=30, rsi_high=70,
                 vix_threshold=20,
                 verbose=False):
        super().__init__(bars, events)
        self.mode = mode
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vwap_mult = vwap_mult
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.vix_threshold = vix_threshold
        self.verbose = verbose
        
        self.current_pos = 0
        self.entry_price = 0.0
        self.current_date = None
        
        # Range tracking
        self.range_high = None
        self.range_low = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        current_date_val = ts.date()
        
        # Daily reset
        if self.current_date != current_date_val:
            self.current_date = current_date_val
            self.range_high = None
            self.range_low = None
        
        # Session window: 13:00 - 16:00 ET
        if current_time < time(13, 0) or current_time >= time(16, 0):
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
        volumes = np.array([b.volume for b in bars[-100:]])
        
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
        
        # ==========================================
        # MODE 1: BB MEAN REVERSION (MULTI-TF)
        # ==========================================
        if self.mode == 'BB_MTF':
            # 5m BB
            ma = pd.Series(closes).rolling(self.bb_period).mean().iloc[-1]
            std = pd.Series(closes).rolling(self.bb_period).std().iloc[-1]
            upper = ma + (self.bb_std * std)
            lower = ma - (self.bb_std * std)
            
            # Simple ADX proxy: check if recent range is tight (choppy)
            recent_range = highs[-20:].max() - lows[-20:].min()
            avg_range = (highs[-50:-20].max() - lows[-50:-20].min())
            is_choppy = recent_range < avg_range * 1.2  # Range compression
            
            if self.current_pos == 0 and is_choppy:
                if curr_price > upper:
                    self.sell(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = -1
                elif curr_price < lower:
                    self.buy(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = 1
            else:
                # Exit at MA
                if self.current_pos > 0 and curr_price >= ma:
                    self.exit(symbol)
                    self.current_pos = 0
                elif self.current_pos < 0 and curr_price <= ma:
                    self.exit(symbol)
                    self.current_pos = 0
                    
        # ==========================================
        # MODE 2: VWAP REVERSION + VOLUME
        # ==========================================
        elif self.mode == 'VWAP_VOL':
            dist = abs(curr_price - vwap)
            threshold = self.vwap_mult * atr
            
            # Volume declining?
            avg_vol = volumes[-20:].mean()
            curr_vol = event.volume
            vol_declining = curr_vol < avg_vol * 0.8
            
            if self.current_pos == 0 and vol_declining:
                if curr_price > vwap + threshold:
                    self.sell(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = -1
                elif curr_price < vwap - threshold:
                    self.buy(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = 1
            else:
                # Exit at VWAP cross
                if self.current_pos > 0 and curr_price >= vwap:
                    self.exit(symbol)
                    self.current_pos = 0
                elif self.current_pos < 0 and curr_price <= vwap:
                    self.exit(symbol)
                    self.current_pos = 0
                    
        # ==========================================
        # MODE 3: RSI EXTREMES
        # ==========================================
        elif self.mode == 'RSI':
            # Calculate RSI
            deltas = pd.Series(closes).diff()
            gain = deltas.where(deltas > 0, 0).rolling(self.rsi_period).mean()
            loss = -deltas.where(deltas < 0, 0).rolling(self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            curr_rsi = rsi.iloc[-1]
            
            if np.isnan(curr_rsi): return
            
            if self.current_pos == 0:
                if curr_rsi < self.rsi_low:
                    self.buy(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = 1
                elif curr_rsi > self.rsi_high:
                    self.sell(symbol, 1)
                    self.entry_price = curr_price
                    self.current_pos = -1
            else:
                # Exit at RSI 50
                if curr_rsi > 50 and self.current_pos > 0:
                    self.exit(symbol)
                    self.current_pos = 0
                elif curr_rsi < 50 and self.current_pos < 0:
                    self.exit(symbol)
                    self.current_pos = 0
                    
        # ==========================================
        # MODE 4: AFTERNOON RANGE FADE
        # ==========================================
        elif self.mode == 'RANGE_FADE':
            # Build 13:00-14:00 range
            if time(13, 0) <= current_time < time(14, 0):
                if self.range_high is None or event.high > self.range_high:
                    self.range_high = event.high
                if self.range_low is None or event.low < self.range_low:
                    self.range_low = event.low
                return
            
            # Trade after 14:00
            if self.range_high and self.range_low and current_time >= time(14, 0):
                # False breakout: wick outside, close inside
                if self.current_pos == 0:
                    if event.high > self.range_high and curr_price < self.range_high:
                        self.sell(symbol, 1)
                        self.entry_price = curr_price
                        self.current_pos = -1
                    elif event.low < self.range_low and curr_price > self.range_low:
                        self.buy(symbol, 1)
                        self.entry_price = curr_price
                        self.current_pos = 1
                else:
                    # Exit at range midpoint
                    mid = (self.range_high + self.range_low) / 2
                    if abs(curr_price - mid) < atr * 0.2:
                        self.exit(symbol)
                        self.current_pos = 0

        # Risk management for all modes (simple fixed %)
        if self.current_pos != 0:
            self.manage_risk(event, symbol)

    def manage_risk(self, event, symbol):
        sl_pct = 0.005
        tp_pct = 0.01
        
        if self.current_pos > 0:
            sl = self.entry_price * (1 - sl_pct)
            tp = self.entry_price * (1 + tp_pct)
            if event.low < sl or event.high > tp:
                self.exit(symbol)
                self.current_pos = 0
        elif self.current_pos < 0:
            sl = self.entry_price * (1 + sl_pct)
            tp = self.entry_price * (1 - tp_pct)
            if event.high > sl or event.low < tp:
                self.exit(symbol)
                self.current_pos = 0

def run_research():
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd())]
    symbol_list = ['NQ']
    
    print(f"\nAFTERNOON MR RESEARCH: {start_date} to {end_date}")
    
    # 1. BB Multi-TF
    print("\n--- 1. BB Mean Reversion (Multi-TF) ---")
    for bb_std in [2.0, 2.5, 3.0]:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = AfternoonMRStrategy(data, events, mode='BB_MTF', bb_std=bb_std)
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"BB StdDev: {bb_std} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

    # 2. VWAP + Volume
    print("\n--- 2. VWAP Reversion + Volume ---")
    for vwap_mult in [1.0, 1.5, 2.0]:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = AfternoonMRStrategy(data, events, mode='VWAP_VOL', vwap_mult=vwap_mult)
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"VWAP Mult: {vwap_mult} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

    # 3. RSI Extremes
    print("\n--- 3. RSI Extremes ---")
    for rsi_low in [25, 30, 35]:
        events = queue.Queue()
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
        port = Portfolio(data, events, initial_capital=100000.0)
        strat = AfternoonMRStrategy(data, events, mode='RSI', rsi_low=rsi_low, rsi_high=100-rsi_low)
        exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
        engine = BacktestEngine(data, strat, port, exec_h)
        engine.run()
        
        stats = TearSheet(port).analyze()
        print(f"RSI Low: {rsi_low} | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

    # 4. Range Fade
    print("\n--- 4. Afternoon Range Fade ---")
    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port = Portfolio(data, events, initial_capital=100000.0)
    strat = AfternoonMRStrategy(data, events, mode='RANGE_FADE')
    exec_h = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
    engine = BacktestEngine(data, strat, port, exec_h)
    engine.run()
    
    stats = TearSheet(port).analyze()
    print(f"Range Fade | Ret: {stats.get('Total Return',0):.2%} | PF: {stats.get('Profit Factor',0):.2f} | DD: {stats.get('Max Drawdown',0):.2%}")

if __name__ == "__main__":
    run_research()
