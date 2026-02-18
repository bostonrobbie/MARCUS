
import sys
import os
from datetime import time, timedelta
import pandas as pd
import numpy as np
import datetime

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
import queue

class OvernightRobustCollector(Strategy):
    def __init__(self, bars, events, verbose=False):
        super().__init__(bars, events)
        self.verbose = verbose
        self.trade_log = []
        self.current_trade = None
        
        # State
        self.sma50 = None
        self.sma100 = None
        self.sma200 = None
        self.atr14 = None
        
        # Iterative State
        self.prices = [] # Sliding window for crude SMA calc if needed, or use iterative
        
    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        close = event.close
        
        # --- Update Indicators (Iterative for Speed) ---
        # Approximating Daily SMAs using 5m bars is heavy.
        # We will use Exponential Moving Averages (EMA) as they are iterative.
        # Daily 50 SMA ~ EMA 50 on Daily ~ EMA (50 * 78) on 5m? 
        # No, let's use fixed long spans that represent "Trends".
        # Fast Trend: 1 Week (EMA 2000 5m bars)
        # Medium Trend: 1 Month (EMA 8000 5m bars)
        # Slow Trend: 3 Months (EMA 24000 5m bars)
        
        if self.sma50 is None: self.sma50 = close
        else: self.sma50 = (2/(2000+1))*close + (1-(2/(2000+1)))*self.sma50
        
        if self.sma100 is None: self.sma100 = close
        else: self.sma100 = (2/(8000+1))*close + (1-(2/(8000+1)))*self.sma100
        
        if self.sma200 is None: self.sma200 = close
        else: self.sma200 = (2/(24000+1))*close + (1-(2/(24000+1)))*self.sma200
        
        # --- Trade Logic ---
        
        # Check Active Trade (In Position)
        if self.current_trade is not None:
            # Update MAE/MFE
            # Max Adverse = Min Low
            if event.low < self.current_trade['min_low']:
                self.current_trade['min_low'] = event.low
            # Max Favorable = Max High
            if event.high > self.current_trade['max_high']:
                self.current_trade['max_high'] = event.high
                
            # Exit Condition (09:30 RTH Open)
            if current_time == time(9, 30):
                self.current_trade['exit_price'] = event.open
                self.current_trade['exit_time'] = ts
                
                # Close Trade
                self.trade_log.append(self.current_trade)
                self.current_trade = None
                self.exit(symbol) # Only for backtester accounting
                
        # Entry Condition (18:00 ETH Open)
        # We check window 18:00-18:15
        if self.current_trade is None:
            if current_time.hour == 18 and current_time.minute < 15:
                # Open Trade
                self.buy(symbol, 1) # Backtester
                self.current_trade = {
                    'entry_time': ts,
                    'entry_price': event.open, # Use Open of bar
                    'day_of_week': ts.dayofweek,
                    'current_close': close,
                    'sma50': self.sma50,
                    'sma100': self.sma100,
                    'sma200': self.sma200,
                    'min_low': event.low,  # Init MAE
                    'max_high': event.high, # Init MFE
                    'exit_price': 0.0
                }

def run_robust_optimization(start_date, end_date):
    print(f"Running Robust Data Collection: {start_date} to {end_date}")
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd(), 'examples'), os.path.join(os.getcwd())]
    symbol_list = ['NQ']

    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port = Portfolio(data, events, initial_capital=100000.0)
    strat = OvernightRobustCollector(data, events, verbose=False)
    exec_handler = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(0.0)) # Raw pnl first
    engine = BacktestEngine(data, strat, port, exec_handler)
    engine.run()
    
    # --- Offline Grid Search ---
    df = pd.DataFrame(strat.trade_log)
    print(f"\nCollected {len(df)} potential trades.")
    
    if len(df) == 0: return

    # Parameters to Test
    trend_opts = ['None', 'Fast (EMA 2k)', 'Med (EMA 8k)', 'Slow (EMA 24k)']
    sl_opts = [None, 0.01, 0.015, 0.02, 0.03] # 1%, 1.5%, 2%, 3%
    tp_opts = [None, 0.01, 0.015, 0.02]
    day_opts = ['All', 'Skip Thu']
    
    best_res = {'return': -999}
    
    print(f"\nScanning Combinations...")
    
    # Pre-calc columns to speed up loop
    df['ret_raw'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
    df['trend_fast'] = df['current_close'] > df['sma50']
    df['trend_med'] = df['current_close'] > df['sma100']
    df['trend_slow'] = df['current_close'] > df['sma200']
    
    for d_opt in day_opts:
        # Day Filter
        d_mask = df.index >= 0 # All true initially
        if d_opt == 'Skip Thu':
            d_mask = df['day_of_week'] != 3 # Thu is 3
        
        subset_d = df[d_mask]
        
        for t_opt in trend_opts:
            # Trend Filter
            t_mask = subset_d.index >= 0
            if t_opt == 'Fast (EMA 2k)': t_mask = subset_d['trend_fast']
            elif t_opt == 'Med (EMA 8k)': t_mask = subset_d['trend_med']
            elif t_opt == 'Slow (EMA 24k)': t_mask = subset_d['trend_slow']
            
            subset_dt = subset_d[t_mask].copy()
            if len(subset_dt) < 50: continue
            
            for sl in sl_opts:
                for tp in tp_opts:
                    # Calculate PnL with SL/TP
                    # Vectorized logic for speed
                    
                    # 1. Did SL hit? Min Low < Entry * (1-SL)
                    sl_hit_mask = pd.Series([False]*len(subset_dt), index=subset_dt.index)
                    if sl is not None:
                        sl_price = subset_dt['entry_price'] * (1 - sl)
                        sl_hit_mask = subset_dt['min_low'] < sl_price
                        
                    # 2. Did TP hit? Max High > Entry * (1+TP)
                    tp_hit_mask = pd.Series([False]*len(subset_dt), index=subset_dt.index)
                    if tp is not None:
                        tp_price = subset_dt['entry_price'] * (1 + tp)
                        tp_hit_mask = subset_dt['max_high'] > tp_price
                        
                    # 3. Resolve (Assume SL hits first if both? Or verify timing? Impossible with OHLC summary.
                    # Strict: If SL hit, we take loss. (Conservative)
                    # We define Result PnL
                    
                    # Base PnL
                    final_pnl = subset_dt['ret_raw'].copy()
                    
                    # Apply SL (Limit loss to -SL)
                    if sl is not None:
                        final_pnl[sl_hit_mask] = -sl
                    
                    # Apply TP (Limit gain to +TP, ONLY IF SL didn't hit? 
                    # If both hit in same 5m bar, unknown. Usually SL hit is worse.
                    # If TP hit and SL NOT hit:
                    if tp is not None:
                        # If SL is None, sl_hit_mask is all False so this works.
                        # If SL is set, we respect it first.
                        # Logic: If SL hit, we lost. If SL NOT hit AND TP hit, we win fixed amount.
                        final_pnl[tp_hit_mask & ~sl_hit_mask] = tp
                    
                    # Metrics
                    total_ret = final_pnl.sum()
                    count = len(final_pnl)
                    avg = final_pnl.mean()
                    
                    # Score (Simple Return)
                    if total_ret > best_res['return']:
                        best_res = {
                            'return': total_ret,
                            'days': d_opt,
                            'trend': t_opt,
                            'sl': sl,
                            'tp': tp,
                            'count': count,
                            'avg': avg
                        }

    print("\n--- ROBUST OPTIMIZATION RESULTS ---")
    print(f"Best Config:")
    print(f"  Day Filter: {best_res['days']}")
    print(f"  Trend Filt: {best_res['trend']}")
    print(f"  Stop Loss : {best_res['sl']}")
    print(f"  Take Prof : {best_res['tp']}")
    print(f"  Total Ret : {best_res['return']:.2%} (Sum of %)")
    print(f"  Trades    : {best_res['count']}")
    print(f"  Avg Trade : {best_res['avg']:.4%}")

if __name__ == "__main__":
    run_robust_optimization("2015-01-01", "2024-12-31")
