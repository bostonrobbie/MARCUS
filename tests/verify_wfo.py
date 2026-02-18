
import sys
import os
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from backtesting.pipeline.wfo import WalkForwardOptimizer
from backtesting.strategies.short_logic import run_short_alpha_collection

def generate_df(n=50000): # 50k bars ~ 2 years of 5m data? 78 bars/day -> 640 days
    np.random.seed(42)
    closes = np.random.normal(100, 1, n).cumsum()
    highs = closes + np.random.random(n)
    lows = closes - np.random.random(n)
    opens = closes + np.random.random(n) - 0.5
    volumes = np.random.random(n) * 1000
    
    # Dates
    start_date = pd.to_datetime('2020-01-01')
    dates = [start_date + pd.Timedelta(minutes=5*i) for i in range(n)]
    
    df = pd.DataFrame({
        'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes,
        'Datetime': dates
    })
    df.set_index('Datetime', inplace=True)
    df['Datetime'] = df.index # WFO expects this col
    return df

def test_wfo():
    print("Generating Synthetic Data...")
    df = generate_df()
    print(f"Data Range: {df.index.min()} to {df.index.max()}")
    
    # Param Grid for Strategy 2 (Parabolic)
    # Args order: strategy_id, ..., stop_loss, cost | orb_end | ema_p, ext_dist | ...
    # Base args: (2, opens..., closes..., days, times) are passed by WFO.
    # Params: sl, cost, orb_end, ema_p, ext_dist, eod_start, sma_p, rsi_entry, vwap_dist, lookback, bb_p, bb_dev, gap
    # We need to construct full tuples.
    
    # Let's optimize EMA period: 10, 50, 100
    grid = []
    
    # Fixed Params
    sl = 0.005
    cost = 1.0
    start_time = 0; eod = 0; sma = 0; rsi = 0; vwap = 0; look = 0; bb_p = 0; bb_dev = 0; gap = 0
    
    for ema_p in [10, 50, 100]:
        for Dist in [0.005, 0.01]:
            # Tuple: (sl, cost, orb_end, ema_period, ext_dist, eod_start, sma_p, rsi_entry, vwap_dist, lookback, bb_p, bb_dev, gap)
            p = (sl, cost, 0, ema_p, Dist, 0, 0, 0, 0.0, 0, 0, 0.0, 0.0)
            grid.append(p)
    
    # Construct JIT wrapper to inject Strategy ID = 2
    # The WFO calls func(*inputs, *params).
    # inputs = (opens, ..., times)
    # params = (sl, ..., gap)
    # run_short_alpha_collection signature: (strat_id, opens..., times, sl..., gap)
    # So we need a Partial or Wrapper.
    
    def jit_wrapper(opens, highs, lows, closes, volumes, days, times, *params):
        return run_short_alpha_collection(2, opens, highs, lows, closes, volumes, days, times, *params)
        
    print(f"Initializing WFO with {len(grid)} params...")
    # Train 100 days, Test 50 days (Short for test)
    optimizer = WalkForwardOptimizer(jit_wrapper, grid, train_days=100, test_days=50)
    
    print("Running WFO...")
    start_t = time.time()
    results = optimizer.run(df)
    end_t = time.time()
    
    print(f"\nWFO Completed in {end_t - start_t:.2f} seconds.")
    print("\nResults Head:")
    print(results.head())
    
    if not results.empty:
        total_pnl = results['pnl'].sum()
        print(f"\nTotal Out-of-Sample PnL: {total_pnl:.2f}")

if __name__ == "__main__":
    test_wfo()
