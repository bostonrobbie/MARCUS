import pandas as pd
import sys
import os
from backtesting.optimizer import VectorizedGridSearch
from backtesting.data import SmartDataHandler
from examples.nqorb_enhanced import NqOrbEnhanced
from backtesting.vector_engine import VectorizedNQORB

def run_tf_scan(interval, param_grid):
    print(f"\n>>> Running Optimization Scan for {interval}...")
    
    # Define Data
    # Note: We rely on SmartDataHandler to find/resample data
    data_dir = r"C:\Users\User\Desktop\Portfolio\OHLC\Intra OHLC"
    
    optimizer = VectorizedGridSearch(
        data_handler_cls=SmartDataHandler,
        data_handler_args=(['NQ'], [data_dir], "2020-01-01", "2023-12-31", interval),
        strategy_cls=NqOrbEnhanced, # Metadata
        vector_strategy_cls=VectorizedNQORB,
        param_grid=param_grid,
        initial_capital=100000.0,
        n_jobs=-1
    )
    
    results = optimizer.run()
    
    if results.empty:
        return None
        
    best = results.sort_values(by='Total Return', ascending=False).iloc[0]
    return best

if __name__ == "__main__":
    # ... (grids remain same) ...
    
    # Grid for 15m (Validation of current best)
    grid_15m = {
        'sl_atr_mult': [2.0],
        'tp_atr_mult': [4.0],
        'ema_filter': [50],
        'use_htf': [True],
        'htf_ma': [100],
        'use_rvol': [True],
        'rvol_thresh': [1.5],
        'use_trailing_stop': [True],
        'ts_atr_mult': [2.0, 2.5]
    }
    
    # Grid for 5m (Adjustment for higher frequency)
    grid_5m = {
        'sl_atr_mult': [2.0, 3.0], 
        'tp_atr_mult': [4.0, 6.0],
        'ema_filter': [50, 100, 150, 200], 
        'use_htf': [True],
        'htf_ma': [100], 
        'use_rvol': [True],
        'rvol_thresh': [1.5, 2.0],
        'use_trailing_stop': [False, True], # Explicitly test No TS too
        'ts_atr_mult': [2.0, 3.0]
    }
    
    print("="*60)
    print("MULTI-TIMEFRAME ROBUSTNESS TEST (5m vs 15m)")
    print("="*60)
    
    # Run 15m
    res_15m = run_tf_scan('15m', grid_15m)
    
    # Run 5m
    try:
        res_5m = run_tf_scan('5m', grid_5m)
    except Exception as e:
        print(f"5m Scan Failed: {e}")
        res_5m = None
        
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if res_15m is not None:
        # VectorizedGridSearch doesn't calculate Sharpe, only Total Return
        print(f"[15m BEST] Return: {res_15m['Total Return']:.2%} | Equity: {res_15m['Final Equity']:.2f}")
        print(f"    Params: {res_15m.to_dict()}")
    else:
        print("[15m] No Results")
        
    if res_5m is not None:
        print(f"[5m  BEST] Return: {res_5m['Total Return']:.2%} | Equity: {res_5m['Final Equity']:.2f}")
        print(f"    Params: {res_5m.to_dict()}")
    else:
        print("[5m ] No Results or Data Missing")
        
    # Conclusion
    if res_15m is not None and res_5m is not None:
        diff_ret = res_5m['Total Return'] - res_15m['Total Return']
        print("-" * 60)
        if diff_ret > 0.05:
            print(f"CONCLUSION: 5m Timeframe is SUPERIOR (+{diff_ret:.2%} Return)")
        elif diff_ret < -0.05:
            print(f"CONCLUSION: 15m Timeframe is SUPERIOR ({diff_ret:.2%} Return)")
        else:
            print(f"CONCLUSION: No Significant Difference (Delta < 5%)")
    print("="*60)
