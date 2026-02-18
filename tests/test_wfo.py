import sys
import os
import pandas as pd
sys.path.append(os.getcwd())

from backtesting.optimizer import WalkForwardOptimizer
from backtesting.registry import StrategyRegistry
from examples.nqorb_enhanced import NqOrbEnhanced
from examples.nqorb_15m import NqOrb15m

def test_wfo_engine():
    print("Testing Walk-Forward Optimizer (Tooling Check)...")
    
    # 1. Setup
    symbol_list = ['NQ']
    search_dirs = [os.path.join(os.getcwd(), 'examples')]
    
    # Simple Grid
    param_grid = {
        'sl_atr_mult': [1.0, 2.0],
        'tp_atr_mult': [2.0, 4.0]
    }
    
    # 2. Instantiate WFO
    # Use small windows to fit in available data (2020-2025)
    # Train 120 days, Test 60 days, Slide 30 days
    wfo = WalkForwardOptimizer(
        strategy_cls=NqOrb15m, # Faster strategy
        symbol_list=symbol_list,
        search_dirs=search_dirs,
        param_grid=param_grid,
        train_days=120,
        test_days=60,
        step_days=30, # Overlapping windows
        initial_capital=100000.0,
        interval='15m' # Use 15m data if available or 1d? 
        # NqOrb15m needs intraday. Assuming NQ data loads.
    )
    
    # 3. Run
    try:
        results_df, equity = wfo.run()
        
        print("\nTest Results:")
        print(results_df.head())
        
        if not results_df.empty:
            print(f"\n[PASS] WFO Generated {len(results_df)} windows.")
            print(f"       Average Test Return: {results_df['test_return'].mean():.2%}")
        else:
            print("\n[FAIL] WFO produced no results.")
            
    except Exception as e:
        print(f"\n[ERROR] WFO Crash: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wfo_engine()
