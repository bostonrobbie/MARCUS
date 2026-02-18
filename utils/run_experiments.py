
import sys
import os
import pandas as pd
import numpy as np
from backtesting.data import SmartDataHandler
from backtesting.optimizer import VectorizedGridSearch
from examples.nqorb_15m import NqOrb15m
from backtesting.vector_engine import VectorizedNQORB

def run_experiment(name, param_grid, start_date, end_date, symbol, baseline_sharpe=None):
    print(f"\n[EXPERIMENT] {name}...")
    
    # Setup Data
    symbol_list = [symbol]
    interval = "15m"
    
    import json
    
    csv_dir = os.path.join(os.getcwd(), 'examples')
    
    # Load Config for Commission/Slippage
    config_path = os.path.join(os.getcwd(), 'StrategyPipeline', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    specs = config['data']['instrument_specs'].get(symbol, {})
    comm_rate = specs.get('commission', 2.05)
    
    # Slippage: Base standard. NQ tick=0.25, $20/pt -> $5/tick.
    # We'll pass 1 tick ($5) as fixed base, plus engine adds dynamic vol.
    point_val = specs.get('multiplier', 20)
    tick_sz = specs.get('tick_size', 0.25)
    tick_val = point_val * tick_sz # $5 for NQ
    
    base_slippage = tick_val * 1.0 # 1 tick fixed

    
    # Initialize Optimizer
    # We pass the class, but we need to ensure the grid is correct
    optimizer = VectorizedGridSearch(
        data_handler_cls=SmartDataHandler,
        data_handler_args=(symbol_list, [csv_dir], start_date, end_date, interval),
        strategy_cls=NqOrb15m, # This is just for metadata, the engine uses mappings
        param_grid=param_grid,
        initial_capital=100000.0,
        commission=comm_rate,
        slippage=base_slippage,
        n_jobs=-1
    )
    
    results = optimizer.run()
    
    if results.empty:
        print(f"  No results for {name}")
        return None
        
    best_res = results.iloc[0]
    print(f"  Best Params: {best_res.to_dict()}")
    print(f"  Total Return: {best_res['Total Return']:.2%}")
    print(f"  Final Equity: ${best_res['Final Equity']:,.2f}")
    
    return best_res

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NQ ORB Systematic Experiments")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End Date (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="NQ", help="Symbol to test (NQ, ES)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital")
    
    args = parser.parse_args()
    
    print(f"Starting Systematic Feature Testing ({args.start} to {args.end}) on {args.symbol}...")

    
    # 0. Baseline
    # Fixed params from previous optimization
    baseline_grid = {
        'orb_start': ["09:30"],
        'orb_end': ["09:45"],
        'sl_atr_mult': [2.0],
        'tp_atr_mult': [4.0],
        'ema_filter': [50],
        'atr_max_mult': [2.5]
    }
    
    print("\n--- BASELINE ---")
    base_res = run_experiment("Baseline", baseline_grid, args.start, args.end, args.symbol)
    base_ret = base_res['Total Return']
    
    experiments = [
        {
            "name": "Exp 1: HTF Trend (Daily MA)",
            "grid": {
                **baseline_grid,
                "use_htf": [True],
                "htf_ma": [50, 100, 200]
            }
        },
        {
            "name": "Exp 2: RVOL Filter",
            "grid": {
                **baseline_grid,
                "use_rvol": [True],
                "rvol_thresh": [1.0, 1.5, 2.0]
            }
        },
        {
            "name": "Exp 3: Hurst Exponent (Trend Persistence)",
            "grid": {
                **baseline_grid,
                "use_hurst": [True],
                "hurst_thresh": [0.45, 0.5, 0.55]
            }
        },
        {
            "name": "Exp 4: ADX Filter (Volatility)",
            "grid": {
                **baseline_grid,
                "use_adx": [True],
                "adx_thresh": [15, 20, 25, 30]
            }
        },
        {
            "name": "Exp 5: ATR Trailing Stop",
            "grid": {
                **baseline_grid,
                "use_trailing_stop": [True],
                "ts_atr_mult": [2.0, 3.0, 4.0, 5.0],
            }
        },
        {
            "name": "Exp 6: SYNTHESIS (HTF + RVOL + TS)",
            "grid": {
                **baseline_grid,
                "use_htf": [True], "htf_ma": [100],
                "use_rvol": [True], "rvol_thresh": [1.5],
                "use_trailing_stop": [True], "ts_atr_mult": [2.0, 2.5, 3.0]
            }
        }
    ]
    
    summary = []
    summary.append(base_res)
    
    for exp in experiments:
        res = run_experiment(exp['name'], exp['grid'], args.start, args.end, args.symbol)
        if res is not None:
            # Calculate Delta
            delta = res['Total Return'] - base_ret
            res['Delta Return'] = delta
            res['Experiment'] = exp['name']
            summary.append(res)
            
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    df_summary = pd.DataFrame(summary)
    cols = ['Experiment', 'Total Return', 'Delta Return', 'Final Equity'] 
    # Add param cols dynamically
    for c in ['htf_ma', 'rvol_thresh', 'hurst_thresh', 'adx_thresh', 'ts_atr_mult']:
        if c in df_summary.columns: cols.append(c)
        
    print(df_summary[cols].to_string())
