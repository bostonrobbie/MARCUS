
import json
import sys
import os
import pandas as pd
from datetime import datetime, date

# Ensure src is in path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("src"))

from src.backtesting.engine import BacktestEngine
from src.backtesting.data import SmartDataHandler
from src.backtesting.execution import SimulatedExecutionHandler
from src.backtesting.portfolio import Portfolio
from strategies.clean_orb_15m import CleanOrb15m

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

import queue

def run_backtest(name, start_date, end_date, **params):
    print(f"\n--- Running: {name} ---")
    config = load_config()
    search_dirs = config['data']['search_dirs']
    
    # Setup
    events = queue.Queue()
    
    data_handler = SmartDataHandler(
        symbol_list=['NQ'],
        search_dirs=search_dirs,
        start_date=pd.Timestamp(start_date),
        interval="15m" 
    )
    
    if 'NQ' in data_handler.symbol_data:
        df = data_handler.symbol_data['NQ']
        print(f"DEBUG: NQ Data Range: {df.index.min()} to {df.index.max()}")
        print(f"DEBUG: NQ Data Rows: {len(df)}")
    else:
        print("DEBUG: NQ not found in symbol_data!")
    
    data_handler.events = events
    
    portfolio = Portfolio(
        data_handler,
        events,
        initial_capital=100000.0,
        instruments={'NQ': {'multiplier': 20}} # Set multiplier for NQ
    )
    
    strategy = CleanOrb15m(
        data_handler, 
        events, 
        **params
    )
    
    if params.get('_use_chop', False):
        strategy.set_advanced_filters(use_chop=True, chop_thresh=61.8)
    
    execution = SimulatedExecutionHandler(events, data_handler)
    
    engine = BacktestEngine(
        data_handler,
        strategy,
        portfolio,
        execution
    )
    
    # results = engine.run() -> Returns None in current engine implementation
    engine.run()
    
    # Calculate Stats Manually from Portfolio
    equity_curve = pd.DataFrame(engine.portfolio.equity_curve)
    
    if not equity_curve.empty:
        equity_curve['datetime'] = pd.to_datetime(equity_curve['datetime'])
        equity_curve.set_index('datetime', inplace=True)
        equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)
        
        total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1.0
        
        # Max DD
        hwm = equity_curve['equity'].cummax()
        dd = (equity_curve['equity'] - hwm) / hwm
        max_dd = dd.min()
        
        # Sharpe (Annualized)
        mean_ret = equity_curve['returns'].mean()
        std_ret = equity_curve['returns'].std()
        sharpe = (mean_ret / std_ret) * (252**0.5) if std_ret != 0 else 0.0
        
        total_trades = len(engine.portfolio.trade_log)
        
        results = {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'total_trades': total_trades
        }
    else:
        results = {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'total_trades': 0
        }

    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"Max DD: {results['max_drawdown']:.2%}")
    print(f"Trades: {results['total_trades']}")
    
    return results

def main():
    # Robust Params from WFO
    base_params = {
        'sl_atr_mult': 2.0,
        'tp_atr_mult': 2.0,
        'ema_filter': 50,
        'adx_thresh': 20,
        'use_adx': True,
        'atr_max_mult': 3.0,
        'verbose': False
    }

    start = "2018-01-01"
    end = "2019-01-01"

    # 1. Baseline
    res_base = run_backtest("Baseline (Robust)", start, end, **base_params)

    # 3. RVOL
    params_rvol = base_params.copy()
    params_rvol['use_rvol'] = True
    params_rvol['rvol_thresh'] = 1.2
    # res_rvol = run_backtest("With RVOL Filter (>1.2)", start, end, **params_rvol)
    
    # 4. Choppiness (Advanced)
    print("\n--- Running: With Choppiness Filter ---")
    
    # We need to manually set this since I didn't add it to __init__ cleanly in the rush
    # But wait, run_backtest instantiates the strategy. 
    # I need to modify run_backtest to accept a setup_callback or just modify the class.
    # Actually, I added set_advanced_filters.
    
    # Hack for the script structure:
    # Redefine run_backtest or just patch the strategy param in the loop?
    # Better: Update run_backtest to look for 'use_chop_filter' in params and call the setter if strategy instance exists.
    
    # ACTUALLY, simpler: The Strategy accepts **params. 
    # But I defined `set_advanced_filters` instead of putting it in `__init__`.
    # Let's fix `__init__` in the strategy file to accept these kwargs properly or 
    # update the strategy to look for them.
    
    # Re-reading my previous edit to CleanOrb15m: I added `self.use_chop_filter = False` in __init__ 
    # but didn't accept it as an argument! 
    # I should have added it to __init__.
    # For now, I will subclass or just hack the init in the file on the fly? No.
    # I will rely on `set_advanced_filters` method.
    # So I need to modify `run_backtest` in this script to call `set_advanced_filters`.
    
    res_base = run_backtest("Baseline", start, end, **base_params)

    params_chop = base_params.copy()
    params_chop['_use_chop'] = True # Marker for my hack
    res_chop = run_backtest("With Chop Filter (<61.8)", start, end, **params_chop)

    print("\n\n=== COMPARISON SUMMARY ===")
    print(f"{'Strategy':<25} | {'Return':<10} | {'Sharpe':<8} | {'Max DD':<10} | {'Trades':<6}")
    print("-" * 75)
    for name, res in [("Baseline", res_base), ("Choppiness", res_chop)]:
        print(f"{name:<25} | {res['total_return']:<10.2%} | {res['sharpe']:<8.2f} | {res['max_drawdown']:<10.2%} | {res['total_trades']:<6}")

if __name__ == "__main__":
    main()
