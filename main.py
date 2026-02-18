
import argparse
import sys
import os
import json
from datetime import datetime

# Path Setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from backtesting.pipeline import ResearchPipeline
from backtesting.audit import PreFlightCheck
from backtesting.genetic import EvolutionaryOptimizer
from backtesting.data import SmartDataHandler
import backtesting.strategies # Ensure strategies are loaded (if using registry)
# Or dynamic import helper

def load_config(path='config.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def main():
    # 1. Pre-Flight Verification
    auditor = PreFlightCheck()
    if not auditor.run():
        print("❌ Pipeline Aborted due to Pre-Flight Failures.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Unified Strategy Pipeline CLI")
    parser.add_argument('strategy', help="Name of the strategy class (e.g. NqOrb15m)")
    parser.add_argument('--symbol', default='NQ', help="Symbol to test (default: NQ)")
    parser.add_argument('--wfo', action='store_true', help="Enable Walk-Forward Optimization")
    parser.add_argument('--robustness', action='store_true', help="Run Monte Carlo & Regime checks")
    parser.add_argument('--tv-mode', action='store_true', help="Use TradingView Broker Emulator (Intrabar)")
    parser.add_argument('--campaign', action='store_true', help="Run Evolutionary Strategy Discovery Campaign")
    parser.add_argument('--interval', default='5m', help="Data interval (default: 5m)")
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Override config with CLI args
    if args.tv_mode:
        config['execution_mode'] = 'TV_BROKER_EMULATOR'
    
    if args.campaign:
        print(f"🚀 Starting WORKHORSE CAMPAIGN on {args.symbol}")
        print("   Mode: Evolutionary Strategy Discovery")
        
        # Setup GA
        # Using Dummy Strategy Class for now as VectorEngine handles logic based on params
        # But we need a valid class to pass. We'll use NqOrb15m as the 'Container'
        from strategies.nqorb_15m import NqOrb15m 
        
        ga = EvolutionaryOptimizer(
            data_handler_cls=SmartDataHandler,
            data_handler_args=([args.symbol], config.get('data', {}).get('search_dirs', []) + [os.path.join(os.path.dirname(__file__), 'data')], datetime(2025, 11, 1), datetime(2026, 1, 1), '15m'),
            population_size=20, # Small for demo
            generations=5,
            initial_capital=100000.0,
            n_jobs=4
        )
        
        results = ga.run()
        if not results.empty:
            best = results.iloc[0].to_dict()
            print("\n🏆 CAMPAIGN WINNER:")
            print(f"   Return: {best.get('Total Return', 0):.2%}")
            print(f"   Params: {best}")
            
            # Optionally: Run Validation on Winner
            # pipeline = ResearchPipeline(...)
            # pipeline.run_event_certification(best)
        else:
            print("   No viable strategies found.")
            
        sys.exit(0)

    print(f"🚀 Starting Pipeline for {args.strategy} on {args.symbol}")
    print(f"   Config: {config}")

    # Dynamic Import of Strategy
    import importlib.util
    import inspect
    
    # Try importing from 'strategies' package
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategies'))
        module_name = args.strategy.lower() # assumption: file is lowercase version of class
        # Heuristic: 'NqOrb15m' -> file 'nqorb_15m.py' or 'nqorb15m.py'? 
        # Let's try direct import first
        try:
             # Try direct module name if user passed filename-like stem
             module = importlib.import_module(module_name)
        except ImportError:
             # Try underscore conversion (Simple heuristic)
             # NqOrb15m -> nqorb_15m ?
             # Or just search the directory
             found_mod = None
             for f in os.listdir('strategies'):
                 if f.endswith('.py'):
                     if f.replace('_', '').lower() == module_name.replace('_', '').lower() + ".py":
                         found_mod = f[:-3]
                         break
             if found_mod:
                 module = importlib.import_module(found_mod)
             else:
                 raise ImportError(f"Could not find module for {args.strategy}")
                 
        # Look for class in module
        strategy_cls = getattr(module, args.strategy, None)
        if not strategy_cls:
             # Case-insensitive search
             for name, obj in inspect.getmembers(module):
                 if name.lower() == args.strategy.lower() and inspect.isclass(obj):
                     strategy_cls = obj
                     break
        
        if not strategy_cls:
            raise ValueError(f"Class {args.strategy} not found in module.")
            
    except Exception as e:
        print(f"Error loading strategy: {e}")
        # Fallback for testing execution with a Dummy if needed, or exit
        sys.exit(1)
    
    pipeline = ResearchPipeline(
        strategy_name=args.strategy,
        strategy_cls=strategy_cls,
        symbol_list=[args.symbol],
        param_grid={ 
            'sl_atr_mult': [1.0, 2.0, 3.0],
            'tp_atr_mult': [2.0, 3.0, 4.0],
            'use_adx': [True],
            'adx_thresh': [20],
            'ema_filter': [50]
        },
        data_range_start=datetime(2011, 1, 1),
        data_range_end=datetime(2025, 6, 1),
        interval=args.interval,
        search_dirs=config.get('data', {}).get('search_dirs', []) + [os.path.join(os.path.dirname(__file__), 'data')],
        config=config
    )
    
    pipeline.execute(wfo=args.wfo)

if __name__ == "__main__":
    main()
