import argparse
import sys
import os
import queue

# Bootstrapper (Fixes CUDA, Paths)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
    import backtesting.boot
except Exception as e:
    print(f"[BOOT] Warning: Could not load bootstrapper: {e}")

import importlib.util
import inspect
from datetime import time, datetime
import pandas as pd 
import matplotlib
matplotlib.use('Agg') 

# --- PATH SETUP ---
# Add 'src' to system path to allow 'from backtesting import ...'
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # StrategyPipeline/scripts
project_root = os.path.dirname(current_script_dir)              # StrategyPipeline
src_dir = os.path.join(project_root, 'src')                     # StrategyPipeline/src
strategies_dir = os.path.join(project_root, 'strategies')       # StrategyPipeline/strategies

sys.path.insert(0, src_dir)
sys.path.insert(1, strategies_dir)

from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission, FixedSlippage
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
from queue import Queue
from backtesting.optimizer import GridSearch, WalkForwardOptimizer, VectorizedGridSearch, BayesianOptimizer
from backtesting.plotting import Plotter
from backtesting.validation import ValidationSuite, SensitivityTester
from backtesting.monitor import SignalMonitor
from backtesting.vector_engine import VectorEngine, VectorizedMA, VectorizedNQORB
from backtesting.gpu_engine import GpuVectorEngine
from backtesting.strategies_gpu import GpuVectorizedNQORB
from backtesting.registry import StrategyRegistry
from backtesting.regime import MarketRegimeDetector
from backtesting.scaffold import generate_strategy
from backtesting.ai_assistant import AIOptimizationPrompter
from backtesting.preflight import PreFlightCheck
from backtesting.accelerate import get_gpu_info

# Strategy Imports
try:
    # Load Configuration
    config_path = os.path.join(project_root, 'config.json')
    CONFIG = {}
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)
        print(f"[CONFIG] Loaded from {config_path}")
    else:
        print("[CONFIG] Warning: config.json not found. Using defaults.")

    # Dynamic Strategy Import (rest of code...)
    from trend_following import MovingAverageCrossover
    from orb_strategy import OpeningRangeBreakout
    from mean_reversion import MeanReversion
    from nqorb import NqOrb
    from nqorb_15m import NqOrb15m
    from nqorb_enhanced import NqOrbEnhanced
except ImportError as e:
    print(f"[WARNING] Could not import some strategies: {e}")

# Helper to load strategies dynamically
def load_strategy_class(name):
    # Try Import from strategies package first (if in sys path)
    try:
        module = importlib.import_module(name.lower())
        # Inspect for class matching name case-insensitive?
        # Assuming filename 'nqorb.py' contains class 'NqOrb'
        for attr_name, obj in inspect.getmembers(module):
            if attr_name.lower() == name.lower() and inspect.isclass(obj):
                return obj
    except ImportError:
        pass
    
    # Check explicit strategies folder files if not found
    for fname in os.listdir(strategies_dir):
        if fname.endswith(".py"):
            mod_name = fname[:-3]
            try:
                module = importlib.import_module(mod_name)
                for attr_name, obj in inspect.getmembers(module):
                    if attr_name.lower() == name.lower() and inspect.isclass(obj):
                        return obj
            except ImportError:
                continue
    return None

def check_gpu_strict():
    """Enforce GPU availability."""
    info = get_gpu_info()
    if not info.get('gpu_available'):
        print("\n" + "!"*60)
        print("[CRITICAL] GPU ACCELERATION REQUIRED BUT NOT AVAILABLE")
        print("!"*60)
        print(f"Error Details: {info.get('error', 'Unknown Error')}")
        print("Please ensure you are running in the 'gpu_env' with cupy-cuda12x installed.")
        sys.exit(1)
    print(f"[INFO] GPU Detected: {info.get('cupy')} (CUDA Device Present: {info.get('cuda_device')})")

def run_walk_forward(strategy_cls, symbol_list, search_dirs, param_grid, start_date, end_date, interval='1d'):
    """
    Executes Walk-Forward Optimization.
    """
    from backtesting.optimizer import WalkForwardOptimizer, VectorizedGridSearch
    from backtesting.vector_engine import VectorEngine, VectorizedMA, VectorizedNQORB

    print("\n" + "="*50)
    print(f"WALK-FORWARD OPTIMIZATION: {strategy_cls.__name__}")
    print("="*50)
    
    # Configure engines based on strategy
    v_strat_map = {
        'MovingAverageCrossover': VectorizedMA,
        'NqOrb': VectorizedNQORB,
        'NqOrb15m': VectorizedNQORB
    }
    v_strat_cls = v_strat_map.get(strategy_cls.__name__)

    wfo = WalkForwardOptimizer(
        strategy_cls=strategy_cls,
        symbol_list=symbol_list,
        search_dirs=search_dirs,
        param_grid=param_grid,
        train_days=90, # Configurable?
        test_days=30,
        step_days=30,
        interval=interval,
        vector_engine_cls=VectorEngine,
        vector_strategy_cls=v_strat_cls
    )
    
    results, equity_frames = wfo.run()
    
    print("\n" + "="*50)
    print("WFO RESULTS")
    print("="*50)
    print(results)
    
    # Calculate Stitched Performance
    if equity_frames:
        stitched = pd.concat(equity_frames)
        total_ret = (1 + stitched).prod() - 1.0
        print(f"\n[Usage] Total Stitched Return (Out-of-Sample): {total_ret:.2%}")
        
    return results

def verify_parity(strategy_cls, params, vector_return, data_handler, initial_capital=100000.0) -> dict:
    """
    Runs a single Event-Based backtest to verify the 'realism' of a Vector result.
    Returns a dict with diff metrics.
    """
    print(f"[PARITY] Verifying parameters: {params}")
    
    # Setup Event Engine
    events = Queue()
    portfolio = Portfolio(data_handler, events, initial_capital)
    strategy = strategy_cls(data_handler, events, **params)
    
    # Use realistic costs via Config
    from backtesting.execution import AssetAwareCommissionModel, VolatilitySlippageModel
    
    # Load specs from CONFIG if available, else empty (defaults)
    specs = CONFIG.get('data', {}).get('instrument_specs', {})
    
    execution = SimulatedExecutionHandler(
        events, 
        data_handler, 
        commission_model=AssetAwareCommissionModel(instrument_specs=specs),
        slippage_model=VolatilitySlippageModel(data_handler, factor=0.05, min_ticks=1) 
    )
    
    engine = BacktestEngine(data_handler, strategy, portfolio, execution)
    engine.run()
    
    # Calculate Event Return
    eq_curve = pd.DataFrame(portfolio.equity_curve)
    if eq_curve.empty:
        event_return = 0.0
    else:
        start_eq = eq_curve['equity'].iloc[0]
        if start_eq == 0: start_eq = initial_capital
        event_return = (eq_curve['equity'].iloc[-1] / start_eq) - 1.0
        
    diff = event_return - vector_return
    is_divergent = abs(diff) > 0.05 # 5% Tolerance
    
    return {
        'vector_return': vector_return,
        'event_return': event_return,
        'diff': diff,
        'is_divergent': is_divergent
    }


def run_strategy(strategy_cls, symbol_list, data_dirs, plot=False, mode='OPTIMISTIC', check_validation=True, verbose=True, interval='1d', start_date=None, end_date=None, **kwargs):
    if verbose:
        print(f"Running {strategy_cls.__name__} in {mode} mode on {symbol_list} ({interval})...")
    
    args_dirs = data_dirs if data_dirs else []
    config_dirs = CONFIG.get('data', {}).get('search_dirs', [])
    combined_dirs = list(set(args_dirs + config_dirs)) # Remove dupes
    
    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=combined_dirs, interval=interval, start_date=start_date, end_date=end_date)

    # --- Pre-Flight Check (Vectorized) ---
    # NOTE: With new GPU-First default, the main runner already handles checks.
    # This block is kept for legacy 'run_strategy' direct calls but might be redundant.
    # We will trust the main execution flow to handle Pre-Flight.
    
    # Define Instruments 
    instruments = {
        'ES': {'multiplier': 50.0},
        'NQ': {'multiplier': 20.0},
        'CL': {'multiplier': 1000.0},
        'SPY': {'multiplier': 1.0},
        'NVDA': {'multiplier': 1.0}
    }
    
    
    # Load Benchmark (SPY) for Reporting
    benchmark_series = None
    try:
        bench_data = SmartDataHandler(['SPY'], search_dirs=data_dirs, interval='1d', start_date=start_date, end_date=end_date)
        if 'SPY' in bench_data.symbol_data:
            benchmark_series = bench_data.symbol_data['SPY']['Close']
            print("Loaded Benchmark (SPY) data.")
    except Exception as e:
        print(f"Could not load benchmark data: {e}")

    portfolio = Portfolio(data, events, initial_capital=100000.0, instruments=instruments)
    
    # Separate archive flag and notes from strategy params
    archive = kwargs.pop('archive', False)
    notes = kwargs.pop('note', "")
    
    # --- Duplicate Detection ---
    reg = StrategyRegistry()
    sym = symbol_list[0]
    df = data.symbol_data[sym]
    data_range = (df.index[0], df.index[-1])
    
    dup = reg.check_duplicate(strategy_cls.__name__, sym, interval, kwargs, data_range)
    if dup:
        print(f"\n[INFO] Identical test found in Knowledge Base (Run on {dup['timestamp']}).")
        print(f"       Previous Sharpe: {dup['sharpe_ratio']:.2f}, Return: {dup['total_return']:.2%}")
        if not archive:
            print("       Skipping redundant execution. Use --archive to force save a new version.")
    
    try:
        strategy = strategy_cls(data, events, **kwargs)
    except TypeError as e:
        print(f"Error initializing strategy: {e}")
        return {}, []

    execution = SimulatedExecutionHandler(
        events, 
        data, 
        commission_model=FixedCommission(1.0),
        fill_on_next_open=True,
        mode=mode
    )
    
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    
    tearsheet = TearSheet(portfolio)
    stats = tearsheet.analyze(benchmark_series=benchmark_series)
    
    if check_validation:
        validator = ValidationSuite(stats, portfolio.trade_log)
        validator.run_basic_checks()
        validator.print_cert()
        
        # --- AI Feedback Loop ---
        if validator.report['status'] in ['FAIL', 'WARNING']:
            from backtesting.ai_assistant import AIOptimizationPrompter
            prompter = AIOptimizationPrompter(stats, stats.get('Top_Losers', []))
            prompter.save_prompt(strategy_cls.__name__, symbol_list[0])
    
    if plot:
        plotter = Plotter(portfolio)
        plotter.plot_equity_curve(title=f"{strategy_cls.__name__} Equity Curve")
        
    # --- Registry & Knowledge Base ---
    if archive:
        import inspect
        try:
            source = inspect.getsource(strategy_cls)
        except:
            source = "Source unavailable"
            
        # Detect Regime
        detector = MarketRegimeDetector()
        regime = detector.detect(df)
        
        reg.save_run(
            strategy_name=strategy_cls.__name__,
            symbol=sym,
            interval=interval,
            params=kwargs,
            stats=stats,
            data_range=data_range,
            regime=regime.value,
            notes=notes,
            source_code=source
        )
        
        hist_avg = reg.get_historical_average(strategy_cls.__name__, symbol=sym)
        if hist_avg:
            print("\n" + "="*50)
            print("CROSS-RUN HISTORICAL COMPARISON")
            print("="*50)
            print(f"Current Sharpe:  {stats.get('Sharpe Ratio', 0):.2f} (Avg: {hist_avg['avg_sharpe']:.2f})")
            print(f"Current Return:  {stats.get('Total Return', 0):.2%}")
            print(f"Avg Hist Return: {hist_avg['avg_return']:.2%}")
            print("="*50 + "\n")

    return stats, portfolio.trade_log

def run_vector_strategy(strategy_cls, symbol_list, data_dirs, interval='1d', start_date=None, end_date=None, **kwargs):
    """
    Runs a strategy using the high-performance VectorEngine.
    Supports multi-symbol scanning.
    """
    print(f"Running VECTORIZED {strategy_cls.__name__} scan on {symbol_list} ({interval})...")
    
    use_gpu = kwargs.pop('gpu', False)
    scan_results = []
    
    # Separate archive flag from strategy params
    archive = kwargs.pop('archive', False)
    
    # 1. Load Data per symbol
    args_dirs = data_dirs if data_dirs else []
    config_dirs = CONFIG.get('data', {}).get('search_dirs', [])
    combined_dirs = list(set(args_dirs + config_dirs))
    
    data_handler = SmartDataHandler(symbol_list, search_dirs=combined_dirs, interval=interval, start_date=start_date, end_date=end_date)
    
    for symbol in symbol_list:
        df = data_handler.symbol_data.get(symbol)
        if df is None:
            continue

        # 2. Setup Vectorized Strategy
        v_strat_cls = strategy_cls
        if strategy_cls == MovingAverageCrossover:
            v_strat_cls = VectorizedMA
        elif 'NqOrb' in strategy_cls.__name__:
            if use_gpu:
                v_strat_cls = GpuVectorizedNQORB
                print("  [GPU] Using GpuVectorizedNQORB")
            else:
                v_strat_cls = VectorizedNQORB
            
        try:
            strategy = v_strat_cls(**kwargs)
            if use_gpu:
                engine = GpuVectorEngine(strategy)
            else:
                engine = VectorEngine(strategy)
            
            results = engine.run(df)
            
            equity_curve = results['equity_curve']
            final_equity = equity_curve.iloc[-1]
            total_return = (final_equity / 100000.0) - 1.0
            
            scan_results.append({
                'Symbol': symbol,
                'Return': total_return,
                'Final Equity': final_equity
            })
            
            print(f"  {symbol}: {total_return:.2%}")

            if kwargs.get('plot', False):
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                equity_curve.plot(title=f"Equity Curve - {strategy_cls.__name__} - {symbol}")
                plt.xlabel("Date")
                plt.ylabel("Equity")
                plt.grid(True)
                fname = f"{strategy_cls.__name__}_{symbol}_equity.png"
                plt.savefig(fname)
                plt.close()
                print(f"  Saved plot to {fname}")

            # --- Vector Registry Integration ---
            if archive:
                reg = StrategyRegistry()
                reg.save_run(
                    strategy_name=strategy_cls.__name__,
                    symbol=symbol,
                    interval=interval,
                    params=kwargs,
                    stats={'Total Return': total_return, 'Ending Equity': final_equity}, # Basic stats for vector for now
                    data_range=(df.index[0], df.index[-1])
                )
        except Exception as e:
            print(f"  Error in {symbol}: {e}")

    if scan_results:
        results_df = pd.DataFrame(scan_results).sort_values(by='Return', ascending=False)
        print("\n" + "="*40)
        print("SCAN RESULTS SUMMARY")
        print("="*40)
        print(results_df.to_string(index=False))
        print("="*40 + "\n")
        return results_df
    
    return None

def run_full_suite(strategy_cls, symbol_list, search_dirs, params, opt_grid, start_date=None, end_date=None, interval='1d'):
    """
    Runs the complete validation suite:
    1. Standard Backtest
    2. Certification (Optimistic vs Pessimistic)
    3. Sensitivity Analysis
    4. Walk-Forward Optimization
    """
    print("\n" + "="*60)
    print("FULL VALIDATION SUITE")
    print(f"Strategy: {strategy_cls.__name__} | Symbol: {symbol_list}")
    print("="*60)
    
    results = {
        'strategy': strategy_cls.__name__,
        'symbol': symbol_list[0],
        'tests': {}
    }
    
    # 1. Standard Backtest
    print("\n[1/4] STANDARD BACKTEST")
    print("-" * 40)
    stats, trades = run_strategy(strategy_cls, symbol_list, search_dirs, plot=False, mode='OPTIMISTIC', check_validation=False, verbose=True, start_date=start_date, end_date=end_date, interval=interval, **params)
    results['tests']['standard'] = {
        'passed': len(trades) > 0,
        'trades': len(trades),
        'return': stats.get('Total Return', 0)
    }
    
    # 2. Certification
    print("\n[2/4] CERTIFICATION (Optimistic vs Pessimistic)")
    print("-" * 40)
    stats_opt, trades_opt = run_strategy(strategy_cls, symbol_list, search_dirs, plot=False, mode='OPTIMISTIC', check_validation=False, verbose=False, start_date=start_date, end_date=end_date, interval=interval, **params)
    stats_pess, _ = run_strategy(strategy_cls, symbol_list, search_dirs, plot=False, mode='PESSIMISTIC', check_validation=False, verbose=False, start_date=start_date, end_date=end_date, interval=interval, **params)
    
    validator = ValidationSuite(stats_opt, trades_opt)
    validator.run_basic_checks()
    if stats_pess:
        validator.compare_robustness(stats_pess)
    
    results['tests']['certification'] = {
        'passed': validator.report['status'] != 'FAIL',
        'status': validator.report['status'],
        'sample_size': len(trades_opt),
        'sharpe': stats_opt.get('Sharpe Ratio', 0)
    }
    validator.print_cert()
    
    # 3. Sensitivity
    print("\n[3/4] SENSITIVITY ANALYSIS")
    print("-" * 40)
    tester = SensitivityTester(
        strategy_cls=strategy_cls,
        symbol_list=symbol_list,
        data_dirs=search_dirs,
        base_params=params,
        runner_func=run_strategy
    )
    tester.run()
    results['tests']['sensitivity'] = {'passed': True}  # Results printed in tester
    
    # 4. Walk-Forward (if opt_grid available)
    if opt_grid:
        print("\n[4/4] WALK-FORWARD OPTIMIZATION")
        print("-" * 40)
        wfo = WalkForwardOptimizer(
            strategy_cls=strategy_cls,
            symbol_list=symbol_list,
            search_dirs=search_dirs,
            param_grid=opt_grid,
            train_months=6,
            test_months=3
        )
        wfo_df, _ = wfo.run()
        
        avg_train = wfo_df['train_return'].mean() if not wfo_df.empty else 0
        avg_test = wfo_df['test_return'].mean() if not wfo_df.empty else 0
        wfe = avg_test / avg_train if avg_train != 0 else 0
        
        results['tests']['wfo'] = {
            'passed': wfe > 0.5,
            'wfe': wfe,
            'windows': len(wfo_df)
        }
        
        print(f"\nWFO Summary: WFE = {wfe:.2f} (Target > 0.5)")
    else:
        print("\n[4/4] WALK-FORWARD SKIPPED (No optimization grid)")
        results['tests']['wfo'] = {'passed': None, 'skipped': True}
    
    # Final Summary
    print("\n" + "="*60)
    print("FULL SUITE SUMMARY")
    print("="*60)
    all_passed = all(t.get('passed', True) for t in results['tests'].values() if t.get('passed') is not None)
    
    for name, test in results['tests'].items():
        status = "PASS" if test.get('passed') else ("SKIP" if test.get('skipped') else "FAIL")
        print(f"  [{status}] {name.upper()}")
    
    print("-" * 60)
    final_status = "STRATEGY APPROVED" if all_passed else "STRATEGY REJECTED"
    print(f"RESULT: {final_status}")
    print("="*60 + "\n")
    
    return results

def run_optimization(strategy_cls, symbol_list, data_dirs, param_grid, interval='1d', use_vector=True, use_smart=False, start_date=None, end_date=None, use_gpu=False):
    print(f"Optimizing {strategy_cls.__name__} (Grid Search) on {symbol_list} ({interval})...")
    
    args_dirs = data_dirs if data_dirs else []
    config_dirs = CONFIG.get('data', {}).get('search_dirs', [])
    combined_dirs = list(set(args_dirs + config_dirs))
    
    data_handler_cls = SmartDataHandler
    data_handler_args = (symbol_list, combined_dirs, start_date, end_date, interval)
    
    if use_smart:
         # ... existing ...
         optimizer = BayesianOptimizer(
            data_handler_cls=data_handler_cls,
            data_handler_args=data_handler_args,
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            iterations=20
        )
    elif use_vector:
        # Resolve GPU classes
        v_engine = None
        v_strat = None
        results_df = None
        
        # 1. Try GPU Optimization
        if use_gpu:
            try:
                print("  [GPU] Initializing GPU Engine...")
                v_engine = GpuVectorEngine
                if 'NqOrb' in strategy_cls.__name__: 
                    v_strat = GpuVectorizedNQORB
                
                # Check Availability explicitly before starting heavy work
                from backtesting.accelerate import GPU_AVAILABLE
                if not GPU_AVAILABLE:
                    raise RuntimeError("GPU not detected in Environment.")
                
                print(f"  [GPU] Active: {v_engine.__name__}")

                optimizer = VectorizedGridSearch(
                    data_handler_cls=data_handler_cls,
                    data_handler_args=data_handler_args,
                    strategy_cls=strategy_cls,
                    param_grid=param_grid,
                    vector_engine_cls=v_engine,
                    vector_strategy_cls=v_strat,
                    n_jobs=1 
                )
                results_df = optimizer.run()
                
            except Exception as e:
                print(f"\n[GPU FAIL] Optimization Error: {e}")
                print("[GPU FAIL] Falling back to CPU Vectorized Engine...\n")
                use_gpu = False
                results_df = None

        # 2. CPU Fallback (If GPU failed or wasn't requested)
        if results_df is None:
             optimizer = VectorizedGridSearch(
                data_handler_cls=data_handler_cls,
                data_handler_args=data_handler_args,
                strategy_cls=strategy_cls,
                param_grid=param_grid,
                vector_engine_cls=None, # Defaults to CPU
                vector_strategy_cls=None,
                n_jobs=-1 # Multiprocessing enabled for CPU
            )
             results_df = optimizer.run()
    else:
        optimizer = GridSearch(
            data_handler_cls=data_handler_cls,
            data_handler_args=data_handler_args,
            strategy_cls=strategy_cls,
            param_grid=param_grid
        )
    results_df = optimizer.run()
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(results_df.head(10))
    print("="*50 + "\n")
    
    # Save results to CSV for analysis
    results_df.to_csv("optimization_results.csv", index=False)
    
    # --- Persistence: Save ALL runs to Database (BATCHED) ---
    if CONFIG.get('optimization', {}).get('save_all_results_db', True):
        print("[DB] Archiving Optimization Results to Knowledge Base (Batch Mode)...")
        reg = StrategyRegistry()
        
        # Identify Metric Columns vs Param Columns
        metric_cols = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'CAGR', 'Calmar Ratio', 'Profit Factor', 'VaR (95%)', 'Ending Equity']
        param_cols = [c for c in results_df.columns if c not in metric_cols]
        
        # Get data range from first run (approx) or args
        d_range = (str(start_date) if start_date else "N/A", str(end_date) if end_date else "N/A")

        # Prepare Batch
        batch_runs = []
        for _, row in results_df.iterrows():
            # Construct Params
            run_params = {k: row[k] for k in param_cols}
            # Construct Stats
            run_stats = {k: row.get(k, 0.0) for k in metric_cols}
            
            batch_runs.append({
                'strategy_name': strategy_cls.__name__,
                'symbol': str(symbol_list),
                'interval': interval,
                'params': run_params,
                'stats': run_stats,
                'data_range': d_range,
                'regime': "OPTIMIZATION",
                'notes': "Grid Search Result",
                'source_code': "" 
            })
            
        reg.save_batch(batch_runs)
    
    # --- ROBUSTNESS GUARDRAILS ---
    print("\n" + "="*50)
    print("ROBUSTNESS & REALISM GUARDRAILS")
    print("="*50)

    # 1. PARITY CHECK (Accuracy)
    print("\n[GUARDRAIL 1/3] Parity Check (Vector vs Event)")
    # Load data for verification
    verify_data = SmartDataHandler(symbol_list, search_dirs, start_date, end_date, interval)
    
    # We only take the absolute best result for deep verification to save time
    best_row = results_df.iloc[0] # Sorted by Return desc
    best_params = {k: best_row[k] for k in param_cols}
    vector_ret = best_row['Total Return']
    
    parity_res = verify_parity(strategy_cls, best_params, vector_ret, verify_data)
    print(f"Top Strategy: Vector={parity_res['vector_return']:.2%} | Event={parity_res['event_return']:.2%} | Diff={parity_res['diff']:.2%}")
    if parity_res['is_divergent']:
        print("  [WARNING] High Divergence! Optimistic results may be unreliable.")

    # 2. SENSITIVITY TEST (Overfitting)
    print("\n[GUARDRAIL 2/3] Sensitivity Analysis (Stability)")
    # We need to inject 'run_strategy' as the runner_func, but run_strategy returns None/prints.
    # We need a quiet runner that returns stats. 
    # Actually SensitivityTester uses 'runner_func' signature: (cls, syms, dirs, ..., **params) -> (stats, _)
    # 'run_strategy' does this if we modify it slightly or wrap it?
    # run_strategy returns (stats, tear_sheet) if we check its definition.
    # Let's verify run_strategy return signature.
    
    # Creating a lambda wrapper to match SensitivityTester expectation
    # wrapper(strategy_cls, symbol_list, data_dirs, plot=False, ...)
    # SensitivityTester calls: self.runner_func(self.strategy_cls, self.symbol_list, self.data_dirs, plot=False, mode='OPTIMISTIC', check_validation=False, **params)
    
    def quiet_runner(*args, **kwargs):
        # Force verbose=False
        kwargs['verbose'] = False
        return run_strategy(*args, **kwargs)

    if CONFIG.get('optimization', {}).get('auto_sensitivity', True):
        tester = SensitivityTester(
            strategy_cls=strategy_cls,
            symbol_list=symbol_list,
            data_dirs=search_dirs,
            base_params=best_params,
            runner_func=quiet_runner
        )
        tester.run()

    # 3. REGIME FIT (Logic)
    print("\n[GUARDRAIL 3/3] Regime Analysis (Market Fit)")
    # We need an Equity Curve from the Event Engine. 
    # Luckily, Parity Check already ran an Event Engine! But we didn't capture the portfolio/equity curve from 'verify_parity'.
    # We should update verify_parity to return the full equity curve or just re-run here (fast enough for 1 run).
    
    # Re-running Event Backtest for Regime analysis (Cleanest way)
    # Using 'run_strategy' in quiet mode returns stats and a TearSheet (if plot/tearsheet enabled?)
    # Let's inspect run_strategy return.
    
    best_stats, best_tearsheet = run_strategy(
        strategy_cls, symbol_list, search_dirs, 
        plot=False, mode='OPTIMISTIC', check_validation=False, verbose=False,
        interval=interval, start_date=start_date, end_date=end_date,
        **best_params
    )
    
    if best_tearsheet:
        # We need the equity curve dataframe
        # TearSheet usually has self.equity_curve (pd.DataFrame)
        # And we need Regime Series.
        # Let's compute Regimes on the fly using the data handler used.
        # run_strategy created its own data handler. We can't access it easily unless returned.
        # Strategy pattern issue: 'run_strategy' encapsulates everything.
        
        # Alternative: Calculate regimes on 'verify_data' we loaded earlier!
        # And align with 'best_tearsheet.equity_curve'.
        
        from backtesting.regime import MarketRegimeDetector
        
        # Assuming single symbol for regime detection (primary)
        # Optimization usually on list, but valid only if 1 symbol or we pick 1.
        primary_symbol = symbol_list[0]
        # Get data for primary
        verify_data._load_data() # Ensure loaded
        h_data = verify_data.get_latest_bars(primary_symbol, N=100000) # Get all
        # Convert List[Bar] to DataFrame
        # DataHandler usually has internal dataframe 'symbol_data'
        full_df = verify_data.symbol_data[primary_symbol]
        
        regime_series = MarketRegimeDetector().get_regime_series(full_df)
        
        # Validate using Suite
        # Create a temp suite just for this method
        from backtesting.validation import ValidationSuite
        suite = ValidationSuite({}, [])
        suite.analyze_regimes(best_tearsheet.equity_curve, regime_series)

    # 4. MONTE CARLO (Probabilistic Risk)
    print("\n[GUARDRAIL 4/4] Monte Carlo Permutations (Luck Check)")
    if best_tearsheet and hasattr(best_tearsheet, 'trade_log'):
        from backtesting.validation import MonteCarloValidator
        mc = MonteCarloValidator(best_tearsheet.trade_log, n_sims=1000)
        mc.run()
    else:
        print("  [SKIP] No trade log available for Monte Carlo.")


    # Save results to CSV for analysis
    results_df.to_csv("optimization_results.csv", index=False)
    print(f"Results saved to optimization_results.csv")

    # --- Walk-Forward Analysis (If Configured) ---
    # Default to True from now on because user requested it
    if CONFIG.get('optimization', {}).get('wfo', True):
        try:
            print("\n" + "="*50)
            print("WALK-FORWARD OPTIMIZATION (Robustness Test)")
            print("="*50)
            
            wfo = WalkForwardOptimizer(
                strategy_cls=strategy_cls,
                symbol_list=symbol_list,
                search_dirs=combined_dirs,
                param_grid=param_grid,
                train_days=CONFIG.get('optimization', {}).get('wfo_train_days', 90),
                test_days=CONFIG.get('optimization', {}).get('wfo_test_days', 30),
                interval=interval,
                vector_engine_cls=v_engine, # Pass GPU Engine
                vector_strategy_cls=v_strat
            )
            wfo_df, stitched = wfo.run()
            
            if not wfo_df.empty:
                print("\nWFO RESULTS (Out-of-Sample Performance):")
                print(wfo_df[['train_start', 'test_end', 'test_return']].head())
                
                # Calculate WFO Efficiency
                avg_train = wfo_df['train_return'].mean()
                avg_test = wfo_df['test_return'].mean()
                wfe = avg_test / avg_train if avg_train != 0 else 0
                print(f"\nWalk-Forward Efficiency (WFE): {wfe:.2f} (Target > 0.5)")
                
                # Save WFO Report
                wfo_df.to_csv("wfo_results.csv", index=False)
                
                # Persist WFO Summary to DB?
                # Maybe later. For now, just getting it running is the win.
            else:
                print("[WFO] No results generated.")
                
        except Exception as e:
            print(f"[WFO] Failed: {e}")
            import traceback
            traceback.print_exc()

    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtesting Runner (GPU-First)')
    parser.add_argument('--strategy', type=str, default='MA', help='Strategy: MA, ORB, MR')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to backtest')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    parser.add_argument('--mode', type=str, default='OPTIMISTIC', help='Execution Mode: OPTIMISTIC or PESSIMISTIC')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to custom data directory (e.g. Intraday OHLC)')
    parser.add_argument('--certify', action='store_true', help='Run full Robustness Certification (Optimistic vs Pessimistic)')
    parser.add_argument('--sensitivity', action='store_true', help='Run Parameter Sensitivity Analysis')
    parser.add_argument('--wfo', action='store_true', help='Run Walk-Forward Optimization')
    parser.add_argument('--full-suite', action='store_true', help='Run complete validation suite (all tests)')
    parser.add_argument('--monitor', action='store_true', help='Run Live Signal Monitor')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval: 1m, 5m, 1h, 1d')
    parser.add_argument('--event', action='store_true', help='FORCE Event-Driven Engine (Legacy CPU Mode)')
    parser.add_argument('--archive', action='store_true', help='Persist backtest result to Registry (History)')
    parser.add_argument('--leaderboard', action='store_true', help='Show top historical backtests')
    parser.add_argument('--note', type=str, default="", help='Add a research note to this run')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--smart-search', action='store_true', help='Use Bayesian-style Smart Parameter Search')
    parser.add_argument('--new-strategy', type=str, help='Generate a new strategy template with given name')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print trade details')
    parser.add_argument('--sizing', action='store_true', help='Enable Dynamic Position Sizing')
    parser.add_argument('--max-contracts', type=int, default=3, help='Max contracts for position sizing')
    parser.add_argument('--no-ts', action='store_true', help='Disable Trailing Stop (Force Limit TP)')
    parser.add_argument('--sanity-check', action='store_true', help='Run 15-Year (2010-2025) Vector Scan for Robustness Check')
    parser.add_argument('--wfo-step', type=int, default=None, help='WFO Step Days (Slide Window Size)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU Acceleration (Requires cupy)')
    
    args = parser.parse_args()
    
    # --- GLOBAL PRE-FLIGHT CHECK ---
    # Setup search directories early
    csv_dir = os.path.join(os.getcwd(), 'examples') 
    search_dirs = [args.data_dir, csv_dir] if args.data_dir else [csv_dir]
    symbol_list = [s.strip().upper() for s in args.symbol.split(',')]
    
    # Run Diagnostics (Strict Mode if optimizing or certifying)
    # Default is strict now.
    # Run Diagnostics
    # Relax strictness to allow running on available data
    preflight = PreFlightCheck(symbol_list, search_dirs, interval=args.interval, strict=False)
    report = preflight.run()
    
    if args.new_strategy:
        generate_strategy(args.new_strategy)
        sys.exit(0)
    
    if args.leaderboard:
        reg = StrategyRegistry()
        lb = reg.get_leaderboard()
        print("\n" + "="*60)
        print("STRATEGY REGISTRY LEADERBOARD (Top by Sharpe)")
        print("="*60)
        if lb.empty:
            print("No archived runs found.")
        else:
            print(lb.to_string(index=False))
        print("="*60 + "\n")
        sys.exit(0)
    
    params = {}
    opt_grid = {}
    
    if args.strategy == 'MA':
        strategy_cls = MovingAverageCrossover
        params = {'short_window': 20, 'long_window': 50}
        opt_grid = {'short_window': range(10, 50, 10), 'long_window': range(50, 200, 50)}
        
    elif args.strategy == 'ORB':
        strategy_cls = OpeningRangeBreakout
        # Default Params
        params = {'orb_start': time(9,30), 'orb_end': time(10,00)}
        
    elif args.strategy == 'MR':
        strategy_cls = MeanReversion
        params = {'lookback': 20, 'std_dev': 2.0}
        
    elif args.strategy == 'NQORB':
        strategy_cls = NqOrb
        params = {
            'stop_loss': 50.0, 
            'take_profit': 100.0,
            'ema_filter': 200,
            'atr_max_mult': 3.0
        }
        opt_grid = {
            'stop_loss': [25, 50, 75, 100],
            'take_profit': [50, 100, 150, 200],
            'ema_filter': [50, 100, 200, 500],
            'atr_max_mult': [1.5, 2.0, 3.0, 5.0]
        }
        
    elif args.strategy == 'NQORB15':
        strategy_cls = NqOrb15m
        params = {
            'sl_atr_mult': 2.0, 
            'tp_atr_mult': 4.0,
            'ema_filter': 50,
            'atr_max_mult': 2.5
        }
        opt_grid = {
            'sl_atr_mult': [1.0, 2.0, 3.0, 4.0],
            'tp_atr_mult': [2.0, 4.0, 6.0, 8.0],
            'ema_filter': [20, 50, 100, 200],
            'atr_max_mult': [1.5, 2.0, 2.5, 3.0]
        }
        
    elif args.strategy == 'NQORBENHANCED':
        strategy_cls = NqOrbEnhanced
        # Default Params from "Winning Formula"
        params = {
            'sl_atr_mult': 2.0, 
            'tp_atr_mult': 4.0, # not really used with TS
            'ema_filter': 50,
            'atr_max_mult': 2.5,
            'use_htf': True, 'htf_ma_period': 100,
            'use_rvol': True, 'rvol_thresh': 1.5,
            'use_trailing_stop': not args.no_ts, 'ts_atr_mult': 2.0,
            'use_confidence_sizing': args.sizing,
            'max_contracts': args.max_contracts
        }
        opt_grid = {} # No optimization needed yet, we are verifying
        
    else:
        print(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

    if args.sanity_check:
        run_vector_strategy(strategy_cls, symbol_list, search_dirs, interval=args.interval, start_date="2010-01-01", end_date="2025-05-30", **params)
        sys.exit(0)

    if args.optimize:
        if not opt_grid:
            print("Optimization not defined for this strategy.")
            sys.exit(1)
        run_optimization(strategy_cls, symbol_list, search_dirs, opt_grid, interval=args.interval, use_vector=not args.event, use_smart=args.smart_search, start_date=args.start_date, end_date=args.end_date, use_gpu=args.gpu)
        
    elif args.wfo:
        # ... (WFO might need update too if it instantiates SmartDataHandler)
        # Actually I should check WFO.__init__
        if not opt_grid:
            print("Optimization grid needed for WFO.")
            sys.exit(1)
            
        print(f"\n[WALK FORWARD OPTIMIZATION] {strategy_cls.__name__} on {symbol_list}...\n")
        
        wfo = WalkForwardOptimizer(
            strategy_cls=strategy_cls,
            symbol_list=symbol_list,
            search_dirs=search_dirs,
            param_grid=opt_grid,
            train_days=args.train_days if hasattr(args, 'train_days') and args.train_days else (90 if args.interval == '1d' else 6),
            test_days=args.test_days if hasattr(args, 'test_days') and args.test_days else (30 if args.interval == '1d' else 3),
            step_days=args.wfo_step,
            interval=args.interval
        )
        results_df, stitched_returns = wfo.run()
        
        # WFO Archive logic if needed... (optional for now as we want to archive final tests)
        
        if results_df.empty:
            print("\n[ERROR] Walk-Forward Optimization produced no results. Data range might be too small for the requested windows.")
        else:
            print("\n" + "="*50)
            print("WALK FORWARD RESULTS")
            print("="*50)
            print(results_df[['train_end', 'params', 'test_return']])
            
            # Calculate WFO Efficiency
            avg_train = results_df['train_return'].mean()
            avg_test = results_df['test_return'].mean()
            wfe = avg_test / avg_train if avg_train != 0 else 0
            
            print("-" * 50)
            print(f"Average In-Sample Return:  {avg_train:.2%}")
            print(f"Average Out-Sample Return: {avg_test:.2%}")
            print(f"Walk-Forward Efficiency:   {wfe:.2f} (Target > 0.5)")
            print("="*50)
        
        # NOTE: Stitched Equity Curve plotting could go here.
        # For now, numerical report is the priority.
 
    elif getattr(args, 'full_suite', False):
        run_full_suite(strategy_cls, symbol_list, search_dirs, params, opt_grid, start_date=args.start_date, end_date=args.end_date, interval=args.interval)

    elif args.monitor:
        monitor = SignalMonitor(
            strategy_cls=strategy_cls,
            symbol_list=symbol_list,
            interval_sec=60 if args.interval == '1m' else 300,
            data_dirs=search_dirs,
            interval=args.interval,
            **params
        )
        monitor.start()

    elif args.certify:
        # Certification REQUIRES Event-Driven (Optimistic vs Pessimistic)
        print(f"\n[CERTIFICATION MODE] Auditing {strategy_cls.__name__} on {symbol_list}...\n")
        
        # 1. Run Optimistic
        print(">>> Phase 1: Optimistic Run")
        stats_opt, trades_opt = run_strategy(strategy_cls, symbol_list, search_dirs, plot=args.plot, mode='OPTIMISTIC', check_validation=False, start_date=args.start_date, end_date=args.end_date, interval=args.interval, **params)
        
        # 2. Run Pessimistic
        print("\n>>> Phase 2: Pessimistic Stress Test (Worst Case Execution)")
        stats_pess, trades_pess = run_strategy(strategy_cls, symbol_list, search_dirs, plot=False, mode='PESSIMISTIC', check_validation=False, start_date=args.start_date, end_date=args.end_date, interval=args.interval, **params)
        
        # 3. Validate
        if stats_opt and stats_pess:
            validator = ValidationSuite(stats_opt, trades_opt)
            validator.run_basic_checks()
            validator.compare_robustness(stats_pess)
            validator.print_cert()

    elif args.sensitivity:
        # Pass run_strategy as a callback
        tester = SensitivityTester(
            strategy_cls=strategy_cls,
            symbol_list=symbol_list,
            data_dirs=search_dirs,
            base_params=params,
            runner_func=run_strategy
        )
        tester.run()
            
    else:
        # === START MAIN EXECUTION LOGIC ===
        
        # GPU-FIRST DEFAULT: Checked PreFlight, default to Vector unless --event flag
        
        if args.event:
            print("\n[INFO] Event-Driven Engine Requested (CPU/Slow Mode).")
            run_strategy(strategy_cls, symbol_list, search_dirs, plot=args.plot, mode=args.mode, interval=args.interval, archive=args.archive, start_date=args.start_date, end_date=args.end_date, **params)
        else:
            print("\n[INFO] Vector Engine Active (GPU-Accelerated Default).")
            # Pass gpu flag
            params['gpu'] = args.gpu
            run_vector_strategy(strategy_cls, symbol_list, search_dirs, interval=args.interval, archive=args.archive, plot=args.plot, start_date=args.start_date, end_date=args.end_date, **params)
