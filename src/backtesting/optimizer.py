import itertools
import pandas as pd
from typing import Dict, List, Type
from queue import Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

from .data import DataHandler
from .portfolio import Portfolio
from .execution import SimulatedExecutionHandler, FixedCommission
from .engine import BacktestEngine
from .strategy import Strategy
from .wfo_analytics import WFOAnalytics, analyze_wfo_results

# --- Worker function for parallel execution (must be at module level for pickling) ---
from .data import MemoryDataHandler

def _run_single_backtest(args):
    """
    Runs a single backtest with given parameters.
    Returns a dict with params and results.
    Refactored to receive PRE-LOADED data (dict of dfs) to avoid disk I/O.
    """
    # args signature changed: (data_dict, strategy_cls, params, initial_capital)
    data_dict, strategy_cls, params, initial_capital = args
    
    try:
        # 1. Data (Instant In-Memory)
        data_handler = MemoryDataHandler(data_dict)
        
        # 2. Events & Portfolio
        events = Queue()
        portfolio = Portfolio(data_handler, events, initial_capital)
        
        # 3. Strategy with specific params
        strategy = strategy_cls(data_handler, events, **params)
        
        # 4. Execution
        execution = SimulatedExecutionHandler(events, data_handler, commission_model=FixedCommission(1.0))
        
        # 5. Engine
        engine = BacktestEngine(data_handler, strategy, portfolio, execution)
        engine.run()
        
        # 6. Capture Metrics
        equity_curve = pd.DataFrame(portfolio.equity_curve)
        if not equity_curve.empty:
            total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1.0
            return {
                **params,
                'Total Return': total_return,
                'Final Equity': equity_curve['equity'].iloc[-1]
            }
        else:
            return {
                **params,
                'Total Return': 0.0,
                'Final Equity': initial_capital
            }
    except Exception as e:
        return {
            **params,
            'Total Return': 0.0,
            'Final Equity': initial_capital,
            'Error': str(e)
        }

class GridSearch:
    """
    Iterates over a range of parameters for a given strategy.
    Supports parallel execution using multiple CPU cores.
    """
    def __init__(self, 
                 data_handler_cls: Type[DataHandler],
                 data_handler_args: tuple,
                 strategy_cls: Type[Strategy], 
                 param_grid: Dict[str, List],
                 initial_capital: float = 100000.0,
                 n_jobs: int = -1,
                 vector_strategy_cls=None):
        self.data_handler_cls = data_handler_cls
        self.data_handler_args = data_handler_args
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.initial_capital = initial_capital
        
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        self.results = []

    def _generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def run(self, parallel=True):
        combinations = self._generate_param_combinations()
        n_combos = len(combinations)
        print(f"Starting Grid Search with {n_combos} combinations using {self.n_jobs} workers...")
        
        # --- PHASE 1 OPTIMIZATION: SINGLE DATA LOAD ---
        print("  [Main Process] Loading Data from Disk...")
        # Instantiate the DataHandler (e.g. SmartDataHandler) ONCE here
        loader = self.data_handler_cls(*self.data_handler_args)
        
        # Extract the raw DataFrames to a dictionary
        # This dict is what we will pickle and send to workers.
        # Check if loader.symbol_data is populated
        if not loader.symbol_data:
             print("  [Error] No data loaded.")
             return pd.DataFrame()
             
        preload_data = loader.symbol_data
        print(f"  [Main Process] Data Loaded. Symbols: {list(preload_data.keys())}")
        
        if parallel and n_combos > 1 and self.n_jobs > 1:
            # Construct Args: (data_dict, strategy_cls, params, initial_capital)
            # Passing 'preload_data' (dict of DFs)
            args_list = [
                (preload_data, self.strategy_cls, params, self.initial_capital)
                for params in combinations
            ]
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(_run_single_backtest, args): args[2] for args in args_list}
                
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    if 'Error' not in result:
                        # Less verbose to speed up console I/O
                        # print(f"Completed: {futures[future]} -> Return: {result.get('Total Return', 0):.2%}")
                        pass
        else:
            for params in combinations:
                print(f"Testing params: {params}")
                # Use the optimized signature even for sequential
                args = (preload_data, self.strategy_cls, params, self.initial_capital)
                result = _run_single_backtest(args)
                self.results.append(result)

        return pd.DataFrame(self.results).sort_values(by='Total Return', ascending=False)


class WalkForwardOptimizer:
    """
    Rolling Walk-Forward Analysis.
    Train on X days, Test on Y days, Roll forward by Step Z (default Step = Test).
    """
    def __init__(self, 
                 strategy_cls: Type[Strategy], 
                 symbol_list: List[str], 
                 search_dirs: List[str],
                 param_grid: Dict[str, List],
                 train_days: int = 90,
                 test_days: int = 30,
                 step_days: int = None,
                 initial_capital: float = 100000.0,
                 interval: str = '1d',
                 vector_engine_cls=None,
                 vector_strategy_cls=None):
        self.strategy_cls = strategy_cls
        self.symbol_list = symbol_list
        self.search_dirs = search_dirs
        self.param_grid = param_grid
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days if step_days else test_days
        self.initial_capital = initial_capital
        self.interval = interval
        self.vector_engine_cls = vector_engine_cls
        self.vector_strategy_cls = vector_strategy_cls
        
        from .data import SmartDataHandler
        self.data_handler_cls = SmartDataHandler

    def run(self):
        print(f"\nSTARTING WALK-FORWARD OPTIMIZATION ({self.train_days}d Train -> {self.test_days}d Test, Step {self.step_days}d)")
        
        full_data = self.data_handler_cls(self.symbol_list, self.search_dirs, interval=self.interval)
        sym = self.symbol_list[0]
        df = full_data.symbol_data[sym]
        
        start_date = df.index[0]
        end_date = df.index[-1]
        
        print(f"Data Range: {start_date} to {end_date}")
        
        current_train_start = start_date
        wfo_results = []
        stitched_equity = []
        
        while True:
            # Define Windows
            train_end = current_train_start + pd.DateOffset(days=self.train_days)
            test_end = train_end + pd.DateOffset(days=self.test_days)
            
            if train_end > end_date:
                break
                
            test_end = min(test_end, end_date)
            
            print(f"\n>>> Window: Train[{current_train_start.date()} : {train_end.date()}] -> Test[{train_end.date()} : {test_end.date()}]")
            
            # A. OPTIMIZE (In-Sample)
            print("    Optimizing (Vectorized)...")
            optimizer = VectorizedGridSearch(
                data_handler_cls=self.data_handler_cls,
                data_handler_args=(self.symbol_list, self.search_dirs, current_train_start, train_end, self.interval),
                strategy_cls=self.strategy_cls,
                param_grid=self.param_grid,
                initial_capital=self.initial_capital,
                vector_engine_cls=self.vector_engine_cls,
                vector_strategy_cls=self.vector_strategy_cls
            )
            df_results = optimizer.run()
            
            if df_results.empty:
                print("    No trades in training window. Skipping.")
                current_train_start = current_train_start + pd.DateOffset(days=self.step_days)
                continue

            best_params = df_results.iloc[0].to_dict()
            clean_params = {}
            for k, v in best_params.items():
                if k in self.param_grid:
                    if isinstance(v, float) and v.is_integer():
                        clean_params[k] = int(v)
                    else:
                        clean_params[k] = v
            print(f"    Best Params: {clean_params} (Ret: {best_params.get('Total Return',0):.2%})")
            
            # B. TEST (Out-Of-Sample)
            print("    Testing Out-of-Sample...")
            # For Validation step, we use the Event Engine (standard BacktestEngine)
            # Need to pass data correctly. 
            # SmartDataHandler is fine here because WFO doesn't run in parallel (yet).
            oos_data = self.data_handler_cls(self.symbol_list, self.search_dirs, train_end, test_end, interval=self.interval)
            events = Queue()
            portfolio = Portfolio(oos_data, events, initial_capital=100000.0) 
            strategy = self.strategy_cls(oos_data, events, **clean_params)
            execution = SimulatedExecutionHandler(events, oos_data, commission_model=FixedCommission(1.0))
            engine = BacktestEngine(oos_data, strategy, portfolio, execution)
            engine.run()
            
            segment_res = {
                'train_start': current_train_start,
                'train_end': train_end,
                'test_end': test_end,
                'params': clean_params,
                'train_return': best_params.get('Total Return', 0),
                'test_return': 0.0,
                'train_trades': 0,
                'test_trades': 0
            }
            
            eq_curve = pd.DataFrame(portfolio.equity_curve)
            if not eq_curve.empty:
                 start_eq = eq_curve['equity'].iloc[0]
                 end_eq = eq_curve['equity'].iloc[-1]
                 seg_ret = (end_eq / start_eq) - 1.0
                 segment_res['test_return'] = seg_ret
                 segment_res['test_trades'] = len(portfolio.trade_log)

                 if 'returns' not in eq_curve:
                      eq_curve['returns'] = eq_curve['equity'].pct_change().fillna(0)
                 stitched_equity.append(eq_curve['returns'])

            wfo_results.append(segment_res)
            
            # MOVE FORWARD (Rolling via Step)
            current_train_start = current_train_start + pd.DateOffset(days=self.step_days)
            if test_end >= end_date:
                break
                
        wfo_df = pd.DataFrame(wfo_results)

        # Run WFO Analytics
        if not wfo_df.empty:
            try:
                wfo_analysis = analyze_wfo_results(wfo_df)
                print("\n" + "="*50)
                print("WFO ANALYTICS SUMMARY")
                print("="*50)
                summary = wfo_analysis.get('wfo_summary', {})
                if summary:
                    print(f"Health Score: {summary.get('health_score', 0):.0f}/100")
                    print(f"Verdict: {summary.get('overall_verdict', 'N/A')}")

                    divergence = summary.get('divergence', {})
                    if divergence:
                        print(f"IS/OOS Degradation: {divergence.get('degradation_pct', 0):.1f}%")

                    stability = summary.get('stability', {})
                    if stability:
                        print(f"Parameter Stability: {stability.get('overall_stability', 0):.2f}")

                    consistency = summary.get('consistency', {})
                    if consistency:
                        print(f"OOS Consistency: {consistency.get('consistency', 0):.0%}")
                print("="*50)
            except Exception as e:
                print(f"[WFO Analytics] Could not run: {e}")

        return wfo_df, stitched_equity


# --- Helper for Parallel Vectorized Backtest ---
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def _run_single_vector_backtest(args):
    """
    Runs a single vectorized backtest.
    args: (vector_engine_cls, v_strat_cls, params, initial_capital, df)
    """
    vector_engine_cls, v_strat_cls, params, initial_capital, df = args
    try:
        v_strat = v_strat_cls(**params)
        engine = vector_engine_cls(v_strat, initial_capital)
        res = engine.run(df)
        
        final_eq = res['equity_curve'].iloc[-1]
        total_return = (final_eq / initial_capital) - 1.0
        
        return {
            **params,
            'Total Return': total_return,
            'Final Equity': final_eq
        }
    except Exception as e:
        return {
            **params,
            'Total Return': 0.0,
            'Final Equity': initial_capital,
            'Error': str(e)
        }

class VectorizedGridSearch:
    """
    Ultra-High-Performance Parameter Optimizer.
    Uses VectorEngine to run backtests in bulk.
    Supports parallel CPU execution.
    """
    def __init__(self, 
                 data_handler_cls: Type[DataHandler],
                 data_handler_args: tuple,
                 strategy_cls: Type[Strategy], 
                 param_grid: Dict[str, List],
                 initial_capital: float = 100000.0,
                 n_jobs: int = -1,
                 vector_strategy_cls=None,
                 vector_engine_cls=None):
        self.data_handler_cls = data_handler_cls
        self.data_handler_args = data_handler_args
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.initial_capital = initial_capital
        self.results = []
        
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        self.vector_strategy_cls = vector_strategy_cls
        
        from .vector_engine import VectorEngine, VectorizedMA, VectorizedNQORB
        self.vector_engine_cls = vector_engine_cls if vector_engine_cls else VectorEngine
        
        self.v_strategy_map = {
            'MovingAverageCrossover': VectorizedMA,
            'NqOrb': VectorizedNQORB,
            'NqOrb15m': VectorizedNQORB
        }

    def _generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def run(self):
        combinations = self._generate_param_combinations()
        print(f"Starting VECTORIZED Grid Search with {len(combinations)} combinations using {self.n_jobs} workers...")
        
        data_handler = self.data_handler_cls(*self.data_handler_args)
        symbol = self.data_handler_args[0][0]
        df = data_handler.symbol_data.get(symbol)
        
        if df is None:
            return pd.DataFrame()

        strat_name = self.strategy_cls.__name__
        
        v_strat_cls = None
        if self.vector_strategy_cls:
            v_strat_cls = self.vector_strategy_cls
        else:
            v_strat_cls = self.v_strategy_map.get(strat_name)
        
        if not v_strat_cls:
            print(f"No vectorized strategy found for {strat_name}")
            return pd.DataFrame()

        # Prepare arguments for parallel execution
        # Note: passing large DF to workers via pickle can be slow. 
        # But for 15m data over 15 years (~300k rows), it's manageable (few MBs).
        
        if self.n_jobs > 1 and len(combinations) > 1:
            args_list = [
                (self.vector_engine_cls, v_strat_cls, params, self.initial_capital, df)
                for params in combinations
            ]
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(_run_single_vector_backtest, args): args[2] for args in args_list}
                
                for future in as_completed(futures):
                    res = future.result()
                    self.results.append(res)
        else:
            # Sequential Fallback
            for params in combinations:
                args = (self.vector_engine_cls, v_strat_cls, params, self.initial_capital, df)
                res = _run_single_vector_backtest(args)
                self.results.append(res)

        return pd.DataFrame(self.results).sort_values(by='Total Return', ascending=False)

class BayesianOptimizer:
    """
    Smarter parameter search using a 'Simulated Annealing' / 'Random Refinement' approach.
    Finds optimal parameters faster than exhaustive Grid Search.
    
    Refactored to use Single-Load MemoryDataHandler.
    """
    def __init__(self, 
                 data_handler_cls: Type[DataHandler],
                 data_handler_args: tuple,
                 strategy_cls: Type[Strategy], 
                 param_grid: Dict[str, List],
                 initial_capital: float = 100000.0,
                 iterations: int = 20):
        self.data_handler_cls = data_handler_cls
        self.data_handler_args = data_handler_args
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.initial_capital = initial_capital
        self.iterations = iterations
        self.results = []

    def run(self):
        import random
        print(f"Starting SMART SEARCH (Bayesian-Style) with {self.iterations} iterations...")
        
        # --- PRELOAD DATA ---
        print("  [Main Process] Loading Data from Disk...")
        loader = self.data_handler_cls(*self.data_handler_args)
        if not loader.symbol_data:
             return pd.DataFrame()
        preload_data = loader.symbol_data
        print(f"  [Main Process] Data Loaded.")
        
        # 1. Start with some random samples to explore the space
        best_return = -float('inf')
        best_params = None
        
        # Keys to iterate
        keys = list(self.param_grid.keys())
        
        for i in range(self.iterations):
            # Pick random params from grid
            current_params = {k: random.choice(self.param_grid[k]) for k in keys}
            
            # If we already have a best, maybe nudge it slightly (Exploitation)
            if best_params and random.random() > 0.5:
                # Local refinement: Pick one parameter and move it slightly within the grid
                refine_key = random.choice(keys)
                current_val = best_params[refine_key]
                grid_vals = self.param_grid[refine_key]
                idx = grid_vals.index(current_val)
                # Move up or down index
                new_idx = max(0, min(len(grid_vals)-1, idx + random.choice([-1, 1])))
                current_params = best_params.copy()
                current_params[refine_key] = grid_vals[new_idx]

            # Run backtest (Using Memory Loader Signature)
            args = (preload_data, self.strategy_cls, current_params, self.initial_capital)
            res = _run_single_backtest(args)
            
            ret = res.get('Total Return', 0.0)
            self.results.append(res)
            
            if ret > best_return:
                best_return = ret
                best_params = current_params
                print(f"  Iteration {i}: New Best Found! Return: {ret:.2%} Params: {best_params}")

        return pd.DataFrame(self.results).sort_values(by='Total Return', ascending=False)
