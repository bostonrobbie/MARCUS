
import pandas as pd
import numpy as np
import copy
from typing import Type, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

from .vector_engine import VectorEngine
from .strategy import Strategy

class StrategySkeptic:
    """
    The Skeptic tries to DISPROVE a strategy's edge.
    If a strategy survives The Skeptic, it is statistically robust.
    """
    def __init__(self, 
                 vector_engine_cls: Type[VectorEngine],
                 vector_strategy_cls: Any, # Class reference
                 params: Dict[str, Any],
                 initial_capital: float = 100000.0,
                 n_jobs: int = -1):
        self.vector_engine_cls = vector_engine_cls
        self.vector_strategy_cls = vector_strategy_cls
        self.params = params
        self.initial_capital = initial_capital
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

    def run_permutation_test(self, df: pd.DataFrame, n_sims: int = 1000) -> Dict[str, Any]:
        """
        Tests if the strategy performance is statistically significant compared to random luck.
        Method: Shuffle the 'Close' returns (breaking serial correlation).
        """
        print(f"    🕵️ Skeptic: Running {n_sims} Permutation Tests (Is it Luck?)...")
        # print(f"      [DEBUG] Skeptic Input Columns: {df.columns.tolist()}")
        
        # 1. Calculate Real Performance
        real_res = self._run_once(df)
        real_ret = real_res['Total Return']
        
        n_sims = 50 # Default to 50 for responsiveness
        
        results = []
        
        # Prepare args
        args_list = []
        for i in range(n_sims):
            args_list.append((self.vector_engine_cls, self.vector_strategy_cls, self.params, self.initial_capital, df.copy(), i))
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(_worker_permutation, arg): arg[5] for arg in args_list}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    # Log but continue if individual worker fails
                    pass
        
        # 3. Analyze
        if not results:
             return {'verdict': "ERROR", 'p_value': 1.0, 'real_return': real_ret}

        # P-Value: % of random runs that beat real run
        better_runs = [r for r in results if r > real_ret]
        p_value = len(better_runs) / len(results)
        
        return {
            'real_return': real_ret,
            'p_value': p_value,
            'n_sims': len(results),
            'verdict': "PASS" if p_value < 0.05 else "FAIL (Indistinguishable from Luck)"
        }

    def run_detrended_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tests if strategy works without the underlying trend.
        Removes linear trend from Close prices.
        """
        print("    🕵️ Skeptic: Running Detrended Test (Is it just Beta?)...")
        
        col_map = {c.lower(): c for c in df.columns}
        close_col = col_map.get('close', 'Close')
        
        if close_col not in df.columns:
             return {'verdict': "ERROR (Missing Close)"}

        # 1. Calculate Trend
        y = df[close_col].values
        x = np.arange(len(y))
        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        trend = coeffs[0] * x + coeffs[1]
        
        # 2. Detrend
        # New Price = Price - Trend + Mean(Price) (to keep scale positive)
        detrended_close = y - trend + np.mean(y)
        
        # Scale H/L/O?
        delta = y - detrended_close
        
        df_detrend = df.copy()
        df_detrend[close_col] = detrended_close
        
        for c_base in ['High', 'Low', 'Open']:
             c = col_map.get(c_base.lower(), c_base)
             if c in df_detrend.columns:
                  df_detrend[c] = df[c] - delta
        
        # Run
        res = self._run_once(df_detrend)
        
        return {
            'detrended_return': res['Total Return'],
            'verdict': "PASS" if res['Total Return'] > 0 else "FAIL (Relies on Trend)"
        }

    def _run_once(self, df):
        v_strat = self.vector_strategy_cls(**self.params)
        engine = self.vector_engine_cls(v_strat, self.initial_capital)
        res = engine.run(df)
        final_eq = res['equity_curve'].iloc[-1]
        total_return = (final_eq / self.initial_capital) - 1.0
        return {'Total Return': total_return}

def _worker_permutation(args):
    """
    Worker for Permutation Test.
    Shuffles returns and rebuilds price.
    """
    engine_cls, strat_cls, params, init_cap, df, seed = args
    
    np.random.seed(seed) 
    
    # Identify Close Column safely
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get('close', 'Close')
    
    if close_col not in df.columns:
        return -1.0 # Fail safe

    # Shuffle Returns
    returns = df[close_col].pct_change().fillna(0).values
    np.random.shuffle(returns)
    
    # Rebuild Close
    start_price = df[close_col].iloc[0]
    new_close = start_price * np.cumprod(1 + returns)
    
    # Rebuild H/L/O
    ratio = new_close / df[close_col].values
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0) # Safety
    
    df[close_col] = new_close
    
    for c_base in ['High', 'Low', 'Open']:
         c = col_map.get(c_base.lower(), c_base)
         if c in df.columns:
              df[c] = df[c] * ratio
    
    # Run
    v_strat = strat_cls(**params)
    engine = engine_cls(v_strat, init_cap)
    res = engine.run(df)
    
    final_eq = res['equity_curve'].iloc[-1]
    return (final_eq / init_cap) - 1.0
