
import numpy as np
import pandas as pd
from typing import List, Dict

class MonteCarloSimulator:
    """
    Performs Monte Carlo simulation on an equity curve.
    Shuffles trade returns to generate thousands of alternative realities.
    """
    def __init__(self, returns: pd.Series, n_simulations: int = 2500, initial_equity: float = 100000.0):
        self.returns = returns
        self.n_simulations = n_simulations
        self.initial_equity = initial_equity
        self.results = []
    
    def run(self) -> Dict[str, float]:
        """
        Runs the simulation.
        Returns risk metrics: Drawdown at 95% confidence, Ruin Probability, etc.
        """
        if self.returns.empty:
            return {}
            
        final_equities = []
        max_drawdowns = []
        
        # Convert to numpy for speed
        rets = self.returns.values
        n_trades = len(rets)
        
        for _ in range(self.n_simulations):
            # 1. Shuffle returns (Bootstrap)
            shuffled_rets = np.random.choice(rets, size=n_trades, replace=True)
            
            # 2. Reconstruct Equity Curve
            # (1 + r).cumprod() * initial
            equity_curve = self.initial_equity * np.cumprod(1 + shuffled_rets)
            
            final_equities.append(equity_curve[-1])
            
            # 3. Calculate Drawdown for this path
            running_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - running_max) / running_max
            max_drawdowns.append(dd.min()) # min is negative, so largest DD
            
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        # Stats
        return {
            'mc_median_cagr': 0.0, # Todo: calc from median equity
            'mc_var_95': np.percentile(max_drawdowns, 5), # 5th percentile of DD (e.g. -0.25)
            'mc_worst_case_dd': np.min(max_drawdowns),
            'mc_ruin_prob': np.mean(final_equities < (self.initial_equity * 0.5)), # Ruin = 50% loss
            'mc_median_equity': np.median(final_equities)
        }

class ParameterStabilityAnalysis:
    """
    Analyzes how 'sharp' the peak is.
    Check neighbors of the best parameter set.
    """
    def __init__(self, optimizer_results: pd.DataFrame, param_names: List[str]):
        self.df = optimizer_results
        self.param_names = param_names
        
    def run(self, best_params: Dict) -> Dict[str, float]:
        """
        Calculates stability score.
        Avg return of neighbors / best return.
        Stability > 0.8 is desirable.
        """
        # Find neighbors (e.g. +/- 1 step in grid)
        # This assumes the dataframe contains all grid combinations.
        
        # Simple approach: Top 10% average vs Best
        if self.df.empty:
            return {}
            
        top_return = self.df['Total Return'].max()
        top_10_percent = self.df['Total Return'].quantile(0.90)
        
        avg_top_tier = self.df[self.df['Total Return'] >= top_10_percent]['Total Return'].mean()
        
        stability_score = avg_top_tier / top_return if top_return != 0 else 0
        
        return {
            'param_stability_score': stability_score,
            'top_tier_avg_return': avg_top_tier
        }
