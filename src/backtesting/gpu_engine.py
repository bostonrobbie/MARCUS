import numpy as np
from datetime import datetime
from .accelerate import get_dataframe_library, get_array_library, GPU_AVAILABLE

class GpuVectorEngine:
    """
    Dedicated GPU Execution Engine using CuPy.
    Performs fully vectorized backtests on GPU memory.
    """
    def __init__(self, strategy, initial_capital=100000.0, commission=1.0, slippage=0.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.xp = get_array_library() # Returns cupy if avail, else numpy
        if self.xp.__name__ != 'cupy':
            raise RuntimeError("GpuVectorEngine requires 'cupy' to be installed and active.")
        
    def run(self, df):
        """
        Runs the strategy on GPU.
        Expects df to constitute the Data.
        """
        # Ensure data is on GPU (cudf or cupy arrays)
        # Note: Strategy.generate_signals handles the logic
        
        # 1. Signals
        signals = self.strategy.generate_signals(df)
        
        # 2. Backtest Logic (Vectorized on GPU)
        # Convert necessary columns to cupy arrays if not already
        if hasattr(df, 'values'):
            # if cudf, .values is cupy-like? check implementation
            # usually cudf['col'].values is cupy array or can be converted
            col_name = 'close' if 'close' in df.columns else 'Close'
            if hasattr(df[col_name], 'to_cupy'):
                closes = df[col_name].to_cupy()
            else:
                closes = self.xp.asarray(df[col_name].values)
        else:
            # Series?
            closes = df['close'] if 'close' in df else df['Close']

        # Calculate Returns: (Price_t - Price_t-1) / Price_t-1
        # cupy diff
        price_diff = self.xp.diff(closes, prepend=closes[0])
        returns = price_diff / self.xp.roll(closes, 1)
        returns[0] = 0.0 # Fix first NaN/mess
        
        # Shift signals to align with execution (Next Open)
        # Signal at i acts on i+1 return
        pos = self.xp.roll(signals, 1)
        pos[0] = 0
        
        # Strategy Returns
        strat_returns = pos * returns
        
        # Costs
        # Turnover = abs(pos - pos_prev)
        turnover = self.xp.abs(self.xp.diff(pos, prepend=0))
        
        # Cost per unit approx? 
        # Simplified vector backtest often does bp per trade
        # Cost = turnover * (Comm + Slip) / Price ??
        # Let's approximate bps cost: (Comm + Slip) / AveragePrice ? 
        # Precise: (Comm + Slip) / Price[i]
        cost_pct = (self.commission + self.slippage) / closes
        costs = turnover * cost_pct
        
        net_returns = strat_returns - costs
        
        # Equity Curve
        cum_ret = self.xp.cumprod(1 + net_returns)
        equity = self.initial_capital * cum_ret
        
        # Move result to CPU for reporting if needed
        # But return GPU handles map
        
        return {
            'equity_curve': equity,
            'signals': signals,
            'returns': net_returns
        }

class GpuVectorStrategy:
    """Base class for strategies running on GPU"""
    def __init__(self, **kwargs):
        self.params = kwargs
        self.xp = get_array_library()
        if self.xp.__name__ != 'cupy':
            raise RuntimeError("GpuVectorStrategy requires 'cupy' to be installed and active.")
        
    def generate_signals(self, df):
        raise NotImplementedError
