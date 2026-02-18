
import sys
import os

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

import pandas as pd
import numpy as np
from backtesting.data import SmartDataHandler
from backtesting.optimizer import VectorizedGridSearch
from strategies.linreg_mr import LinRegMR
# Mock the Vectorized Engine for now using the Event Driven one if Vectorized doesn't support generic strategies yet
# Actually run_experiments uses VectorizedGridSearch which maps to Vectorized Engines.
# Since I haven't implemented a VectorizedLinReg engine, I must use the EventDriven engine for true validation.
# However, for 'run_experiments' style grid search, it expects vectorized.

# IMPLEMENTATION DETAIL:
# The user's system seems to support EventDriven (slow) and Vectorized (fast).
# I created an EventDriven strategy (strategies/linreg_mr.py).
# To run this efficiently in a grid search, I should ideally have a Vectorized implementation.
# BUT, for now, I will create a script that runs the EventDriven engine for a single pass or small grid 
# using the 'BacktestEngine' class directly, OR adapt the GridSearch to use the EventDriven engine (slower but works).

# Let's write a script that runs the EventDriven engine directly for validation.

from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

def run_backtest(symbol="NQ", start="2023-01-01", end="2023-12-31", params={}):
    print(f"Running Event-Driven Backtest for {symbol} [{start} to {end}]")
    print(f"Params: {params}")
    
    # 1. Setup Data
    # Assuming CSVs are in 'examples' or standard path
    csv_dir = os.path.join(os.getcwd(), 'examples')
    # Or try to find where data is. 'run_experiments.py' used 'examples' dir.
    # Let's use the same data handler setup
    
    interval = "15m"
    symbol_list = [symbol]
    
    events = queue.Queue()
    # Note: SmartDataHandler needs concrete paths or robust search.
    # Let's hope it finds the data in default locations or 'data' dir.
    # We will pass the current dir and 'data' just in case.
    search_dirs = [
        os.getcwd(), 
        os.path.join(os.getcwd(), 'data'), 
        os.path.join(os.getcwd(), 'examples'),
        os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    ]
    
    try:
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start, end_date=end, interval=interval)
    except Exception as e:
        print(f"Data Error: {e}")
        # Fallback to generating dummy data if needed? No, let's fail loud.
        return None

    instruments = {'NQ': {'multiplier': 20.0, 'commission': 2.05}} # Specs
    
    portfolio = Portfolio(data, events, initial_capital=100000.0, instruments=instruments)
    
    # Init Strategy with params
    strategy = LinRegMR(data, events, **params)
    
    execution = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
    
    engine = BacktestEngine(data, strategy, portfolio, execution)
    
    # Run
    engine.run()
    
    # Analyze
    tearsheet = TearSheet(portfolio)
    stats = tearsheet.analyze()
    
    print("\n--- Results ---")
    print(f"Total Return: {stats.get('Total Return', 'N/A')}")
    print(f"Sharpe: {stats.get('Sharpe Ratio', 'N/A')}")
    print(f"Max DD: {stats.get('Max Drawdown', 'N/A')}")
    print(f"Trades: {stats.get('Total Trades', 0)}")
    
    return stats

if __name__ == "__main__":
    # Test simple params
    params = {
        'length': 50, 
        'width': 2.5,     # Wider channel
        'adx_thresh': 20, # Stricter chop filter
        'verbose': False
    }
    # Run 10 Year Backtest
    run_backtest(symbol="NQ", start="2015-01-01", end="2024-12-31", params=params)
