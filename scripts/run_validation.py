
import sys
import os
import queue

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

import pandas as pd
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
from backtesting.data import MemoryDataHandler
from strategies.es_orb_combo import EsOrbCombo

# MONKEY PATCH to fix missing symbol_list in MemoryDataHandler
def get_symbol_list(self):
    return list(self.symbol_data.keys())

MemoryDataHandler.symbol_list = property(get_symbol_list)

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'ES_5m.csv')

def run_validation():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load DF
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.capitalize() for c in df.columns] 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Ensure Columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in required):
        print(f"Error: Data missing columns. Found {df.columns}")
        return
        
    df.sort_index(inplace=True)

    print(f"Data loaded: {len(df)} bars. {df.index[0]} to {df.index[-1]}")
    
    events = queue.Queue()
    
    symbol_data = {"ES": df}
    bars = MemoryDataHandler(symbol_data)
    
    portfolio = Portfolio(bars, events, initial_capital=50000.0)
    
    strategy = EsOrbCombo(bars, events, verbose=True) # Verbose to see trades
    strategy.portfolio = portfolio 
    
    # Pass 'bars' to execution handler
    exec_handler = SimulatedExecutionHandler(events, bars)
    
    engine = BacktestEngine(bars, strategy, portfolio, exec_handler)
    
    print("Running backtest...")
    engine.run()
    
    print("\n--- Generating Report ---")
    ts = TearSheet(portfolio)
    stats = ts.analyze()
    print("\nComplete.")

if __name__ == "__main__":
    run_validation()
