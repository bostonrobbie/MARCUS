import pandas as pd
import numpy as np
import sys
import os
import queue

# Ensure we can import the backtesting package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.strategy import Strategy
from backtesting.schema import Bar, SignalType
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet

class MovingAverageCrossover(Strategy):
    """
    A simple Moving Average Crossover strategy.
    Long when Fast MA > Slow MA.
    Short/Exit when Fast MA < Slow MA.
    """
    def __init__(self, bars, events, short_window=20, long_window=50):
        super().__init__(bars, events)
        self.short_window = short_window
        self.long_window = long_window
        self.bought = {} # Track if we are in a position

    def calculate_signals(self, event: Bar):
        symbol = event.symbol
        
        # We need enough bars
        bars = self.bars.get_latest_bars(symbol, N=self.long_window)
        if len(bars) < self.long_window:
            return

        # Create DataFrame for calculation (inefficient but clear)
        closes = pd.Series([b.close for b in bars])
        
        short_ma = closes.rolling(window=self.short_window).mean().iloc[-1]
        long_ma = closes.rolling(window=self.long_window).mean().iloc[-1]
        
        # Logic
        if short_ma > long_ma:
            if symbol not in self.bought or not self.bought[symbol]:
                # Entry Signal
                print(f"LONG Signal: {symbol} at {event.timestamp}")
                self.buy(symbol, quantity=100) # Simple fixed quantity
                self.bought[symbol] = True
        elif short_ma < long_ma:
             if symbol in self.bought and self.bought[symbol]:
                 # Exit Signal
                 print(f"EXIT Signal: {symbol} at {event.timestamp}")
                 self.exit(symbol) # Helper method from Strategy class
                 self.bought[symbol] = False

if __name__ == "__main__":
    # 1. Setup Data
    csv_dir = os.path.dirname(os.path.abspath(__file__)) # CSVs in same folder
    symbol_list = ['SPY']
    
    # 2. Shared Event Queue
    events = queue.Queue()
    
    # 3. Initialize Components
    # Pass csv_dir as a search directory
    data = SmartDataHandler(symbol_list, search_dirs=[csv_dir])
    portfolio = Portfolio(data, events, initial_capital=100000.0)
    strategy = MovingAverageCrossover(data, events, short_window=20, long_window=50)
    execution = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(commission_per_trade=1.0))
    
    # 4. Run Engine
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    
    # 5. Report
    tearsheet = TearSheet(portfolio)
    tearsheet.analyze()
