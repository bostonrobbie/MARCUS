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

class GhostRider(Strategy):
    """
    Auto-generated strategy template: GhostRider
    Hypothesis: [Describe why this strategy should work]
    """
    def __init__(self, bars, events, param1=10, param2=20):
        super().__init__(bars, events)
        self.param1 = param1
        self.param2 = param2
        self.in_position = {} # Symbol -> bool

    def calculate_signals(self, event: Bar):
        """
        The core signal logic. Called for every new bar of data.
        """
        symbol = event.symbol
        
        # 1. Get enough history for indicators
        bars = self.bars.get_latest_bars(symbol, N=self.param2)
        if len(bars) < self.param2:
            return

        # 2. Extract price data
        closes = pd.Series([b.close for b in bars])
        
        # 3. Define Indicators (use pandas-ta for complex ones)
        # indicator = closes.rolling(window=self.param1).mean().iloc[-1]
        
        # 4. Entry/Exit Logic
        # if some_condition:
        #     self.buy(symbol, quantity=100)
        #     self.in_position[symbol] = True
        # elif other_condition:
        #     self.exit(symbol)
        #     self.in_position[symbol] = False
        
        pass

if __name__ == "__main__":
    # --- Quick Backtest Settings ---
    symbol_list = ['SPY']
    interval = '1d'
    
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Initialize Components
    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=[current_dir], interval=interval)
    portfolio = Portfolio(data, events, initial_capital=100000.0)
    strategy = GhostRider(data, events)
    execution = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(1.0))
    
    # 2. Run Engine
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    
    # 3. View Results
    tearsheet = TearSheet(portfolio)
    tearsheet.analyze()
