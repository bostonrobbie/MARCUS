import os
import sys

STRATEGY_TEMPLATE = '''import pandas as pd
import numpy as np
import sys
import os
import queue

# Ensure we can import the backtesting package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .strategy import Strategy
from .schema import Bar, SignalType
from .data import SmartDataHandler
from .portfolio import Portfolio
from .execution import SimulatedExecutionHandler, FixedCommission
from .engine import BacktestEngine
from .performance import TearSheet

class {strategy_name}(Strategy):
    """
    Auto-generated strategy template: {strategy_name}
    Hypothesis: [Describe why this strategy should work]
    """
    def __init__(self, bars, events, param1=10, param2=20):
        super().__init__(bars, events)
        self.param1 = param1
        self.param2 = param2
        self.in_position = {{}} # Symbol -> bool

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
    strategy = {strategy_name}(data, events)
    execution = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(1.0))
    
    # 2. Run Engine
    engine = BacktestEngine(data, strategy, portfolio, execution)
    engine.run()
    
    # 3. View Results
    tearsheet = TearSheet(portfolio)
    tearsheet.analyze()
'''

def generate_strategy(name: str, target_dir: str = "examples"):
    """
    Generates a new strategy file based on the professional template.
    """
    # Ensure name is CamelCase
    if "_" in name:
        name = "".join(x.capitalize() for x in name.split("_"))
    
    filename = f"{name.lower()}.py"
    filepath = os.path.join(target_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[ERROR] Strategy {filename} already exists in {target_dir}.")
        return False
        
    os.makedirs(target_dir, exist_ok=True)
    
    content = STRATEGY_TEMPLATE.format(strategy_name=name)
    
    with open(filepath, "w") as f:
        f.write(content)
        
    print(f"[SUCCESS] Professional Strategy scaffolded at: {filepath}")
    print(f"To run it: python runner.py --strategy {name.lower()} --symbol SPY")
    return True
