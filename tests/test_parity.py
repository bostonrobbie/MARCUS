
import sys
import os
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('scripts'))

import runner # Import directly after adding scripts to path
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler

from backtesting.schema import Bar

# Define a mock strategy that buys on first bar and holds
class HoldStrategy(Strategy):
    def calculate_signals(self, event):
        if isinstance(event, Bar):
             latest = self.bars.get_latest_bar('benchmark_data')
             if latest:
                 self.buy('benchmark_data', 10)

def test_parity():
    # Use existing benchmark data
    data = SmartDataHandler(['benchmark_data'], ['.'], '2023-01-01', '2023-01-10', '1d')
    
    # Fake a "Vector Result" of 10%
    # In reality, holding 102 -> 106 is +3.9%
    # So we expect a DIFF of roughly 10% - 3.9% = 6% (Divergent)
    
    print("Testing Parity Logic...")
    res = runner.verify_parity(
        HoldStrategy, 
        {},           # params
        0.10,         # vector_return (fake high)
        data
    )
    
    print(f"Result: {res}")
    
    if res['is_divergent']:
        print("SUCCESS: Divergence correctly detected.")
    else:
        print("FAILURE: Divergence NOT detected.")

if __name__ == "__main__":
    test_parity()
