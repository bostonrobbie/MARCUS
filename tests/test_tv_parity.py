
import sys
import os
import pandas as pd
from queue import Queue
from datetime import datetime

sys.path.insert(0, os.path.abspath('src'))

from backtesting.execution import SimulatedExecutionHandler
from backtesting.schema import Bar, OrderEvent, OrderType, OrderSide
from backtesting.validation import MonteCarloValidator

class MockData:
    symbol = 'TEST'

def test_tv_emulator():
    print("\nTesting TV Emulator (Intrabar Logic)...")
    # Scenario: Bar Low=100, High=110. Limit Buy at 105.
    # Standard: Fills next bar.
    # TV Mode: Should fill THIS bar.
    
    events = Queue()
    handler = SimulatedExecutionHandler(events, None, mode='TV_BROKER_EMULATOR')
    
    # Create Order
    order = OrderEvent('TEST', datetime.now(), 10, OrderSide.BUY, OrderType.LIMIT, limit_price=105.0)
    handler.pending_orders.append(order)
    
    # Create Bar (Low 100 <= 105 <= High 110)
    bar = Bar('TEST', datetime.now(), 108, 110, 100, 102, 1000)
    
    handler.on_bar(bar)
    
    if not events.empty():
        fill = events.get()
        print(f"[PASS] TV Mode Filled Intrabar! Price: {fill.price}")
    else:
        print("[FAIL] TV Mode did NOT fill Intrabar.")

def test_monte_carlo():
    print("\nTesting Monte Carlo...")
    # Mock Trade Log
    trades = [{'return': 0.05}, {'return': -0.02}, {'return': 0.10}, {'return': -0.05}] * 10
    mc = MonteCarloValidator(trades, n_sims=100)
    res = mc.run()
    if res:
        print("[PASS] Monte Carlo generated stats.")

if __name__ == "__main__":
    test_tv_emulator()
    test_monte_carlo()
