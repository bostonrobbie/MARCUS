
from backtesting.strategies import NqOrb15m
from backtesting.schema import Bar
from datetime import datetime, timedelta
import pandas as pd
from queue import Queue

# Mock DataHandler
class MockData:
    def __init__(self):
        self.symbol_list = ['NQ=F']
        
    def get_latest_bars(self, symbol, N=1):
        # Return fake bars
        # Schema: symbol, timestamp, open, high, low, close, volume
        return [
            Bar('NQ=F', datetime.now(), 100, 110, 90, 105, 1000) 
            for _ in range(N)
        ]
        
    def get_latest_bar(self, symbol):
         return Bar('NQ=F', datetime.now(), 100, 110, 90, 105, 1000) 

def test_trade_frequency():
    dq = MockData()
    q = Queue()
    strat = NqOrb15m(dq, q)
    
    # Simulate a day
    date = datetime(2023, 1, 1, 9, 30)
    
    # 09:30 - 09:45 ORB Construction
    for i in range(16):
        t = date + timedelta(minutes=i)
        bar = Bar('NQ=F', t, 100, 105, 95, 102, 1000)
        strat.calculate_signals(bar)
    
    print(f"ORB High after session: {strat.orb_high}")
    
    # 09:46 - Trigger Entry
    t = date + timedelta(minutes=16)
    # Price > ORB High (105) -> 106
    bar = Bar('NQ=F', t, 106, 107, 106, 106, 1000)
    strat.calculate_signals(bar)
    
    # Check queue
    if not q.empty():
        sig = q.get()
        print(f"Signal 1: {sig.signal_type}")
    else:
        print("No Signal 1")
        
    # Check traded_today
    print(f"Traded Today: {strat.traded_today}")
    
    # 09:47 - Another Breakout
    t = date + timedelta(minutes=17)
    bar = Bar('NQ=F', t, 108, 109, 108, 108, 1000)
    strat.calculate_signals(bar)
    
    if not q.empty():
        print("Signal 2: Generated (FAIL)")
    else:
        print("Signal 2: None (PASS)")

if __name__ == "__main__":
    test_trade_frequency()
