import pandas as pd
import queue
from typing import List, Type, Dict, Any
from .schema import Bar, SignalType
from .strategy import Strategy
from .data import DataHandler

class MockDataHandler(DataHandler):
    """
    A simple DataHandler for Unit Testing.
    Allows manually feeding bars for specific symbols.
    """
    def __init__(self, symbol_list: List[str]):
        self.symbol_list = symbol_list
        self.symbol_data = {symbol: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) for symbol in symbol_list}
        self.latest_bar_dict = {symbol: [] for symbol in symbol_list}

    def add_bar(self, symbol: str, open: float, high: float, low: float, close: float, volume: int = 1000, timestamp=None):
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        bar_data = {
            'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume
        }
        new_row = pd.DataFrame([bar_data], index=[timestamp])
        self.symbol_data[symbol] = pd.concat([self.symbol_data[symbol], new_row])
        
        bar = Bar(symbol, timestamp, open, high, low, close, volume)
        self.latest_bar_dict[symbol].append(bar)
        return bar

    def get_latest_bar(self, symbol: str) -> Bar:
        return self.latest_bar_dict[symbol][-1] if self.latest_bar_dict[symbol] else None

    def get_latest_bars(self, symbol: str, N: int = 1) -> List[Bar]:
        return self.latest_bar_dict[symbol][-N:]

class StrategyTester:
    """
    Unit testing utility for strategies.
    Verifies signal logic against specific price scenarios.
    """
    def __init__(self, strategy_cls: Type[Strategy], **strategy_params):
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params
        
    def assert_signal(self, scenario: List[Dict[str, float]], expected_signal: SignalType, symbol: str = "TEST"):
        """
        Feeds a scenario of bars to the strategy and checks the LAST signal generated.
        scenario: list of dicts with open, high, low, close
        """
        events = queue.Queue()
        data = MockDataHandler([symbol])
        strategy = self.strategy_cls(data, events, **self.strategy_params)
        
        # Feed all but the last bar for history
        for bar_info in scenario[:-1]:
            data.add_bar(symbol, **bar_info)
            # We don't call calculate_signals for history usually, 
            # as typical strategies look BACK from the current event.
            
        # Feed the trigger bar
        trigger_bar = data.add_bar(symbol, **scenario[-1])
        strategy.calculate_signals(trigger_bar)
        
        # Check queue
        signals = []
        while not events.empty():
            sig = events.get()
            if hasattr(sig, 'signal_type'):
                signals.append(sig)
        
        if not signals:
            if expected_signal is None:
                return True # Success, no signal expected
            raise AssertionError(f"Expected signal {expected_signal}, but NO signal was generated.")
        
        last_signal = signals[-1].signal_type
        if last_signal != expected_signal:
            raise AssertionError(f"Expected signal {expected_signal}, but got {last_signal}.")
            
        return True

# --- Example of how to use it ---
# if __name__ == "__main__":
#     from examples.trend_following import MovingAverageCrossover
#     tester = StrategyTester(MovingAverageCrossover, short_window=2, long_window=5)
#     
#     # Scenario: 5 bars of downtrend, then a cross
#     bars = [
#         {'open': 100, 'high': 105, 'low': 95, 'close': 98},
#         {'open': 98, 'high': 100, 'low': 90, 'close': 92},
#         {'open': 92, 'high': 95, 'low': 85, 'close': 88},
#         {'open': 88, 'high': 90, 'low': 80, 'close': 85},
#         {'open': 85, 'high': 95, 'low': 85, 'close': 94} # Trigger Long
#     ]
#     
#     try:
#         tester.assert_signal(bars, SignalType.LONG)
#         print("[PASS] MA Crossover Signal Test.")
#     except AssertionError as e:
#         print(f"[FAIL] {e}")
