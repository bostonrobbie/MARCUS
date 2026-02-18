
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from backtesting.strategies.short_logic import run_short_alpha_collection

def generate_data(n=10000):
    np.random.seed(42)
    closes = np.random.normal(100, 1, n).cumsum()
    highs = closes + np.random.random(n)
    lows = closes - np.random.random(n)
    opens = closes + np.random.random(n) - 0.5
    volumes = np.random.random(n) * 1000
    
    # Create Days
    day_ints = np.zeros(n, dtype=np.int64)
    for i in range(n):
        day_ints[i] = i // 78 # 78 5m bars per day approx
        
    time_ints = np.zeros(n, dtype=np.int64)
    # 930 to 1600
    for i in range(n):
        bar_idx = i % 78
        time_ints[i] = 930 + (bar_idx * 5)
        # Fix Hour rollover (60m)
        minutes = 30 + (bar_idx * 5)
        hour = 9 + (minutes // 60)
        mins = minutes % 60
        time_ints[i] = hour * 100 + mins
        
    return opens, highs, lows, closes, volumes, day_ints, time_ints

def test_engine():
    print("Generating Data...")
    opens, highs, lows, closes, volumes, days, times = generate_data()
    
    print("Testing JIT Strategy Engine...")
    start_t = time.time()
    
    # Test Strategy 2 (Parabolic)
    print("Testing Strat 2...", end="")
    run_short_alpha_collection(
        2, opens, highs, lows, closes, volumes, days, times,
        0.005, 1.0, 
        0, 100, 0.01, 0, 0, 0, 0, 0, 0, 0.0, 0.0
    )
    print("OK")
    
    # Test Strategy 4 (Bear Flag)
    print("Testing Strat 4...", end="")
    run_short_alpha_collection(
        4, opens, highs, lows, closes, volumes, days, times,
        0.005, 1.0, 
        0, 0, 0, 0, 100, 50, 0, 0, 0, 0.0, 0.0
    )
    print("OK")
    
    # Test Strategy 9 (RSI-2)
    print("Testing Strat 9...", end="")
    run_short_alpha_collection(
        9, opens, highs, lows, closes, volumes, days, times,
        0.005, 1.0, 
        0, 0, 0, 0, 0, 90, 0, 0, 0, 0.0, 0.0
    )
    print("OK")
    
    end_t = time.time()
    print(f"\nEngine Verified in {end_t - start_t:.4f} seconds.")

if __name__ == "__main__":
    test_engine()
