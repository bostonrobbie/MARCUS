
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from backtesting.indicators import jit

def generate_data(n=10000):
    np.random.seed(42)
    closes = np.random.normal(100, 1, n).cumsum()
    highs = closes + np.random.random(n)
    lows = closes - np.random.random(n)
    opens = closes + np.random.random(n) - 0.5
    volumes = np.random.random(n) * 1000
    day_ints = np.zeros(n, dtype=np.int64)
    return opens, highs, lows, closes, volumes, day_ints

def test_indicators():
    print("Generating Data...")
    opens, highs, lows, closes, volumes, days = generate_data()
    
    print("Testing JIT Compilation & Execution...")
    start_t = time.time()
    
    # Trend
    print("SMA...", end="")
    jit.sma(closes, 20)
    print("OK")
    
    print("EMA...", end="")
    jit.ema(closes, 20)
    print("OK")
    
    print("TriMA...", end="")
    jit.trima(closes, 20)
    print("OK")
    
    # Momentum
    print("RSI...", end="")
    jit.rsi(closes, 14)
    print("OK")
    
    print("MACD...", end="")
    jit.macd(closes)
    print("OK")
    
    print("Stoch...", end="")
    jit.stochastic(highs, lows, closes)
    print("OK")
    
    print("TSI...", end="")
    jit.tsi(closes)
    print("OK")
    
    # Vol
    print("ATR...", end="")
    jit.atr(highs, lows, closes, 14)
    print("OK")
    
    print("BBands...", end="")
    jit.bbands(closes, 20, 2.0)
    print("OK")
    
    print("Keltner...", end="")
    jit.keltner(highs, lows, closes, 20, 2.0)
    print("OK")
    
    print("Donchian...", end="")
    jit.donchian(highs, lows, 20)
    print("OK")
    
    print("Choppiness...", end="")
    jit.choppiness(highs, lows, closes, 14)
    print("OK")
    
    # Regime
    print("ADX...", end="")
    jit.adx(highs, lows, closes, 14)
    print("OK")

    print("SuperTrend...", end="")
    jit.supertrend(highs, lows, closes, 10, 3.0)
    print("OK")
    
    print("MFI...", end="")
    jit.mfi(highs, lows, closes, volumes, 14)
    print("OK")
    
    print("RVol...", end="")
    jit.rvol(volumes, 20)
    print("OK")
    
    end_t = time.time()
    print(f"\nAll Tests Passed in {end_t - start_t:.4f} seconds.")

if __name__ == "__main__":
    test_indicators()
