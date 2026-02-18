import pandas as pd
import numpy as np
import sys
import os

# Path hack
sys.path.append(os.getcwd())
from backtesting.regime import RegimeFilter, Regime

def test_regime_classification():
    print("Test 1: Bull Trend Generation")
    # Generate 100 bars of uptrend
    idx = pd.date_range("2023-01-01", periods=200, freq="15min")
    close = np.linspace(100, 200, 200) # Strong uptrend
    high = close + 1
    low = close - 1
    # Add noise
    close += np.random.normal(0, 0.5, 200)
    
    df = pd.DataFrame({'close': close, 'high': high, 'low': low}, index=idx)
    
    # Run Filter
    f = RegimeFilter(sma_period=50) # Need 50 bars warmup
    regimes = f.label_regime(df)
    
    # Check last bar (should be BULL)
    last_regime = regimes.iloc[-1]
    print(f"  Last Regime: {last_regime}")
    
    if last_regime == Regime.BULL_TREND:
        print("  [PASS] Bull Trend Detected")
    else:
        print(f"  [FAIL] Expected BULL, got {last_regime}")
        
    print("\nTest 2: Bear Trend Generation")
    # Generate Downtrend
    close = np.linspace(200, 100, 200)
    df = pd.DataFrame({'close': close, 'high': close+1, 'low': close-1}, index=idx)
    regimes = f.label_regime(df)
    last_regime = regimes.iloc[-1]
    print(f"  Last Regime: {last_regime}")
    
    if last_regime == Regime.BEAR_TREND:
        print("  [PASS] Bear Trend Detected")
    else:
        print(f"  [FAIL] Expected BEAR, got {last_regime}")

    print("\nTest 3: Chop Generation")
    # Generate Pure Noise (Random Walk around mean) for Chop
    close = np.random.normal(150, 2, 200) # Flat mean 150, high noise
    df = pd.DataFrame({'close': close, 'high': close+1, 'low': close-1}, index=idx)
    regimes = f.label_regime(df)
    last_regime = regimes.iloc[-1]
    print(f"  Last Regime: {last_regime}")
    
    # Chop might be tricky depending on ADX lag, but sine wave usually kills ADX
    if last_regime in [Regime.CHOPPY_QUIET, Regime.CHOPPY_VOLATILE]:
         print("  [PASS] Chop Detected")
    else:
         print(f"  [FAIL] Expected CHOP, got {last_regime}")

if __name__ == "__main__":
    test_regime_classification()
