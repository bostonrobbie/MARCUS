import pandas as pd
import numpy as np
from backtesting.vector_engine import VectorizedNQORB, VectorEngine
from examples.nqorb_15m import NqOrb15m
from backtesting.data import SmartDataHandler
import queue

# 1. Setup
symbol = 'NQ'
interval = '15m'
params = {'stop_loss': 50, 'take_profit': 100, 'ema_filter': 50, 'atr_max_mult': 2.5}

# 2. Get Data
handler = SmartDataHandler([symbol], interval=interval)
df_vector = handler.symbol_data[symbol]

# 3. Vector Signals
v_strat = VectorizedNQORB(**params)
v_signals = v_strat.generate_signals(df_vector)

# 4. Event Signals (Manual check of a few days)
# We'll just run a short backtest and capture signals if possible
# Or manually inspect a few logical points
print("Vector Signals Value Counts:\n", v_signals.value_counts())

# Check a slice where signals occur
signal_dates = v_signals[v_signals != 0].index.unique()
if len(signal_dates) > 0:
    first_sig_date = signal_dates[0]
    print(f"\nFirst signal at: {first_sig_date}")
    # Show context around first signal
    start_idx = df_vector.index.get_loc(first_sig_date)
    print(df_vector.iloc[start_idx-2:start_idx+2][['Open', 'High', 'Low', 'Close']])
    print("Signal:", v_signals.iloc[start_idx])
