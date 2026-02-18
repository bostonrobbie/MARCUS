import pandas as pd
import numpy as np
from backtesting.vector_engine import VectorizedNQORB, VectorEngine
from backtesting.data import SmartDataHandler

# 1. Load Data
handler = SmartDataHandler(['NQ'], interval='15m')
symbol = 'NQ'
df = handler._fetch_data(symbol)

# Standardize as per data.py
df.columns = [c.capitalize() for c in df.columns]
date_col = next((c for c in df.columns if c in ['Date', 'Datetime', 'Time']), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None)
    df.set_index(date_col, inplace=True)
df.sort_index(inplace=True)

# 2. Run Strategy
params = {'stop_loss': 50, 'take_profit': 100, 'ema_filter': 50, 'atr_max_mult': 2.5}
strat = VectorizedNQORB(**params)
engine = VectorEngine(strat)

res = engine.run(df)

print("Signals count:", res['signals'].value_counts().to_dict())
print("Returns head:", res['returns'].head())
print("Equity final:", res['equity_curve'].iloc[-1])
print("NaNs in returns:", res['returns'].isna().sum())
print("NaNs in equity:", res['equity_curve'].isna().sum())

if res['equity_curve'].isna().any():
    print("First NaN in equity at:", res['equity_curve'][res['equity_curve'].isna()].index[0])
