import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(symbol='SPY', days=365):
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
    data = []
    price = 100.0
    for date in dates:
        change = np.random.normal(0, 1.0) # Daily volatility
        price += change
        if price < 10: price = 10
        open_p = price
        high = price + abs(np.random.normal(0, 0.5))
        low = price - abs(np.random.normal(0, 0.5))
        close = price + np.random.normal(0, 0.2)
        volume = int(np.random.normal(1000000, 200000))
        data.append([date, open_p, high, low, close, volume])
    
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.to_csv(f"{symbol}.csv", index=False)
    print(f"Generated {symbol}.csv with {len(df)} rows.")

if __name__ == "__main__":
    generate_data("SPY", 500)
    generate_data("AAPL", 500)
