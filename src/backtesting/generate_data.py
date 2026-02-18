
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2022)
    args = parser.parse_args()

    start_date = datetime(args.start_year, 1, 1)
    # If end year is current year, cap at now? No, just full year.
    end_date = datetime(args.end_year, 12, 31)
    
    print(f"Generating Synthetic Data: {start_date} -> {end_date}")
    
    price = 4000.0 # Nasdaq roughly low in 2011 was 2000s, let's start decent.
    data = []
    
    current_time = start_date
    while current_time <= end_date:
        # Check if market open (9:30 - 16:00)
        is_market_hours = (current_time.time() >= time(9,30)) and (current_time.time() < time(16,0))
        
        # Overnight Gap (at 9:30)
        if current_time.time() == time(9,30):
             # 50% chance of gap up/down, magnitude ~0.5%
             gap_pct = np.random.normal(0, 0.005) 
             price = price * (1 + gap_pct)
        
        if is_market_hours:
            # Random Walk with Drift (Long term bullish)
            drift = 0.000005 
            change = np.random.normal(drift, 0.001) * price
            price += change
            
            # OHLC noise
            vol = price * 0.001 # 0.1% daily-ish vol
            open_ = price
            high = price + abs(np.random.normal(0, vol))
            low = price - abs(np.random.normal(0, vol))
            close = price + np.random.normal(0, vol/2)
            
            # Volume with spike at open/close
            vol_base = 1000
            if current_time.time() < time(10, 0) or current_time.time() > time(15, 30):
                vol_base = 5000
            volume = int(max(100, np.random.normal(vol_base, 200)))
            
            data.append([current_time, open_, high, low, close, volume])
            
        current_time += timedelta(minutes=5) # 5 Minute Data requested
            
    df = pd.DataFrame(data, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('Datetime', inplace=True)
    
    os.makedirs('backtesting/data', exist_ok=True)
    output_path = 'backtesting/data/NQ=F.csv'
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")
    print(df.tail())

if __name__ == "__main__":
    main()
