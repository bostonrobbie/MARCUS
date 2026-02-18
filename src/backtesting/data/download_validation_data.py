
import yfinance as yf
import pandas as pd
import os
import sys

# Ensure data dir exists
DATA_DIR = r"C:\Users\User\Documents\AI\StrategyPipeline\data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_save(ticker, filename):
    print(f"Downloading {ticker}...")
    # interval='5m', period='60d' is the max for yfinance
    try:
        df = yf.download(ticker, interval='5m', period='60d', progress=False)
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return False
            
        # Flatten MultiIndex columns if present (common in newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Reset index to get Datetime as a column
        df = df.reset_index()
        
        # Renaissance of formatting: Standardize columns
        # yfinance often returns 'Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        # We need specific casing for backtesting
        
        # Rename 'Datetime' or 'Date' to 'Date'
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        # Save
        path = os.path.join(DATA_DIR, filename)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
        return True
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return False

if __name__ == "__main__":
    # ES=F is E-mini S&P 500 Futures
    # ^VIX is CBOE Volatility Index
    success_es = download_and_save("ES=F", "ES_5m.csv")
    success_vix = download_and_save("^VIX", "VIX_5m.csv")
    
    if success_es and success_vix:
        print("Data download complete.")
    else:
        print("Data download failed.")
        sys.exit(1)
