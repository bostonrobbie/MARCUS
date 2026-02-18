
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.backtesting.data import SmartDataHandler

def check_range():
    symbols = ['NQ', 'ES']
    search_dirs = [
        './data', 
        './examples', 
        r'C:\Users\User\Desktop\Portfolio\OHLC\Intra OHLC', 
        r'C:\Users\User\Documents\AI\StrategyPipeline\data'
    ]
    
    # Check broad range to catch everything
    handler = SmartDataHandler(
        symbol_list=symbols,
        search_dirs=search_dirs,
        start_date=pd.Timestamp('2000-01-01'),
        end_date=pd.Timestamp('2030-01-01'),
        interval='5m'
    )
    
    for sym in symbols:
        df = handler.symbol_data.get(sym)
        if df is not None and not df.empty:
            start = df.index[0]
            end = df.index[-1]
            rows = len(df)
            print(f"SYMBOL: {sym}")
            print(f"  Start: {start}")
            print(f"  End:   {end}")
            print(f"  Rows:  {rows}")
            print(f"  Years: {(end - start).days / 365.25:.1f}")
        else:
            print(f"SYMBOL: {sym} -> NO DATA FOUND")

if __name__ == "__main__":
    check_range()
