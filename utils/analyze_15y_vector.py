import sys
import os
import pandas as pd
import pandas_ta as ta
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from examples.nqorb_enhanced import NqOrbEnhanced
from backtesting.vector_engine import VectorizedNQORB, VectorEngine
from backtesting.data import SmartDataHandler

def analyze_15y():
    print("Loading 15m Data (2010-2025)...")
    data_dir = "C:\\Users\\User\\Desktop\\Portfolio\\OHLC\\Intra OHLC"
    symbol = "NQ"
    
    # Load Data
    loader = SmartDataHandler([symbol], search_dirs=[data_dir], interval='15m', start_date="2010-01-01", end_date="2025-05-30")
    df = loader.symbol_data[symbol]
    
    print(f"Data Loaded: {len(df)} bars")
    print("Running Vectorized Strategy...")
    
    # Initialize Strategy with Certified Parameters
    # Note: Vector engine might handle some things differently, but we approximate best matching params
    strategy = VectorizedNQORB(
        sl_atr_mult=2.0, 
        tp_atr_mult=4.0, 
        ema_filter=50, 
        atr_max_mult=2.5,
        use_htf=True,
        use_rvol=True,
        # Vector engine typically optimistically assumes TP/SL fills
    )
    
    engine = VectorEngine(strategy)
    results = engine.run(df)
    
    equity = results['equity_curve']
    
    print("\n" + "="*50)
    print("15-YEAR PERFORMANCE ANALYSIS (2010-2025)")
    print("="*50)
    
    # Overall Stats
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    daily_rets = equity.pct_change().dropna()
    sharpe = np.sqrt(252*96) * daily_rets.mean() / daily_rets.std() # 96 bars per day? No, vector engine usually effectively daily or per-bar.
    # Actually VectorEngine returns equity series indexed by bar time.
    # 15m data -> ~23 bars per session? 
    # Let's resample to daily for standard Sharpe
    daily_eq = equity.resample('D').last().dropna()
    daily_r = daily_eq.pct_change().dropna()
    sharpe_daily = np.sqrt(252) * daily_r.mean() / daily_r.std()
    
    # Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()
    
    print(f"Total Return: {total_ret:.2%}")
    print(f"CAGR:         {( (equity.iloc[-1]/equity.iloc[0])**(1/15) - 1 ):.2%}")
    print(f"Sharpe (D):   {sharpe_daily:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")
    print("-" * 50)
    print("ANNUAL BREAKDOWN")
    print(f"{'Year':<6} | {'Return':<10} | {'Drawdown':<10}")
    print("-" * 30)
    
    # Annual Stats
    years = equity.groupby(equity.index.year)
    for year, y_eq in years:
        if len(y_eq) < 100: continue
        start_eq = y_eq.iloc[0]
        end_eq = y_eq.iloc[-1]
        y_ret = (end_eq / start_eq) - 1
        
        y_peak = y_eq.cummax()
        y_dd = (y_eq - y_peak) / y_peak
        y_mdd = y_dd.min()
        
        print(f"{year:<6} | {y_ret:>9.2%} | {y_mdd:>9.2%}")
        
    print("="*50)
    
    # Standardized Report
    try:
        from reporting_utils import save_standard_equity_curve
        save_standard_equity_curve(equity, 'NQ ORB Enhanced - 15 Year Vector', 'NQORB_15Y_Standard.png')
    except ImportError as e:
        print(f'Error importing cleanup lib: {e}')

if __name__ == "__main__":
    analyze_15y()

