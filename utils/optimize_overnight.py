
import sys
import os
from datetime import time, timedelta

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

import pandas as pd
import numpy as np
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

class OvernightDataCollector(Strategy):
    def __init__(self, bars, events, verbose=False):
        super().__init__(bars, events)
        self.verbose = verbose
        self.entry_price = 0.0
        self.current_pos = 0 # 0, 1
        self.trade_log = []
        
        # Metrics State
        self.sma200 = None
        self.atr14 = None
        self.last_day = -1

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # Simple Indicators Calculation on the fly
        # We need roughly 200 bars (actually 200 * 24/5? No, SMA 200 on 5m is tiny)
        # We want Daily SMA 200.
        # Approximating Daily SMA 200 on 5m data is hard without a daily data loader.
        # Let's use 5m SMA matching roughly a Daily trend.
        # 1 Day ~ 78 bars (RTH) or 276 bars (23h). 
        # Let's use a long period EMA on 5m to proxy "Trend".
        # EMA 4000 (~15 days of 23h data) or EMA 1000.
        # Let's use EMA 2000 (roughly 1 week of data) as "Trend".
        
        start_time = time(9, 30)
        close_time = time(15, 55)
        
        # Iterative EMA Calculation
        # EMA 2000
        span = 2000
        alpha = 2 / (span + 1)
        
        current_price = event.close
        
        if self.sma200 is None:
             self.sma200 = current_price # Seed
        else:
             self.sma200 = (alpha * current_price) + ((1 - alpha) * self.sma200)
             
        ema_trend = self.sma200
        
        # Iterative ATR 14 (Approx)
        # Using a simple iterative smoothing of TR
        if self.atr14 is None:
            self.atr14 = event.high - event.low
        else:
            # Calculate TR
            # We need prev close for TR.
            # event.close is current close. We need prev bar close?
            # In live event loop, we don't have easy access to prev bar close without lookup.
            # But we can assume for "approx volatility" High-Low is decent proxy if gaps aren't huge intraday.
            # Or use bars lookup just for last 2 bars (fast).
            pass
            
        # Fast Lookup for TR
        bars = self.bars.get_latest_bars(symbol, N=2)
        if len(bars) < 2: return
        
        prev_close = bars[-2].close
        tr = max(event.high - event.low, abs(event.high - prev_close), abs(event.low - prev_close))
        
        # ATR EMA
        alpha_atr = 1/14
        self.atr14 = (alpha_atr * tr) + ((1 - alpha_atr) * self.atr14)
        atr = self.atr14
        
        # Store for trade logic
        self.sma200 = ema_trend
        self.atr14 = atr
        
        # Logic
        if current_time == close_time:
            # We ALWAYS take the trade in the simulation to log the result
            # We filter later in Pandas
            self.buy(symbol, 1)
            self.entry_price = event.close
            self.current_pos = 1
            
            # Log Trade Context
            self.current_trade_rec = {
                'entry_time': ts,
                'entry_price': event.close,
                'day_of_week': ts.dayofweek, # 0=Mon, 4=Fri
                'trend_ok': event.close > ema_trend,
                'atr': atr,
                'price_vs_ma_pct': (event.close - ema_trend)/ema_trend * 100
            }

        elif current_time == time(9, 30):
            if self.current_pos == 1:
                self.exit(symbol)
                self.current_pos = 0
                
                # Update Log with Outcome
                exit_price = event.open # Realistically we exit at open auction
                # In backtester, event.close is 9:30 close. 
                # We want gap capture. The fill usually happens effectively at Open.
                # Let's use event.open for accurate "Open" price.
                exit_price = event.open 
                
                pct_chg = (exit_price - self.entry_price) / self.entry_price
                
                self.current_trade_rec['exit_time'] = ts
                self.current_trade_rec['exit_price'] = exit_price
                self.current_trade_rec['return'] = pct_chg
                
                self.trade_log.append(self.current_trade_rec)

def run_optimization(start_date, end_date):
    print(f"Running Optimization Data Collection: {start_date} to {end_date}")
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd(), 'examples'), os.path.join(os.getcwd())]
    symbol_list = ['NQ']

    events = queue.Queue()
    data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port = Portfolio(data, events, initial_capital=100000.0)
    strat = OvernightDataCollector(data, events, verbose=False)
    exec_handler = SimulatedExecutionHandler(events, data, commission_model=FixedCommission(2.05))
    engine = BacktestEngine(data, strat, port, exec_handler)
    engine.run()
    
    # Analysis
    df = pd.DataFrame(strat.trade_log)
    print(f"\nCollected {len(df)} trades.")
    
    # 1. Day of Week Analysis
    print("\n--- Day of Week Performance ---")
    # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    days = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}
    for d in range(5):
        subset = df[df['day_of_week'] == d]
        ret = subset['return'].sum()
        win = len(subset[subset['return'] > 0])
        total = len(subset)
        print(f"{days[d]}: {win}/{total} ({win/total:.1%}) | Total Return: {ret:.2%}")
        
    # 2. Trend Filter Analysis
    print("\n--- Trend Filter (Price > EMA2000) ---")
    trend_ok = df[df['trend_ok'] == True]
    trend_bad = df[df['trend_ok'] == False]
    
    print(f"Trend UP:   {len(trend_ok)} trades | Return: {trend_ok['return'].sum():.2%} | Avg: {trend_ok['return'].mean():.4%}")
    print(f"Trend DOWN: {len(trend_bad)} trades | Return: {trend_bad['return'].sum():.2%} | Avg: {trend_bad['return'].mean():.4%}")

    # 3. Best Combo
    print("\n--- Optimal Combo Estimate ---")
    # Looking for: Trend UP AND (Maybe specific days)
    # Let's try Trend UP + Mon/Tue/Thu/Fri (Skip Wed?)
    best = df[(df['trend_ok'] == True)] # Start with Trend
    print(f"Filtered Strategy Return: {best['return'].sum():.2%} (Simulated)")

if __name__ == "__main__":
    run_optimization("2015-01-01", "2024-12-31")
