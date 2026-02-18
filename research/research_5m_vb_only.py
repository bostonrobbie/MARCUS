
import sys
import os

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
from datetime import time

class VolBreakout5m(Strategy):
    def __init__(self, bars, events, sl_atr=1.5, tp_atr=3.0, verbose=False):
        super().__init__(bars, events)
        self.sl_atr_mult = sl_atr
        self.tp_atr_mult = tp_atr
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0 # 0, 1, -1
        
        # State necessary for Inside Bar
        self.prev_high = None
        self.prev_low = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # Session Filter
        if not (time(9, 35) <= current_time < time(15, 30)): 
            if self.current_pos != 0 and current_time >= time(15, 45):
                self.exit(symbol)
                self.current_pos = 0
            return
            
        # Get ATR
        bars = self.bars.get_latest_bars(symbol, N=300) # Increased lookback for trend EMA
        if len(bars) < 205: 
            self.prev_high = event.high
            self.prev_low = event.low
            return
            
        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
        atr = tr[-14:].mean()
        
        current_high = event.high
        current_low = event.low
        
        # Manage Existing Trade
        if self.current_pos != 0:
             if self.current_pos > 0:
                 sl = self.entry_price - (atr * self.sl_atr_mult)
                 tp = self.entry_price + (atr * self.tp_atr_mult)
                 if event.low < sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.high > tp:
                     self.exit(symbol)
                     self.current_pos = 0
             elif self.current_pos < 0:
                 sl = self.entry_price + (atr * self.sl_atr_mult)
                 tp = self.entry_price - (atr * self.tp_atr_mult)
                 if event.high > sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.low < tp:
                     self.exit(symbol)
                     self.current_pos = 0
                     
             self.prev_high = current_high
             self.prev_low = current_low
             return

        # Entry Logic: Inside Bar
        if self.prev_high is None:
            self.prev_high = current_high
            self.prev_low = current_low
            return

        bar_mother = bars[-3]
        bar_inside = bars[-2]
        bar_current = bars[-1] # The event bar
        
        is_inside = (bar_inside.high < bar_mother.high) and (bar_inside.low > bar_mother.low)
        
        # Trend Filter
        s = pd.Series(closes)
        ema_trend = s.ewm(span=200, adjust=False).mean().iloc[-1]
        
        if is_inside:
            # Trend Filtered Breakout
            if bar_current.close > bar_inside.high and bar_current.close > ema_trend:
                 self.buy(symbol, 1)
                 self.entry_price = bar_current.close
                 self.current_pos = 1
                 
            elif bar_current.close < bar_inside.low and bar_current.close < ema_trend:
                 self.sell(symbol, 1)
                 self.entry_price = bar_current.close
                 self.current_pos = -1

# Runner
def run_vb_backtest(start_date, end_date):
    print(f"Running VB Research: {start_date} to {end_date}")
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd(), 'examples'), os.path.join(os.getcwd())]
    symbol_list = ['NQ']

    events_2 = queue.Queue()
    data2 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port2 = Portfolio(data2, events_2, initial_capital=100000.0)
    strat2 = VolBreakout5m(data2, events_2, verbose=False)
    exec2 = SimulatedExecutionHandler(events_2, data2, commission_model=FixedCommission(2.05))
    engine2 = BacktestEngine(data2, strat2, port2, exec2)
    engine2.run()
    
    ts2 = TearSheet(port2)
    stats2 = ts2.analyze()
    print(f"VolBreakout Return: {stats2.get('Total Return', 0):.2%}")
    print(f"VolBreakout MaxDD:  {stats2.get('Max Drawdown', 0):.2%}")

if __name__ == "__main__":
    run_vb_backtest("2015-01-01", "2024-12-31")
