
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

# ==========================================
# 1. Bollinger Band Fader (Regime Filtered)
# ==========================================
class BollingerFader5m(Strategy):
    def __init__(self, bars, events, bb_len=20, bb_std=2.0, adx_max=25, verbose=False):
        super().__init__(bars, events)
        self.bb_len = bb_len
        self.bb_std = bb_std
        self.adx_max = adx_max
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0 # 0, 1, -1

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # Session (9:30 - 15:45)
        if not (time(9, 30) <= current_time < time(15, 45)):
            if self.current_pos != 0 and current_time >= time(15, 55):
                self.exit(symbol)
                self.current_pos = 0
            return

        # Data lookback
        lookback = max(self.bb_len, 20) + 14 + 5 # +14 for ADX
        bars = self.bars.get_latest_bars(symbol, N=lookback)
        if len(bars) < lookback: return
        
        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        
        # --- Indicators ---
        # 1. Bollinger Bands
        s = pd.Series(closes)
        sma = s.rolling(self.bb_len).mean().iloc[-1]
        std = s.rolling(self.bb_len).std().iloc[-1]
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        
        # 2. ADX (Approx)
        try:
           # Simple DX/ADX calc
           deltas_up = highs[1:] - highs[:-1]
           deltas_down = lows[:-1] - lows[1:]
           
           pos_dm = np.where((deltas_up > deltas_down) & (deltas_up > 0), deltas_up, 0)
           neg_dm = np.where((deltas_down > deltas_up) & (deltas_down > 0), deltas_down, 0)
           
           tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
           
           # Smooth (Wilder's) - simplified to simple mean for speed in validation
           atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
           p_dm_s = pd.Series(pos_dm).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
           n_dm_s = pd.Series(neg_dm).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
           
           p_di = 100 * (p_dm_s / atr)
           n_di = 100 * (n_dm_s / atr)
           dx = 100 * abs(p_di - n_di) / (p_di + n_di)
           # ADX is smoothed DX
           # We will just use DX for strict chop filter (instantaneous) or assume adx ~ dx for last bar
           adx = dx 
        except:
           adx = 50 # Fail safe = no trade

        current_price = event.close
        
        # Logic
        if self.current_pos == 0:
            # Entry: Price Outside Bands AND Low ADX (Chop)
            if adx < self.adx_max:
                if current_price < lower:
                    # Fade the Low
                    if self.verbose: print(f"[{ts}] BB Long: {current_price:.2f} < {lower:.2f} (ADX {adx:.1f})")
                    self.buy(symbol, 1)
                    self.entry_price = current_price
                    self.current_pos = 1
                elif current_price > upper:
                    # Fade the High
                    if self.verbose: print(f"[{ts}] BB Short: {current_price:.2f} > {upper:.2f} (ADX {adx:.1f})")
                    self.sell(symbol, 1)
                    self.entry_price = current_price
                    self.current_pos = -1
                    
        else:
            # Exits: Mean Reversion (Basis)
            if self.current_pos > 0:
                # Target: SMA
                if current_price >= sma:
                    self.exit(symbol)
                    self.current_pos = 0
                # Stop: Hard fixed % (Emergency) or expansion
                elif current_price < self.entry_price * 0.99:
                    self.exit(symbol)
                    self.current_pos = 0
                    
            elif self.current_pos < 0:
                if current_price <= sma:
                    self.exit(symbol)
                    self.current_pos = 0
                elif current_price > self.entry_price * 1.01:
                    self.exit(symbol)
                    self.current_pos = 0

# ==========================================
# 2. RSI-2 Mean Reversion (Connors)
# ==========================================
class RSI2Reversion5m(Strategy):
    def __init__(self, bars, events, adx_max=30, verbose=False):
        super().__init__(bars, events)
        self.adx_max = adx_max
        self.verbose = verbose
        self.current_pos = 0
        self.entry_price = 0.0

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        if not (time(9, 30) <= current_time < time(15, 45)):
            if self.current_pos != 0 and current_time >= time(15, 55):
                self.exit(symbol)
                self.current_pos = 0
            return
            
        bars = self.bars.get_latest_bars(symbol, N=200)
        if len(bars) < 200: return
        
        closes = np.array([b.close for b in bars])
        
        # RSI 2
        delta = np.diff(closes)
        # We need a proper RSI calc for period 2
        # Vectorized for last few bars
        def calc_rsi(prices, n=2):
            deltas = np.diff(prices)
            seed = deltas[-n:] # Simplistic check
            up = seed[seed>=0].sum()
            down = -seed[seed<0].sum()
            if down == 0: return 100
            rs = up/down
            return 100 - (100/(1+rs))
            
        # Better: Rolling function? 
        # Easy manual:
        gains = np.maximum(delta, 0)
        losses = np.abs(np.minimum(delta, 0))
        # Wilder's Smoothing for RSI? Or Simple? Connors uses Simple often or Wilder.
        # Let's use Pandas for accuracy
        s = pd.Series(closes)
        # RSI 2
        # rsi = ta.rsi(s, 2)
        # Manual EWM
        avg_gain = pd.Series(gains).ewm(alpha=1/2, min_periods=2).mean() # RSI 2 is aggressive
        avg_loss = pd.Series(losses).ewm(alpha=1/2, min_periods=2).mean()
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        rsi = 100 - (100/(1+rs))
        
        # SMA 5 (Exit)
        sma5 = s.rolling(5).mean().iloc[-1]
        # SMA 200 (optional filter, Connors uses it. Let's ignore for Chop focus, or use ADX)
        
        # ADX (Reuse logic or assume passed)
        # Let's use a simple volatility filter: Range < Average Range?
        # Or simple ADX.
        # Let's use ADX proxy:
        tr = np.max(np.abs(np.diff(closes)[-14:])) # Crude volatility check
        
        # Logic
        if self.current_pos == 0:
            # Long: RSI < 5 (Oversold)
            if rsi < 5:
                # Chop Filter?
                self.buy(symbol, 1)
                self.entry_price = event.close
                self.current_pos = 1
            # Short: RSI > 95 (Overbought)
            elif rsi > 95:
                self.sell(symbol, 1)
                self.entry_price = event.close
                self.current_pos = -1
        else:
            # Exit Long: Price > SMA 5
            if self.current_pos > 0:
                if event.close > sma5:
                    self.exit(symbol)
                    self.current_pos = 0
            # Exit Short: Price < SMA 5
            elif self.current_pos < 0:
                if event.close < sma5:
                    self.exit(symbol)
                    self.current_pos = 0

# ==========================================
# Runner
# ==========================================
def run_chop_research(start_date, end_date):
    print(f"Running Chop Strategy Research (5m): {start_date} to {end_date}")
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [csv_dir, os.path.join(os.getcwd(), 'examples'), os.path.join(os.getcwd())]
    symbol_list = ['NQ']

    # --- Strat 1: BB Fader ---
    print("\n>>> Testing BB Fader (5m)...")
    events_1 = queue.Queue()
    data1 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port1 = Portfolio(data1, events_1, initial_capital=100000.0)
    strat1 = BollingerFader5m(data1, events_1, verbose=False)
    exec1 = SimulatedExecutionHandler(events_1, data1, commission_model=FixedCommission(2.05))
    engine1 = BacktestEngine(data1, strat1, port1, exec1)
    engine1.run()
    
    ts1 = TearSheet(port1)
    stats1 = ts1.analyze()
    print(f"BB Fader Return: {stats1.get('Total Return', 0):.2%}")
    print(f"BB Fader MaxDD:  {stats1.get('Max Drawdown', 0):.2%}")
    
    # --- Strat 2: RSI 2 ---
    print("\n>>> Testing RSI-2 (5m)...")
    events_2 = queue.Queue()
    data2 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port2 = Portfolio(data2, events_2, initial_capital=100000.0)
    strat2 = RSI2Reversion5m(data2, events_2, verbose=False)
    exec2 = SimulatedExecutionHandler(events_2, data2, commission_model=FixedCommission(2.05))
    engine2 = BacktestEngine(data2, strat2, port2, exec2)
    engine2.run()
    
    ts2 = TearSheet(port2)
    stats2 = ts2.analyze()
    print(f"RSI-2 Return: {stats2.get('Total Return', 0):.2%}")
    print(f"RSI-2 MaxDD:  {stats2.get('Max Drawdown', 0):.2%}")

if __name__ == "__main__":
    run_chop_research("2015-01-01", "2024-12-31")
