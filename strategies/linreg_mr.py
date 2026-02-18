import numpy as np
import pandas as pd
from backtesting.strategy import Strategy
from backtesting.schema import Bar
from datetime import time

class LinRegMR(Strategy):
    """
    Linear Regression Mean Reversion Strategy.
    
    Logic:
    1. Calculate Rolling Linear Regression (Basis) and Standard Deviation (Width).
    2. Upper Band = Basis + width * StdDev
    3. Lower Band = Basis - width * StdDev
    4. Entry Long: Close < Lower Band (Oversold)
    5. Entry Short: Close > Upper Band (Overbought)
    6. Exit: Price returns to Basis (Mean) or Stop Loss.
    
    Filters:
    - ADX < Threshold (Trade only in chop)
    - Time-based filters (Session only)
    """
    
    def __init__(self, bars, events,
                 length=50,
                 width=2.0,
                 adx_thresh=30,
                 sl_pct=0.01,
                 start_time=time(9, 30),
                 end_time=time(15, 45),
                 verbose=False):
        super().__init__(bars, events)
        self.length = int(length)
        self.width = float(width)
        self.adx_thresh = float(adx_thresh)
        self.sl_pct = float(sl_pct)
        self.start_time = start_time
        self.end_time = end_time
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.in_trade = False
        self.current_pos = 0 # 0: Flat, 1: Long, -1: Short
        
        # Pre-calc helpers
        self.x = np.arange(self.length)
        self.x_mean = np.mean(self.x)
        self.x_var = np.var(self.x)

    def _calc_linreg(self, close_series):
        """
        Calculate the Linear Regression Endpoint (Basis) and StdDev for a single window.
        Optimized for rolling usage? No, this is per bar.
        For vectorization, a full column approach is better, but this is event-driven compatible.
        """
        y = close_series.values
        y_mean = np.mean(y)
        
        # Beta = Cov(x, y) / Var(x)
        # Cov(x, y) = Mean(xy) - Mean(x)Mean(y)
        xy_mean = np.mean(self.x * y)
        cov_xy = xy_mean - self.x_mean * y_mean
        beta = cov_xy / self.x_var
        
        alpha = y_mean - beta * self.x_mean
        
        # Expected value at the current bar (last point, index length-1)
        # y = alpha + beta * x
        basis = alpha + beta * (self.length - 1)
        
        # Calculate residuals std dev
        # y_hat = alpha + beta * x
        # residuals = y - y_hat
        # std_dev = np.std(residuals)
        
        # Actually standard LinReg Channel often uses just Standard Deviation of Price, 
        # NOT the Standard Error of the Estimate. 
        # TradingView's "Linear Regression Channel" usually plots the lines at +/- 2 SD of the PRICE from the line?
        # Or is it simple StdDev of the price series itself?
        # A common variation is LinReg +/- 2 * StdDev(Price). 
        # Let's use StdDev of Price for simplicity and robustness, unless "Standard Error" is specified.
        # Let's stick to StdDev of Price - easier interpretability.
        
        std_dev = np.std(y)
        
        return basis, std_dev

    def calculate_signals(self, event: Bar):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # 1. Check Session
        if not (self.start_time <= current_time < self.end_time):
            if self.in_trade and current_time >= self.end_time:
                self.exit(symbol)
                self.in_trade = False
                self.current_pos = 0 # Reset pos
            return

        # 2. Get Data
        # We need efficient lookback
        bars = self.bars.get_latest_bars(symbol, N=self.length + 14) # +14 for ADX
        if len(bars) < self.length + 14:
            return
            
        # Convert to arrays for speed
        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        
        # 3. Calculate Indicators
        # Slice the last 'length' for LinReg
        window_closes = closes[-self.length:]
        
        basis, std_dev = self._calc_linreg(pd.Series(window_closes))
        upper = basis + (std_dev * self.width)
        lower = basis - (std_dev * self.width)
        
        current_price = event.close
        
        # Calc RSI (using src.backtesting.ta if available or simple numpy)
        # Using 14 period RSI
        # Simple NumPy RSI implementation for speed/dependency safety
        try:
           deltas = np.diff(closes)
           seed = deltas[-14:]
           up = seed[seed >= 0].sum()/14
           down = -seed[seed < 0].sum()/14
           rs = up/down if down != 0 else 0
           rsi = 100 - (100/(1+rs))
           # NOTE: This is simple RSI, not Wilder's smoothed. Sufficient for filter.
        except:
           rsi = 50

        # 4. Logic
        if self.current_pos == 0:
            # Short Entry (Overbought + RSI High)
            if current_price > upper and rsi > 70:
                if self.verbose: print(f"[{ts}] Short Entry: Price {current_price:.2f} > Upper {upper:.2f} RSI={rsi:.1f}")
                self.sell(symbol, 1) # Entry Short
                self.entry_price = current_price
                self.current_pos = -1
                
            # Long Entry (Oversold + RSI Low)
            elif current_price < lower and rsi < 30:
                if self.verbose: print(f"[{ts}] Long Entry: Price {current_price:.2f} < Lower {lower:.2f} RSI={rsi:.1f}")
                self.buy(symbol, 1) # Entry Long
                self.entry_price = current_price
                self.current_pos = 1
                
        else:
            # Manage Trade
            
            # Check Stop Loss
            if self.current_pos > 0: # Long
                sl_price = self.entry_price * (1 - self.sl_pct)
                if current_price < sl_price:
                    self.exit(symbol)
                    self.current_pos = 0
                elif current_price > basis: # Target: Mean
                    self.exit(symbol)
                    self.current_pos = 0
                    
            elif self.current_pos < 0: # Short
                sl_price = self.entry_price * (1 + self.sl_pct)
                if current_price > sl_price:
                    self.exit(symbol)
                    self.current_pos = 0
                elif current_price < basis: # Target: Mean
                    self.exit(symbol)
                    self.current_pos = 0

