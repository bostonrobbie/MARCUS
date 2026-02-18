
import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta
import sys
import os
import pytz

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.strategy import Strategy
from backtesting.schema import Bar

class EsOrbCombo(Strategy):
    def __init__(self, bars, events, 
                 or_minutes=15,
                 rvol_lookback=20,
                 rvol_min_follow=0.0, # DISABLED for validation
                 hurst_len=100,
                 hurst_min_follow=0.0, # DISABLED for validation
                 verbose=False):
        super().__init__(bars, events)
        
        self.or_minutes = or_minutes
        self.rvol_lookback = rvol_lookback
        self.rvol_min_follow = rvol_min_follow
        self.hurst_len = hurst_len
        self.hurst_min_follow = hurst_min_follow
        self.verbose = verbose
        
        self.SL_ATR_F = 2.0
        self.TP_ATR_F = 10.0
        
        self.last_date = None
        self.or_high = -999999
        self.or_low = 999999
        self.or_locked = False
        self.or_vol_accum = 0
        self.mode_today = None
        self.atr_today = 5.0
        self.daily_trade_count = 0
        
        self.stop_loss = 0
        self.take_profit = 0
        
        self.or_vol_history = [] 
        self.portfolio = None

    def _get_history(self, symbol, length):
        return self.bars.get_latest_bars(symbol, length)

    def _calculate_indicators(self, symbol):
        lookback = 200
        history = self._get_history(symbol, lookback)
        if len(history) < lookback: return 0.5, 5.0
        
        closes = np.array([b.close for b in history])
        highs = np.array([b.high for b in history])
        lows = np.array([b.low for b in history])
        
        # --- Hurst (Pine Parallel) ---
        series_window = closes[-(self.hurst_len+1):]
        if len(series_window) < self.hurst_len + 1:
            hurst = 0.5
        else:
            ret_series = np.log(series_window[1:] / series_window[:-1])
            mean = np.mean(ret_series)
            dev = ret_series - mean
            cum_dev = np.concatenate(([0], np.cumsum(dev)))
            r_val = np.max(cum_dev) - np.min(cum_dev)
            ssd = np.sum(dev * dev)
            s_val = np.sqrt(ssd / self.hurst_len)
            
            if s_val > 0 and r_val > 0:
                hurst = np.log(r_val / s_val) / np.log(self.hurst_len)
            else:
                hurst = 0.5

        # --- ATR (150) ---
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr[-150:])
        
        return hurst, atr

    def _classify_day(self, event):
        if len(self.or_vol_history) < 5:
            rvol = 1.0 
        else:
            avg_vol = np.mean(self.or_vol_history[-self.rvol_lookback:])
            rvol = self.or_vol_accum / avg_vol if avg_vol > 0 else 1.0
        
        self.or_vol_history.append(self.or_vol_accum)
        
        hurst, atr = self._calculate_indicators(event.symbol)
        self.atr_today = atr
        
        # Logic - DISABLED FILTERS
        follow = rvol >= self.rvol_min_follow and hurst >= self.hurst_min_follow
        self.mode_today = 'follow' if follow else 'fade'
        
        if self.verbose:
            print(f"[{event.timestamp}] OR: {self.or_high:.2f}-{self.or_low:.2f} | RVOL={rvol:.2f} Hurst={hurst:.2f} -> {self.mode_today}")

    def calculate_signals(self, event: Bar):
        symbol = event.symbol
        dt_utc = event.timestamp
        
        # Convert to Eastern Time for Logic
        # assuming dt_utc is timezone aware if loading with +00:00
        # If not aware, localize?
        if dt_utc.tzinfo is None:
            dt_utc = pytz.utc.localize(dt_utc)
        
        dt_et = dt_utc.astimezone(pytz.timezone('US/Eastern'))
        
        current_time = dt_et.time()
        
        market_open = time(9, 30)
        or_finished = (datetime.combine(dt_et.date(), market_open) + timedelta(minutes=self.or_minutes)).time()
        
        # Use dt_et.date() for daily reset tracking
        if self.last_date != dt_et.date():
            self.last_date = dt_et.date()
            self.or_high = -999999
            self.or_low = 999999
            self.or_locked = False
            self.or_vol_accum = 0
            self.mode_today = None
            self.daily_trade_count = 0
            
        if current_time >= time(15, 55):
            if self.portfolio:
                pos_obj = self.portfolio.current_positions.get(symbol)
                pos = pos_obj.quantity if pos_obj else 0
                if pos != 0:
                    self.exit(symbol)
            return

        if current_time < market_open: return

        if not self.or_locked:
            if current_time < or_finished:
                self.or_high = max(self.or_high, event.high)
                self.or_low = min(self.or_low, event.low)
                self.or_vol_accum += event.volume
            else:
                self.or_locked = True
                self._classify_day(event)
        
        pos = 0
        if self.portfolio:
            pos_obj = self.portfolio.current_positions.get(symbol)
            pos = pos_obj.quantity if pos_obj else 0

        # Trade FOLLOW only (Filters Disabled)
        if self.or_locked and self.mode_today == 'follow' and pos == 0 and self.daily_trade_count == 0:
            if event.close > self.or_high:
                self.buy(symbol, 1)
                self.daily_trade_count = 1
                self.take_profit = event.close + (self.atr_today * self.TP_ATR_F)
                self.stop_loss = event.close - (self.atr_today * self.SL_ATR_F)
            elif event.close < self.or_low:
                self.sell(symbol, 1)
                self.daily_trade_count = 1
                self.take_profit = event.close - (self.atr_today * self.TP_ATR_F)
                self.stop_loss = event.close + (self.atr_today * self.SL_ATR_F)
        
        elif pos != 0:
            if pos > 0:
                if event.low < self.stop_loss: self.exit(symbol)
                elif event.high > self.take_profit: self.exit(symbol)
            elif pos < 0:
                if event.high > self.stop_loss: self.exit(symbol)
                elif event.low < self.take_profit: self.exit(symbol)
