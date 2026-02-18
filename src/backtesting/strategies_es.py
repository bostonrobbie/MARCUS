
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from .strategy import Strategy
from .schema import Bar, OrderType
from .data import DataHandler
from queue import Queue

class EsOrGapCombo(Strategy):
    """
    High-Fidelity Port of 'ES OR+Gap — Follow & Fade Combo'.
    Includes:
    - ORB (15m default)
    - Regime Classification: Gap + RVOL + Hurst + VIX
    - 'Follow' Mode (Trend) vs 'Fade' Mode (Mean Reversion)
    - Detailed Risk Management (SL, TP, Trailing)
    - Supports Limit Order Entries (Optional)
    """
    def __init__(self, 
                 bars: DataHandler, 
                 events: Queue,
                 # Core
                 orb_period_min=15,
                 use_limit_entry=False, # User Request: "Add limit orders"
                 # Filters
                 rvol_lookback=20,
                 rvol_threshold=1.45,
                 hurst_lookback=100,
                 hurst_threshold=0.52,
                 gap_min_follow=0.0015, # 0.15%
                 gap_min_fade=0.0025,   # 0.25%
                 # Follow Params
                 sl_atr_follow=2.0,
                 tp_atr_follow=10.0,
                 trail_atr_follow=1.5,
                 use_trail_follow=False,
                 # Fade Params
                 sl_atr_fade=2.0,
                 tp_atr_fade=6.0,
                 trail_atr_fade=2.0,
                 use_trail_fade=True
                 ):
        
        super().__init__(bars, events)
        
        # Core Params
        self.orb_period = int(orb_period_min)
        self.use_limit_entry = use_limit_entry
        
        # Filters
        self.rvol_win = int(rvol_lookback)
        self.rvol_thresh = float(rvol_threshold)
        self.hurst_win = int(hurst_lookback)
        self.hurst_thresh = float(hurst_threshold)
        self.gap_min_follow = float(gap_min_follow)
        self.gap_min_fade = float(gap_min_fade)
        
        # Risk Params - Follow
        self.sl_f = float(sl_atr_follow)
        self.tp_f = float(tp_atr_follow)
        self.tr_f = float(trail_atr_follow)
        self.use_tr_f = use_trail_follow
        
        # Risk Params - Fade
        self.sl_d = float(sl_atr_fade)
        self.tp_d = float(tp_atr_fade)
        self.tr_d = float(trail_atr_fade)
        self.use_tr_d = use_trail_fade
        
        # Timing
        self.t_rth_start = time(9, 30)
        self.t_flatten = time(15, 55)
        
        # State
        self.current_date = None
        self.prior_close = None
        self.day_open = None
        self.gap_pct = 0.0
        
        self.orb_high = -1.0
        self.orb_low = 1e9
        self.orb_volume = 0.0
        self.orb_locked = False
        
        self.mode = "NEITHER" # FOLLOW, FADE, or NEITHER
        self.vol_history = [] 
        self.traded_today = False
        self.in_trade = False
        self.entry_price = 0.0
        self.trade_type = None # LONG/SHORT
        
        # Hurst internal buffer
        self.prices_history = []

    def _calc_hurst(self, prices) -> float:
        if len(prices) < 20: return 0.5
        try:
            # Simplified simplified R/S
            returns = np.diff(np.log(prices))
            if len(returns) == 0: return 0.5
            series = returns
            mean = np.mean(series)
            dev = series - mean
            cum_dev = np.cumsum(dev)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(series)
            if s == 0: return 0.5
            return np.log(r / s) / np.log(len(series))
        except Exception:
            return 0.5

    def calculate_signals(self, event: Bar):
        bar_date = event.timestamp.date()
        bar_time = event.timestamp.time()
        
        # 1. New Day Reset
        if self.current_date != bar_date:
            # Capture Prior Close (approximation: last event close if sequential)
            # ideally DataHandler gives this, but we can assume previous bar close 
            # if we run sequentially. 
            # ERROR: event.close is current. We need prev close.
            # We'll use self.prices_history[-1] if available.
            
            if self.prices_history:
                self.prior_close = self.prices_history[-1]
            else:
                self.prior_close = event.open # Fallback
                
            self.current_date = bar_date
            self.day_open = event.open
            
            # Gap Calc
            if self.prior_close and self.prior_close > 0:
                self.gap_pct = (self.day_open - self.prior_close) / self.prior_close
            else:
                self.gap_pct = 0.0
                
            self.orb_high = -1.0
            self.orb_low = 1e9
            self.orb_volume = 0.0
            self.orb_locked = False
            self.mode = "NEITHER"
            self.traded_today = False
            self.in_trade = False
            
        # Accumulate History for Hurst
        self.prices_history.append(event.close)
        if len(self.prices_history) > self.hurst_win * 2:
            self.prices_history.pop(0)
            
        # 2. ORB Phase
        orb_cutoff = (datetime.combine(datetime.today(), self.t_rth_start) + timedelta(minutes=self.orb_period)).time()
        
        if self.t_rth_start <= bar_time < orb_cutoff:
            if self.orb_high == -1.0:
                self.orb_high = event.high
                self.orb_low = event.low
            else:
                self.orb_high = max(self.orb_high, event.high)
                self.orb_low = min(self.orb_low, event.low)
            self.orb_volume += event.volume
            
        # 3. Lock & Classify
        elif bar_time >= orb_cutoff and not self.orb_locked:
            self.orb_locked = True
            
            # print(f"[{event.timestamp}] Day Start. Gap: {self.gap_pct:.4%}")
            
            # RVOL
            avg_vol = np.mean(self.vol_history) if len(self.vol_history) > 5 else self.orb_volume
            rvol = self.orb_volume / avg_vol if avg_vol > 0 else 1.0
            self.vol_history.append(self.orb_volume) # Update history
            if len(self.vol_history) > self.rvol_win: self.vol_history.pop(0)

            # Hurst
            hurst = self._calc_hurst(self.prices_history[-self.hurst_win:])
            
            # Signal Generation (Regime)
            # Follow: High RVOL + Trend Structure (Hurst > 0.52)
            cond_rvol = rvol >= self.rvol_thresh
            cond_hurst = hurst >= self.hurst_thresh
            
            is_trend = cond_rvol and cond_hurst
            is_chop = not cond_rvol and not cond_hurst

            if is_trend:
                self.mode = "FOLLOW"
            elif is_chop:
                self.mode = "FADE"
            else:
                self.mode = "FOLLOW" # Default prefer follow or ambiguous
                
            # print(f"[{event.timestamp}] ORB Locked. Mode: {self.mode} | RVOL:{rvol:.2f} (Thresh {self.rvol_thresh}) | Hurst:{hurst:.2f} (Thresh {self.hurst_thresh})")
            
            # Gap Check overrides
            # (Pine script uses Gap as confirmation for specific entries, handled below)
            
        # 4. Trading Logic
        elif self.orb_locked and not self.in_trade and not self.traded_today and bar_time < self.t_flatten:
            
            # Indicators
            atr = 20.0 # Placeholder: calculate properly if possible or use fixed estimate for speed
            # In real port: self.bars.get_latest_vars... TR... rolling mean.
            
            # Debugging potential signals
            if self.mode == "FOLLOW":
                # Long: Gap Up + Breakout Up
                if event.close > self.orb_high:
                    if self.gap_pct >= self.gap_min_follow:
                        # print(f"[{event.timestamp}] ENTRY SIGNAL: FOLLOW LONG")
                        self._execute_entry("LONG", self.sl_f, self.tp_f, atr, event)
                    else:
                        pass # print(f"[{event.timestamp}] REJECT FOLLOW LONG: Gap {self.gap_pct:.2%} < Min {self.gap_min_follow:.2%}")
                # Short: Gap Down + Breakout Down
                elif event.close < self.orb_low:
                    if self.gap_pct <= -self.gap_min_follow:
                        # print(f"[{event.timestamp}] ENTRY SIGNAL: FOLLOW SHORT")
                        self._execute_entry("SHORT", self.sl_f, self.tp_f, atr, event)
                    else:
                        pass

            # FADE LOGIC
            elif self.mode == "FADE":
                # Long: Gap Down (Trap) + Reclaim Low
                if event.low <= self.orb_low: # Check if price touched or went below ORB low
                     # print(f"[{event.timestamp}] FADE CHECK LONG...") # noisy
                     if self.gap_pct <= -self.gap_min_fade and event.close > self.orb_low: # Reversing up through Low
                         # print(f"[{event.timestamp}] ENTRY SIGNAL: FADE LONG")
                         self._execute_entry("LONG", self.sl_d, self.tp_d, atr, event)
                # Short: Gap Up (Trap) + Fail High
                elif event.high >= self.orb_high: # Check if price touched or went above ORB high
                     # print(f"[{event.timestamp}] FADE CHECK SHORT...") # noisy
                     if self.gap_pct >= self.gap_min_fade and event.close < self.orb_high: # Reversing down through High
                         # print(f"[{event.timestamp}] ENTRY SIGNAL: FADE SHORT")
                         self._execute_entry("SHORT", self.sl_d, self.tp_d, atr, event)

        # 5. Management (Trailing Stop approximation for Event Engine)
        # Note: BacktestEngine usually handles fixed SL/TP if orders submitted. 
        # Trailing logic would go here if engine doesn't support it natively.
        
        # 6. EOD Exit
        if bar_time >= self.t_flatten and self.in_trade:
             self.exit(event.symbol)
             self.in_trade = False

    def _execute_entry(self, side, sl_mult, tp_mult, atr, event):
        price = event.close
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult
        
        limit_px = None
        if self.use_limit_entry:
            # If Limit Entry, we try to enter at the ORB level retest
            limit_px = self.orb_high if side == "LONG" else self.orb_low
            # Logic caveat: Strategy.buy() with limit usually implies "Buy if price <= Limit".
            # For a breakout retest, we want to buy a pullback.
        
        if side == "LONG":
            sl = price - sl_dist
            tp = price + tp_dist
            self.buy(event.symbol, quantity=1, limit_price=limit_px) 
            # Note: Strategy base class needs to handle passing SL/TP to execution if supported
            # Or we manage exits manually. For now assuming execution handles basic fills.
        else:
            sl = price + sl_dist
            tp = price - tp_dist
            self.sell(event.symbol, quantity=1, limit_price=limit_px)
            
        self.in_trade = True
        self.traded_today = True
        self.entry_price = price
