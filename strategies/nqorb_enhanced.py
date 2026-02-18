
import pandas as pd
import numpy as np
from datetime import time
from .nqorb_15m import NqOrb15m

class NqOrbEnhanced(NqOrb15m):
    """
    Enhanced 15m ORB Strategy with "Synthesis" Filters:
    1. HTF Trend Filter (Daily SMA 100)
    2. RVOL Filter (Volume > 1.5x 20-period Avg)
    3. ATR Trailing Stop (Dynamic Exits)
    """
    def __init__(self, bars, events, 
                 # Base Params
                 orb_start_time=time(9, 30), 
                 orb_end_time=time(9, 45),
                 exit_time=time(15, 45),
                 sl_atr_mult=2.0, 
                 tp_atr_mult=4.0, # Used only if TS disabled or as initial target
                 ema_filter=50,
                 atr_filter=14,
                 atr_max_mult=2.5,
                 # Enhanced Params
                 use_htf=True, htf_ma_period=100,
                 use_rvol=True, rvol_thresh=1.5,
                 use_trailing_stop=True, ts_atr_mult=2.0,
                 use_confidence_sizing=False, max_contracts=3,
                 verbose=True):
        
        super().__init__(bars, events, orb_start_time, orb_end_time, exit_time, 
                         sl_atr_mult, tp_atr_mult, ema_filter, atr_filter, atr_max_mult, verbose)
        
        self.use_htf = use_htf
        self.htf_ma_period = htf_ma_period
        self.use_rvol = use_rvol
        self.rvol_thresh = rvol_thresh
        self.use_trailing_stop = use_trailing_stop
        self.ts_atr_mult = ts_atr_mult
        self.use_confidence_sizing = use_confidence_sizing
        self.max_contracts = max_contracts
        self.last_atr = 0.0
        self.daily_sma_cache = 0.0
        self.daily_sma_50_cache = 0.0
        self.cache_date = None

    def on_fill(self, event):
        from backtesting.schema import OrderSide
        if event.side == OrderSide.BUY:
            if self.in_short:
                # Closed Short
                self.in_short = False
                if self.verbose: print(f"[{event.timestamp}] COVER SHORT FILLED @ {event.price}")
            else:
                # Opened Long
                self.in_long = True
                self.entry_price = event.price
                self.traded_today = True
                
                # Setup Exits
                atr = self.last_atr
                self.current_sl = event.price - (atr * self.sl_atr_mult)
                self.current_tp = event.price + (atr * self.tp_atr_mult)
                
                if self.verbose: print(f"[{event.timestamp}] LONG FILLED @ {event.price}. SL: {self.current_sl:.2f}, TP: {self.current_tp:.2f}")

                # Submit Limit TP if not trailing
                if not self.use_trailing_stop:
                    self.sell(event.symbol, quantity=event.quantity, limit_price=self.current_tp)

        elif event.side == OrderSide.SELL:
            if self.in_long:
                # Closed Long
                self.in_long = False
                if self.verbose: print(f"[{event.timestamp}] SELL LONG FILLED @ {event.price}")
            else:
                # Opened Short
                self.in_short = True
                self.entry_price = event.price
                self.traded_today = True
                
                atr = self.last_atr
                self.current_sl = event.price + (atr * self.sl_atr_mult)
                self.current_tp = event.price - (atr * self.tp_atr_mult)
                
                if self.verbose: print(f"[{event.timestamp}] SHORT FILLED @ {event.price}. SL: {self.current_sl:.2f}, TP: {self.current_tp:.2f}")
                
                 # Submit Limit TP if not trailing
                if not self.use_trailing_stop:
                    self.buy(event.symbol, quantity=event.quantity, limit_price=self.current_tp)

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # 1. Daily State Management (Inherited, but strictly local state update needed?)
        # Base class handles: current_date, orb_high/low reset.
        # We need to ensure we call daily reset if base doesn't do it cleanly before this logic?
        # Actually, we can reuse base logic for ORB capture, but we need to override the ENTRY logic.
        
        # Super call to handle ORB High/Low capture
        # Issue: Base class calculate_signals does EVERYTHING. 
        # We need to replicate the flow or it will double-execute.
        # Recommendation: Copy-paste-modify full method to avoid conflict or tricky inheritance hook.
        
        # --- FULL OVERRIDE OF LOGIC ---
        
        # Daily State Reset
        if self.current_date != ts.date():
            self.current_date = ts.date()
            self.orb_high = -float('inf')
            self.orb_low = float('inf')
            self.traded_today = False
            self.entry_price = None
            self.in_long = False
            self.in_short = False
            self.current_sl = 0
            self.current_tp = 0 # Not strictly used in TS, but good for tracking

        # Define Range
        if self.orb_start <= current_time < self.orb_end:
            self.orb_high = max(self.orb_high, event.high)
            self.orb_low = min(self.orb_low, event.low)
            return

        # Prepare Data
        # We need more history for Daily SMA (approx 100 days * ~78 bars/day = ~8000 bars)
        # NQ 24h market? 
        # SmartDataHandler provides whatever is in CSV. Assuming 15m bars.
        # 100 Days * 96 (if 24h) or 26 (if RTH) bars.
        # Let's request enough bars.
        # Optimize Lookback: Only fetch 10000 bars if we need to recalculate Daily SMA (once per day)
        # Otherwise, just need enough for EMA/ATR (e.g. 500)
        need_htf_recalc = self.use_htf and (self.cache_date != ts.date())
        lookback = 10000 if need_htf_recalc else 500
        
        bars = self.bars.get_latest_bars(symbol, N=lookback)
        if len(bars) < 100:
            return
            
        df = pd.DataFrame([{
            'close': b.close, 'high': b.high, 'low': b.low, 'open': b.open, 'volume': b.volume
        } for b in bars])
        
        # Calculate Indicators
        try:
            import pandas_ta as ta
        except ImportError:
            return

        ema = ta.ema(df['close'], length=self.ema_filter).iloc[-1]
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_filter).iloc[-1]
        self.last_atr = atr
        
        # Enhanced Indicators
        daily_ma_val = None
        if self.use_htf:
            # Resample to Daily
            # df.index is currently integer. We rely on contiguous 15m bars? 
            # No, 'get_latest_bars' returns list of Bar objects. We lost index.
            # We assume they are contiguous relative to the simulation valid stream.
            # Better: Create index from `bars` timestamps if available? 
            # SmartDataHandler Bar objects usually have 'timestamp' attr? 
            # Let's look at `nqorb_15m.py`... standard usage is df from basic dict.
            # We need to assume `bars` has timestamps? 
            # Standard `Bar` tuple might not have timestamp? 
            # Check `backtesting.schema`.
            # If not, we can't accurately resample. 
            # fallback: Simple Moving Average of 15m bars * multiplier? No, inaccurate.
            # Let's try to verify if `Bar` has timestamp.
            pass 
            # Actually, let's look at how vectors did it: It had a full DF with datetime index.
            # Here in event-driven, we just get "latest bars". 
            # To do this robustly in Event-Mode without complex data fetching:
            # We will use a proxy: SMA(15m) * length? No.
            # We will use the `Daily SMA` calculated from `df['close'].rolling(window=100*96).mean()`?
            # Approximation: 100 Days ~ 100 * 26 bars (if RTH).
            # Let's assume we can get timestamps. 
            pass

        # Execution Window
        if self.orb_end <= current_time < self.exit_time:
            if not self.traded_today:
                # --- ENTRY LOGIC ---
                range_size = self.orb_high - self.orb_low
                
                # Filters
                htf_ok_long = True
                htf_ok_short = True
                daily_sma = 0
                daily_sma_50 = 0
                
                if self.use_htf:
                    if self.cache_date != ts.date():
                        # Construct Time Series for Resampling
                        ts_index = [b.timestamp for b in bars]
                        df.index = pd.DatetimeIndex(ts_index)
                        daily_close = df['close'].resample('D').last().dropna()
                        
                        d_sma = 0
                        d_sma_50 = 0
                        
                        if len(daily_close) > self.htf_ma_period:
                            d_sma = ta.sma(daily_close, length=self.htf_ma_period).iloc[-1]
                            
                            if self.use_confidence_sizing:
                                d_sma_50 = ta.sma(daily_close, length=50).iloc[-1]
                        
                        self.daily_sma_cache = d_sma
                        self.daily_sma_50_cache = d_sma_50
                        self.cache_date = ts.date()
                    
                    # Use Cache
                    daily_sma = self.daily_sma_cache
                    daily_sma_50 = self.daily_sma_50_cache
                    
                    if daily_sma > 0:
                        htf_ok_long = (event.close > daily_sma)
                        htf_ok_short = (event.close < daily_sma)
                    
                rvol_ok = True
                if self.use_rvol:
                    # RVOL = Current Vol / SMA(Vol, 20)
                    # 20 periods = 5 hours of 15m bars
                    vol_sma = ta.sma(df['volume'], length=20).iloc[-1]
                    if vol_sma > 0:
                        rvol = df['volume'].iloc[-1] / vol_sma
                        if rvol < self.rvol_thresh:
                            rvol_ok = False
                
                atr_cond = (range_size <= (atr * self.atr_max_mult))
                
                # Check Long
                if self.orb_high != -float('inf'):
                    if event.close > self.orb_high and event.close > ema:
                        if atr_cond and htf_ok_long and rvol_ok:
                            qty = 1
                            if self.use_confidence_sizing:
                                # Trend Bonus
                                if daily_sma_50 > 0 and event.close > daily_sma_50 and daily_sma_50 > daily_sma:
                                    qty += 1
                                # Volume Bonus
                                if rvol > 2.0:
                                    qty += 1
                                qty = min(qty, self.max_contracts)

                            if self.verbose:
                                print(f"[{ts}] ENHANCED LONG ENTRY SIGNAL @ {event.close} | RVOL: {rvol:.2f}, SMA50: {daily_sma_50:.2f}, SMA100: {daily_sma:.2f}, QTY: {qty}")
                            self.buy(symbol, quantity=qty)

                    # Check Short
                    elif event.close < self.orb_low and event.close < ema:
                        if atr_cond and htf_ok_short and rvol_ok:
                            qty = 1
                            if self.use_confidence_sizing:
                                # Trend Bonus
                                if daily_sma_50 > 0 and event.close < daily_sma_50 and daily_sma_50 < daily_sma:
                                    qty += 1
                                # Volume Bonus
                                if rvol > 2.0:
                                    qty += 1
                                qty = min(qty, self.max_contracts)

                            if self.verbose:
                                print(f"[{ts}] ENHANCED SHORT ENTRY SIGNAL @ {event.close} | RVOL: {rvol:.2f}, SMA50: {daily_sma_50:.2f}, SMA100: {daily_sma:.2f}, QTY: {qty}")
                            self.sell(symbol, quantity=qty)

            # --- EXIT MANAGAMENT (Trailing Stop) ---
            elif self.in_long:
                # Dynamic Trailing Stop
                if self.use_trailing_stop:
                    # New Level = High - (ATR * Mult)
                    # We update SL if new level is higher than current SL
                    # Using current bar's high? Or previous? 
                    # Live trading: We update based on observed highs.
                    possible_sl = event.high - (atr * self.ts_atr_mult)
                    if possible_sl > self.current_sl:
                        self.current_sl = possible_sl
                
                # Check Exits
                if event.low <= self.current_sl:
                    if self.verbose: print(f"[{ts}] TRAILING STOP HIT (LONG) @ {event.low}")
                    self.exit(symbol)
                    
            elif self.in_short:
                # Dynamic Trailing Stop
                if self.use_trailing_stop:
                    # New Level = Low + (ATR * Mult)
                    possible_sl = event.low + (atr * self.ts_atr_mult)
                    if possible_sl < self.current_sl:
                        self.current_sl = possible_sl
                        
                # Check Exits
                if event.high >= self.current_sl:
                    if self.verbose: print(f"[{ts}] TRAILING STOP HIT (SHORT) @ {event.high}")
                    self.exit(symbol)

        # Session End
        if current_time >= self.exit_time:
            if self.in_long or self.in_short:
                if self.verbose: print(f"[{ts}] SESSION END")
                self.exit(symbol)
                self.in_long = False
                self.in_short = False
