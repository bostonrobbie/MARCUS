import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import time
from .accelerate import get_dataframe_library, get_array_library
from .monitor import PipelineMonitor
from .type_utils import ensure_pandas_series, normalize_returns

monitor = PipelineMonitor()

try:
    from numba import jit
    monitor.log_gpu_status(True, "Numba JIT available. GPU/Fast CPU acceleration enabled.")
except ImportError:
    monitor.log_gpu_status(False, "Numba not found. Vector Engine will run in strict CPU mode (Slow).")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class VectorStrategy(ABC):
    """
    Abstract Base Class for Vectorized Strategies.
    Generates signals for an entire DataFrame at once.
    """
    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def generate_signals(self, df):
        """
        Takes a DataFrame (cudf or pandas) and returns a Series of signals.
        1 = LONG, -1 = SHORT, 0 = FLAT
        """
        raise NotImplementedError

class VectorEngine:
    """
    High-Performance Backtest Engine using Vectorized Operations.
    Ideal for GPU acceleration.
    Includes cost modeling (Slippage + Commissions).
    """
    def __init__(self, strategy: VectorStrategy, initial_capital=100000.0, commission=1.0, slippage=1.0, volatility_factor=0.01, point_value=20.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_per_unit = commission
        self.slippage_per_unit = slippage
        self.volatility_factor = volatility_factor
        self.point_value = point_value  # NQ: 1 point = $20 (E-mini)
        self.pd = get_dataframe_library()
        self.np = get_array_library()

    def run(self, df):
        """
        Runs the vectorized backtest on the provided DataFrame.
        """
        # 1. Generate Signals
        signals = self.strategy.generate_signals(df)

        # 2. Calculate Returns
        # We assume execution on NEXT OPEN (shift signals by 1)
        # Normalize column names to capitalized form
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower == 'close' and col != 'Close': col_map[col] = 'Close'
            elif lower == 'high' and col != 'High': col_map[col] = 'High'
            elif lower == 'low' and col != 'Low': col_map[col] = 'Low'
            elif lower == 'open' and col != 'Open': col_map[col] = 'Open'
            elif lower == 'volume' and col != 'Volume': col_map[col] = 'Volume'
        if col_map:
            df = df.rename(columns=col_map)

        # Returns from Open to Close or Close to Close?
        # Standard: Close to Close returns applied to position held at shift(1)
        returns = df['Close'].pct_change().fillna(0)

        # Position is held for the bar AFTER the signal
        pos = signals.shift(1).fillna(0)

        # --- COST MODELING (Futures-Aware) ---
        # For futures: costs are per-contract, not per-share.
        # Convert to % of notional: notional = price * point_value
        # E.g., NQ at 20000 with point_value=20 -> notional = $400K
        # Commission $2.06 -> 2.06/400000 = 0.000515% per side
        prices = df['Close']
        highs = df['High'] if 'High' in df.columns else prices
        lows = df['Low'] if 'Low' in df.columns else prices

        # Dynamic slippage: small % of bar range for volatile periods
        volatility = (highs - lows).abs()
        vol_slippage = volatility * self.volatility_factor

        # Total dollar cost per trade = commission + slippage + vol-dependent slippage
        total_cost_dollars = self.commission_per_unit + self.slippage_per_unit + vol_slippage

        # Convert to % of notional value (price * point_value)
        safe_prices = prices.replace(0, np.nan).ffill().fillna(1.0)
        notional = safe_prices * self.point_value  # e.g. 20000 * 20 = $400K
        cost_pct = total_cost_dollars / notional

        # Turnover (position changes)
        turnover = pos.diff().abs().fillna(0)

        # Transaction Costs
        transaction_costs = turnover * cost_pct

        # Strategy Returns (Gross)
        strat_returns = pos * returns

        # Net Returns
        net_returns = strat_returns - transaction_costs

        # cumulative equity
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()

        # Normalize all return values to pandas Series to ensure consistent types
        # This handles mixed pandas/cuDF/numpy environments
        result = {
            'equity_curve': equity_curve,
            'signals': signals,
            'returns': net_returns,
            'turnover': turnover
        }
        return normalize_returns(result, index=df.index)

class VectorizedMA(VectorStrategy):
    """
    Vectorized implementation of Moving Average Crossover.
    """
    def __init__(self, short_window=50, long_window=200, **kwargs):
        super().__init__(short_window=short_window, long_window=long_window, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        pd_lib = get_dataframe_library()
        close = df['Close'] if 'Close' in df.columns else df['close']
        short_ma = close.rolling(window=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window).mean()

        signals = pd_lib.Series(0, index=df.index)
        signals[short_ma > long_ma] = 1   # Long when fast > slow
        signals[short_ma < long_ma] = -1   # Short when fast < slow
        # Stay flat during warmup period (NaN moving averages)
        signals[long_ma.isna()] = 0
        return signals

class VectorizedNQORB(VectorStrategy):
    """
    Vectorized implementation of 15m Opening Range Breakout (ORB).
    Optimized for GPU (cuDF/Pandas).
    """
    def __init__(self, orb_start="09:30", orb_end="09:45", ema_filter=50, atr_filter=14, sl_atr_mult=2.0, tp_atr_mult=4.0, atr_max_mult=2.5,
                 use_htf=False, htf_ma=200,
                 use_rvol=False, rvol_thresh=1.5,
                 use_hurst=False, hurst_thresh=0.5,
                 use_adx=False, adx_thresh=20,
                 use_trailing_stop=False, ts_atr_mult=3.0,
                 **kwargs):
        super().__init__(orb_start=orb_start, orb_end=orb_end, ema_filter=ema_filter, atr_filter=atr_filter, sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, atr_max_mult=atr_max_mult, **kwargs)
        self.orb_start = orb_start
        self.orb_end = orb_end
        self.ema_filter = int(ema_filter)
        self.atr_filter = int(atr_filter)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)
        self.atr_max_mult = float(atr_max_mult)

        # Phase 2 Features
        self.use_htf = use_htf
        self.htf_ma = int(htf_ma)
        self.use_rvol = use_rvol
        self.rvol_thresh = float(rvol_thresh)
        self.use_hurst = use_hurst
        self.hurst_thresh = float(hurst_thresh)
        self.use_adx = use_adx
        self.adx_thresh = float(adx_thresh)
        self.use_trailing_stop = use_trailing_stop
        self.ts_atr_mult = float(ts_atr_mult)

    def generate_signals(self, df):
        from . import ta

        pd_lib = get_dataframe_library()

        # Prepare Data
        # We need numpy arrays for Numba
        # Ensure we have all necessary columns
        if 'close' not in df.columns: df.rename(columns={'Close': 'close'}, inplace=True)
        if 'high' not in df.columns: df.rename(columns={'High': 'high'}, inplace=True)
        if 'low' not in df.columns: df.rename(columns={'Low': 'low'}, inplace=True)

        # Calculate Indicators (Pandas/TA-Lib is fast enough for vector calc)
        if 'volume' not in df.columns and 'Volume' in df.columns: df.rename(columns={'Volume': 'volume'}, inplace=True)
        if 'volume' not in df.columns: df['volume'] = 1.0 # fallback

        # Calculate Base Indicators
        # Using custom ta which returns Series
        ema = ta.ema(df['close'], length=self.ema_filter).fillna(0).values.astype(np.float64)
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_filter).fillna(0).values.astype(np.float64)

        # --- Phase 2: Advanced Indicators ---
        # 1. HTF Trend (Daily MA)
        # Resample to Daily -> Calc MA -> Reindex to Intraday
        if self.use_htf:
            daily_close = df['close'].resample('D').last().dropna()
            daily_ma = ta.sma(daily_close, length=self.htf_ma)

            if daily_ma is None or daily_ma.empty:
                # Not enough data for MA
                daily_ma = pd.Series(0, index=daily_close.index)

            # Reindex and ffill (careful: avoid lookahead bias, shift 1 day)
            # Daily MA for today should be based on YESTERDAY's close
            daily_ma = daily_ma.shift(1).reindex(df.index).ffill().fillna(0).values
        else:
            daily_ma = np.zeros(len(df))

        # 2. RVOL (Relative Volume)
        if self.use_rvol:
            avg_vol = ta.sma(df['volume'], length=20).fillna(1.0) # Avoid div/0
            rvol = df['volume'] / avg_vol
            rvol = rvol.fillna(0).values
        else:
            rvol = np.zeros(len(df))

        # 3. Hurst Exponent (requires specialized calc, ta libs usually lack rolling hurst)
        # We will use a simplified Efficiency Ratio (ER) as a proxy if simple Hurst isn't avail.
        # ER = Change / Sum of ranges. High ER => Trend.
        # To strictly do Hurst: log(R/S) / log(T)
        # For speed in Numba/Vector, implementation is tricky.
        # Let's placeholder with 0.5 (neutral) if disabled, or use ER.
        if self.use_hurst:
            # Using KAMA Efficiency Ratio as fast Trend Persistence proxy
            # ER = abs(Change) / Volatility-Sum
            change = df['close'].diff(10).abs()
            volatility = df['close'].diff().abs().rolling(10).sum()
            er = change / volatility
            hurst_proxy = er.fillna(0.5).values
            # Note: ER ranges 0-1. ER > 0.5 implies trendiness similar to Hurst > 0.5
        else:
            hurst_proxy = np.zeros(len(df))

        # 4. ADX
        if self.use_adx:
            # Custom ta.adx returns a Series of ADX values
            adx_series = ta.adx(df['high'], df['low'], df['close'], length=14)
            adx_val = adx_series.fillna(0).values
        else:
            adx_val = np.zeros(len(df))

        volume = df['volume'].values

        # Convert Time logic to integers for faster comparison
        # We'll use minute of day: 9:30 = 9*60 + 30 = 570
        times = df.index.hour * 60 + df.index.minute

        # Create trading day IDs for proper session detection.
        # NQ trades nearly 24h, so times[i] < times[i-1] is unreliable.
        # Instead, use calendar date to detect new trading sessions.
        dates = df.index.date
        # Convert dates to integer day IDs that Numba can compare
        # Use ordinal day number (integer)
        day_ids = np.array([d.toordinal() for d in dates], dtype=np.int64)

        # For backward compat
        timestamps = df.index.values.astype(np.int64)

        # Parse Strategy Times
        t_start = pd.to_datetime(self.orb_start).time()
        start_min = t_start.hour * 60 + t_start.minute

        t_end = pd.to_datetime(self.orb_end).time()
        end_min = t_end.hour * 60 + t_end.minute

        # Exit time typically 15:45
        exit_min = 15 * 60 + 45

        # extract numpy arrays (enforce float64 for Numba JIT compatibility)
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)

        # Run Numba Core
        signals = _numba_orb_logic(
            day_ids, times.values, closes, highs, lows, ema, atr,
            start_min, end_min, exit_min,
            self.atr_max_mult, self.sl_atr_mult, self.tp_atr_mult,
            self.use_htf, daily_ma,
            self.use_rvol, rvol, self.rvol_thresh,
            self.use_hurst, hurst_proxy, self.hurst_thresh,
            self.use_adx, adx_val, self.adx_thresh,
            self.use_trailing_stop, self.ts_atr_mult
        )

        return pd.Series(signals, index=df.index)

@jit(nopython=True)
def _numba_orb_logic(day_ids, times, closes, highs, lows, ema, atr,
                     start_min, end_min, exit_min,
                     atr_max_mult, sl_mult, tp_mult,
                     use_htf, daily_ma,
                     use_rvol, rvol, rvol_thresh,
                     use_hurst, hurst, hurst_thresh,
                     use_adx, adx, adx_thresh,
                     use_ts, ts_mult):
    n = len(closes)
    signals = np.zeros(n, dtype=np.int32)

    # State Variables
    orb_high = -1.0
    orb_low = 1e9

    # Session State
    traded_today = False
    in_pos = 0  # 1 long, -1 short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    # We iterate chronologically
    for i in range(1, n):
        t = times[i]

        # 1. Detect New Trading Day using calendar date ordinals
        # This correctly handles 24h NQ futures sessions
        if day_ids[i] != day_ids[i - 1]:
            orb_high = -1.0
            orb_low = 1e9
            traded_today = False
            in_pos = 0

        # ORB Calculation Window
        if t >= start_min and t < end_min:
            if orb_high == -1.0: # first bar of ORB
                orb_high = highs[i]
                orb_low = lows[i]
            else:
                if highs[i] > orb_high: orb_high = highs[i]
                if lows[i] < orb_low: orb_low = lows[i]

        # Trading Window
        elif t >= end_min and t < exit_min:
            # Check Exits First
            if in_pos != 0:
                if in_pos == 1:
                    # Long Exit

                    # Trailing Stop Update
                    if use_ts:
                        # For long, SL moves UP only.
                        # TS Level = High - (ATR * mult)
                        # We used entry-based SL initially.
                        # Dynamic SL based on Current High - TS
                        # Check if we assume 'high' of this bar contributes to TS move?
                        # Usually TS is calculated on close or high of PREV bar to avoid peeking?
                        # Or intra-bar high?
                        # If intra-bar, we can survive provided Low doesn't hit SL first.
                        # Simplistic Conservative: Update SL for NEXT bar based on THIS bar's High.
                        new_sl = highs[i] - (atr[i] * ts_mult)
                        if new_sl > sl_price:
                            sl_price = new_sl

                    if lows[i] <= sl_price:
                        # Stop Hit
                        signals[i] = 0
                        in_pos = 0
                    elif not use_ts and highs[i] >= tp_price:
                        # TP Hit (Only if Fixed Target is used)
                        in_pos = 0

                elif in_pos == -1:
                    # Short Exit // Trailing Stop
                    if use_ts:
                        new_sl = lows[i] + (atr[i] * ts_mult)
                        if new_sl < sl_price:
                            sl_price = new_sl

                    if highs[i] >= sl_price:
                        in_pos = 0
                    elif not use_ts and lows[i] <= tp_price:
                        in_pos = 0

            # Check Entries
            if in_pos == 0 and not traded_today and orb_high != -1.0:
                range_size = orb_high - orb_low
                cur_atr = atr[i]

                if range_size > 0 and cur_atr > 0 and range_size <= (cur_atr * atr_max_mult):
                    # Long Entry
                    if closes[i] > orb_high and closes[i] > ema[i]:
                        # Check Filters
                        valid = True
                        if use_htf and closes[i] <= daily_ma[i]: valid = False
                        if use_rvol and rvol[i] <= rvol_thresh: valid = False
                        if use_hurst and hurst[i] <= hurst_thresh: valid = False
                        if use_adx and adx[i] <= adx_thresh: valid = False

                        if valid:
                            in_pos = 1
                            entry_price = closes[i]
                            sl_price = entry_price - (cur_atr * sl_mult)
                            tp_price = entry_price + (cur_atr * tp_mult)
                            traded_today = True

                    # Short Entry
                    elif closes[i] < orb_low and closes[i] < ema[i]:
                        # Check Filters
                        valid = True
                        if use_htf and closes[i] >= daily_ma[i]: valid = False
                        if use_rvol and rvol[i] <= rvol_thresh: valid = False
                        if use_hurst and hurst[i] <= hurst_thresh: valid = False
                        if use_adx and adx[i] <= adx_thresh: valid = False

                        if valid:
                            in_pos = -1
                            entry_price = closes[i]
                            sl_price = entry_price + (cur_atr * sl_mult)
                            tp_price = entry_price - (cur_atr * tp_mult)
                            traded_today = True

        # Session End
        elif t >= exit_min:
            in_pos = 0

        # Record Signal
        # VectorEngine logic: 'pos' is held for NEXT bar.
        # So signals[i] determines position at i+1.
        signals[i] = in_pos

    return signals


class VectorizedOvernight(VectorStrategy):
    """
    Vectorized implementation of Overnight Session Mean-Reversion strategy.

    Trades the extended-hours NQ session (18:00 ET - 09:15 ET next day) on 5-min bars.
    Establishes a range during the first N minutes after session_start, then trades
    mean-reversion fades back into the range when price breaks out and reverts.

    Key design principles:
    - Max 1 trade per overnight session (robust, avoids overtrading in thin liquidity)
    - Simple rules: range formation -> breakout -> fade back into range
    - EMA trend filter to avoid fading strong directional moves
    - ATR-based SL/TP for adaptive risk management
    - Hard exit at session_end (before RTH open chaos)
    - Max 6-8 numeric parameters for anti-overfit compliance
    """
    def __init__(self, session_start="18:00", session_end="08:00",
                 range_minutes=60, ema_filter=50, atr_filter=14,
                 sl_atr_mult=2.0, tp_atr_mult=3.0, **kwargs):
        super().__init__(
            session_start=session_start, session_end=session_end,
            range_minutes=range_minutes, ema_filter=ema_filter,
            atr_filter=atr_filter, sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult, **kwargs
        )
        self.session_start = session_start
        self.session_end = session_end
        self.range_minutes = int(range_minutes)
        self.ema_filter = int(ema_filter)
        self.atr_filter = int(atr_filter)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)

    def generate_signals(self, df):
        from . import ta

        # Ensure lowercase column names
        if 'close' not in df.columns:
            df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low'})
        if 'high' not in df.columns and 'High' in df.columns:
            df = df.rename(columns={'High': 'high'})
        if 'low' not in df.columns and 'Low' in df.columns:
            df = df.rename(columns={'Low': 'low'})

        # Calculate indicators
        ema = ta.ema(df['close'], length=self.ema_filter).fillna(0).values.astype(np.float64)
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_filter).fillna(0).values.astype(np.float64)

        # Time arrays
        times = (df.index.hour * 60 + df.index.minute).values
        day_ids = np.array([d.toordinal() for d in df.index.date], dtype=np.int64)

        # Price arrays
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)

        # Parse session times
        t_start = pd.to_datetime(self.session_start).time()
        start_min = t_start.hour * 60 + t_start.minute

        t_end = pd.to_datetime(self.session_end).time()
        end_min = t_end.hour * 60 + t_end.minute

        # Range formation end = session_start + range_minutes
        range_end_min = start_min + self.range_minutes
        if range_end_min >= 1440:
            range_end_min -= 1440  # wrap past midnight

        signals = _numba_overnight_logic(
            day_ids, times, closes, highs, lows, ema, atr,
            start_min, end_min, range_end_min,
            self.sl_atr_mult, self.tp_atr_mult
        )

        return pd.Series(signals, index=df.index)


@jit(nopython=True)
def _numba_overnight_logic(day_ids, times, closes, highs, lows, ema, atr,
                           start_min, end_min, range_end_min,
                           sl_mult, tp_mult):
    """Numba-accelerated overnight session mean-reversion logic.

    Session timing: The overnight session spans across midnight.
    - session_start (e.g. 18:00 = 1080) to midnight is "evening"
    - midnight to session_end (e.g. 08:00 = 480) is "morning"

    Range formation: First range_minutes after session_start.
    Trading window: After range formation until session_end.

    Mean-reversion logic:
    - If price breaks ABOVE the overnight range high, then closes back
      BELOW range high, go SHORT (fade the failed breakout).
    - If price breaks BELOW the overnight range low, then closes back
      ABOVE range low, go LONG (fade the failed breakdown).
    - EMA filter: only fade if price is on the "reversion side" of EMA
      (short if close < ema, long if close > ema).

    Hard exit at session_end to avoid RTH open volatility.
    """
    n = len(closes)
    signals = np.zeros(n, dtype=np.int32)

    # Determine if session crosses midnight
    # If start_min > end_min, the session wraps (e.g., 18:00 -> 08:00)
    crosses_midnight = start_min > end_min

    # Similarly for range_end
    # range_end_min might have wrapped past midnight
    range_crosses_midnight = start_min > range_end_min if range_end_min < start_min else False

    # State
    range_high = -1.0
    range_low = 1e9
    traded_session = False
    in_pos = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    in_session = False
    range_formed = False
    broke_high = False  # Price broke above range high at some point
    broke_low = False   # Price broke below range low at some point

    for i in range(1, n):
        t = times[i]

        # Detect new calendar day -- reset session tracking
        if day_ids[i] != day_ids[i - 1]:
            # Force exit any held position on day boundary
            if in_pos != 0:
                in_pos = 0
            # If we were in the evening portion, a new day means we continue
            # the same overnight session into the morning.
            # But if we're entering a new evening session, reset.
            if t >= start_min:
                # New evening session starting
                range_high = -1.0
                range_low = 1e9
                traded_session = False
                in_session = True
                range_formed = False
                broke_high = False
                broke_low = False

        # Determine if current bar is in the overnight session
        if crosses_midnight:
            bar_in_session = (t >= start_min) or (t < end_min)
        else:
            bar_in_session = (t >= start_min) and (t < end_min)

        # Detect session start (first bar at or after start_min)
        if t >= start_min and t < start_min + 5 and not in_session:
            # Reset for new session
            range_high = -1.0
            range_low = 1e9
            traded_session = False
            in_session = True
            range_formed = False
            broke_high = False
            broke_low = False

        if not bar_in_session:
            # Outside session -- flatten
            if in_pos != 0:
                in_pos = 0
            in_session = False
            signals[i] = 0
            continue

        # Determine if we're in range formation window
        if crosses_midnight:
            if range_end_min > start_min:
                # Range ends before midnight (unlikely but handle)
                in_range_window = (t >= start_min) and (t < range_end_min)
            else:
                # Range crosses midnight: evening part OR morning part before range_end
                in_range_window = (t >= start_min) or (t < range_end_min)
        else:
            in_range_window = (t >= start_min) and (t < range_end_min)

        # Range Formation
        if in_range_window and not range_formed:
            if range_high == -1.0:
                range_high = highs[i]
                range_low = lows[i]
            else:
                if highs[i] > range_high:
                    range_high = highs[i]
                if lows[i] < range_low:
                    range_low = lows[i]
            signals[i] = in_pos
            continue

        # Mark range as formed once we exit range window
        if not range_formed and not in_range_window:
            range_formed = True

        # Trading Window (after range formation, before session end)
        if range_formed and bar_in_session:
            # Hard exit at session_end
            if crosses_midnight:
                near_exit = (t < start_min) and (t >= end_min - 5)
            else:
                near_exit = t >= end_min - 5
            if near_exit:
                in_pos = 0
                signals[i] = 0
                continue

            # Check exits first
            if in_pos != 0:
                if in_pos == 1:
                    # Long: check SL/TP
                    if lows[i] <= sl_price:
                        in_pos = 0
                    elif highs[i] >= tp_price:
                        in_pos = 0
                elif in_pos == -1:
                    # Short: check SL/TP
                    if highs[i] >= sl_price:
                        in_pos = 0
                    elif lows[i] <= tp_price:
                        in_pos = 0

            # Check entries (max 1 per session)
            if in_pos == 0 and not traded_session and range_high > range_low:
                cur_atr = atr[i]
                if cur_atr <= 0:
                    signals[i] = in_pos
                    continue

                # Track breakout state
                if highs[i] > range_high:
                    broke_high = True
                if lows[i] < range_low:
                    broke_low = True

                # Mean-reversion fade: price broke out then came back
                # SHORT fade: broke above range, now closing back below range_high
                if broke_high and closes[i] < range_high and closes[i] < ema[i]:
                    in_pos = -1
                    entry_price = closes[i]
                    sl_price = entry_price + (cur_atr * sl_mult)
                    tp_price = entry_price - (cur_atr * tp_mult)
                    traded_session = True
                    broke_high = False
                    broke_low = False

                # LONG fade: broke below range, now closing back above range_low
                elif broke_low and closes[i] > range_low and closes[i] > ema[i]:
                    in_pos = 1
                    entry_price = closes[i]
                    sl_price = entry_price - (cur_atr * sl_mult)
                    tp_price = entry_price + (cur_atr * tp_mult)
                    traded_session = True
                    broke_high = False
                    broke_low = False

        signals[i] = in_pos

    return signals
