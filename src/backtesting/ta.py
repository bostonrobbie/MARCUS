
import pandas as pd
import numpy as np

"""
Lightweight Technical Analysis Library.
Replacements for pandas_ta to avoid scipy usage on Python 3.14.
"""

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Running Moving Average (Wilder's Smoothing).
    Used for RSI and ATR in TradingView.
    """
    # RMA is equivalent to EMA with alpha = 1 / length
    return series.ewm(alpha=1/length, adjust=False).mean()

def tr(key_high: pd.Series, key_low: pd.Series, key_close: pd.Series) -> pd.Series:
    """True Range"""
    prev_close = key_close.shift(1)
    tr1 = key_high - key_low
    tr2 = (key_high - prev_close).abs()
    tr3 = (key_low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range (using RMA smoothing to match TradingView)"""
    true_range = tr(high, low, close)
    return rma(true_range, length)

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (using RMA smoothing to match TradingView)"""
    delta = close.diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Average Directional Index (using RMA smoothing to match TradingView)
    Returns: adx
    """
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    pos_dm = np.where((up > down) & (up > 0), up, 0.0)
    neg_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Convert to Series for rolling
    pos_dm = pd.Series(pos_dm, index=high.index)
    neg_dm = pd.Series(neg_dm, index=high.index)
    
    # Smooth DM and TR
    atr_val = atr(high, low, close, length)
    pos_dm_s = rma(pos_dm, length)
    neg_dm_s = rma(neg_dm, length)
    
    # Calculate DI
    pos_di = 100 * (pos_dm_s / atr_val)
    neg_di = 100 * (neg_dm_s / atr_val)
    
    # Calculate DX
    denom = pos_di + neg_di
    dx = 100 * (abs(pos_di - neg_di) / denom)
    
    # Smooth DX to get ADX
    adx_val = rma(dx, length)
    return adx_val

def chop_index(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Choppiness Index (0-100).
    100 * LOG10( SUM(ATR(1), Length) / ( MaxHi(Length) - MinLo(Length) ) ) / LOG10(Length)
    """
    # 1. TR for ATR(1) is just TR.
    tr1 = tr(high, low, close)
    sum_tr = tr1.rolling(window=length).sum()
    
    # 2. Range (Max High - Min Low)
    max_hi = high.rolling(window=length).max()
    min_lo = low.rolling(window=length).min()
    range_hl = max_hi - min_lo
    
    # Avoid division by zero
    range_hl = range_hl.replace(0, np.nan) 
    
    # 3. Calculation
    # numerator = Log10(Sum(TR) / Range) -- Wait, formula is Sum(TR) / Range? 
    # TradingView: 100 * log10( sum(atr(1), length) / (highest(length) - lowest(length)) ) / log10(length)
    # Yes.
    
    x = sum_tr / range_hl
    
    # Log10 of x
    log_x = np.log10(x)
    
    # Denominator: Log10(Length)
    log_len = np.log10(length)
    
    ci = 100 * log_x / log_len
    return ci
