
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    ta = None
from enum import Enum

class Regime(Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    CHOPPY_VOLATILE = "CHOPPY_VOLATILE"
    CHOPPY_QUIET = "CHOPPY_QUIET"
    UNKNOWN = "UNKNOWN"

class RegimeFilter:
    def __init__(self, adx_thresh=20, sma_period=50):
        self.adx_thresh = adx_thresh
        self.sma_period = sma_period

    def label_regime(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        if 'Close' in df.columns: df.rename(columns={'Close': 'close'}, inplace=True)
        if 'High' in df.columns: df.rename(columns={'High': 'high'}, inplace=True)
        if 'Low' in df.columns: df.rename(columns={'Low': 'low'}, inplace=True)
        if 'close' not in df.columns: return pd.Series(Regime.UNKNOWN, index=df.index)

        try:
            if len(df) < self.sma_period:
                return pd.Series(Regime.UNKNOWN, index=df.index)

            if ta is None:
                # Manual
                price = df['close']
                sma = price.rolling(window=self.sma_period).mean()
                
                high = df['high']
                low = df['low']
                close = df['close']
                prev_close = close.shift(1)
                
                tr1 = high - low
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                up_move = high - high.shift(1)
                down_move = low.shift(1) - low
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                
                alpha = 1.0 / 14.0
                atr = tr.ewm(alpha=alpha, adjust=False).mean()
                plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr)
                minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr)
                
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
                adx = dx.ewm(alpha=alpha, adjust=False).mean()
                
            else:
                sma = ta.sma(df['close'], length=self.sma_period)
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                adx = adx_df['ADX_14'] if adx_df is not None else pd.Series(0, index=df.index)
            
            if adx is None or sma is None:
                 return pd.Series(Regime.UNKNOWN, index=df.index)
            
            # Reindex to match df exactly to avoid alignment issues
            sma = sma.reindex(df.index)
            adx = adx.reindex(df.index)
            
            # Use values for comparison to bypass index type mismatch
            # (e.g. if one has RangeIndex and other has DateTimeIndex)
            adx_val = adx.values
            sma_val = sma.values
            close_val = df['close'].values
            
            # Create updated checks handling NaNs safely
            # We assume NaNs will result in False for comparisons usually, or runtime warning.
            # Using simple iterables or numpy select
            
            cond_bull = (adx_val > self.adx_thresh) & (close_val > sma_val)
            cond_bear = (adx_val > self.adx_thresh) & (close_val < sma_val)
            cond_chop = (adx_val <= self.adx_thresh)
            
            conditions = [cond_bull, cond_bear, cond_chop]
            choices = [Regime.BULL_TREND, Regime.BEAR_TREND, Regime.CHOPPY_QUIET]
            
            regime_codes = np.select(conditions, choices, default=Regime.UNKNOWN)
            return pd.Series(regime_codes, index=df.index)
            
        except Exception as e:
            # print(f"Regime detection error: {e}") 
            return pd.Series(Regime.UNKNOWN, index=df.index)

class MarketRegimeDetector:
    def __init__(self, adx_period: int = 14, sma_period: int = 50, vol_period: int = 20):
        self.filter = RegimeFilter(adx_thresh=20, sma_period=sma_period)

    def detect(self, df: pd.DataFrame) -> Regime:
        regimes = self.filter.label_regime(df)
        return regimes.iloc[-1] if not regimes.empty else Regime.UNKNOWN

    def get_regime_series(self, df: pd.DataFrame) -> pd.Series:
        return self.filter.label_regime(df)
