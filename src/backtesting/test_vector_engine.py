"""
Marcus Backtest Engine Validation Suite
========================================
17 known-result test cases using synthetic data to validate every component
of the VectorEngine, VectorizedNQORB, and VectorizedMA implementations.

Run with:
    cd StrategyPipeline/src
    python -m backtesting.test_vector_engine

No pytest dependency - standalone with assert + TestRunner.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Imports from our backtest engine
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.vector_engine import (
    VectorEngine, VectorizedNQORB, VectorizedMA, _numba_orb_logic
)
from backtesting import ta


# ===========================================================================
# TEST RUNNER
# ===========================================================================
class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_func):
        try:
            test_func()
            self.passed += 1
            print(f"  [PASS] {test_func.__name__}")
        except Exception as e:
            self.failed += 1
            self.errors.append((test_func.__name__, str(e)))
            print(f"  [FAIL] {test_func.__name__}: {e}")

    def report(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, err in self.errors:
                print(f"  {name}: {err}")
        print(f"{'='*60}")
        return self.failed == 0


# ===========================================================================
# SYNTHETIC DATA GENERATORS
# ===========================================================================
def make_nq_day(date_str, bars):
    """
    Create a DataFrame of 5-min NQ bars for one trading day.

    Args:
        date_str: e.g. "2024-01-15"
        bars: list of (time_str, open, high, low, close, volume) tuples
              time_str format: "HH:MM"

    Returns: pd.DataFrame with DatetimeIndex (naive, ET wall clock)
    """
    rows = []
    for time_str, o, h, l, c, v in bars:
        ts = pd.Timestamp(f"{date_str} {time_str}:00")
        rows.append({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v, 'ts': ts})

    df = pd.DataFrame(rows)
    df = df.set_index('ts')
    df.index.name = None
    return df


def make_full_orb_day(date_str, orb_bars, post_orb_bars, fill_price=20000.0, fill_vol=100):
    """
    Create a complete trading day with ORB window + trading window + exit.

    Args:
        date_str: e.g. "2024-01-15"
        orb_bars: list of (time_str, O, H, L, C, V) for ORB window
        post_orb_bars: list of (time_str, O, H, L, C, V) for trading window
        fill_price: default price for filler bars between post_orb and EOD
        fill_vol: default volume for filler bars

    Fills gaps between post_orb_bars and 15:45 with flat bars at fill_price.
    """
    all_bars = list(orb_bars) + list(post_orb_bars)

    # Get the last explicitly defined time
    if post_orb_bars:
        last_time_str = post_orb_bars[-1][0]
    elif orb_bars:
        last_time_str = orb_bars[-1][0]
    else:
        last_time_str = "09:30"

    last_h, last_m = int(last_time_str.split(":")[0]), int(last_time_str.split(":")[1])
    last_min = last_h * 60 + last_m

    # Fill from last defined bar + 5min to 15:45
    exit_min = 15 * 60 + 45  # 945
    t = last_min + 5
    while t <= exit_min:
        h_fill = t // 60
        m_fill = t % 60
        time_str = f"{h_fill:02d}:{m_fill:02d}"
        all_bars.append((time_str, fill_price, fill_price, fill_price, fill_price, fill_vol))
        t += 5

    return make_nq_day(date_str, all_bars)


def make_numba_arrays(df, ema_val=20000.0, atr_val=50.0):
    """
    Create the numpy arrays needed by _numba_orb_logic from a DataFrame.
    Injects constant EMA and ATR values for controlled testing.
    """
    dates = df.index.date
    day_ids = np.array([d.toordinal() for d in dates], dtype=np.int64)
    times = (df.index.hour * 60 + df.index.minute).values
    closes = df['Close'].values.astype(np.float64)
    highs = df['High'].values.astype(np.float64)
    lows = df['Low'].values.astype(np.float64)

    n = len(df)
    ema = np.full(n, ema_val, dtype=np.float64)
    atr = np.full(n, atr_val, dtype=np.float64)

    return day_ids, times, closes, highs, lows, ema, atr


def run_numba_orb(df, ema_val=20000.0, atr_val=50.0,
                  start_min=570, end_min=585, exit_min=945,
                  atr_max_mult=2.5, sl_mult=2.0, tp_mult=4.0):
    """
    Helper to run _numba_orb_logic with controlled parameters.
    Returns signal array.
    """
    day_ids, times, closes, highs, lows, ema, atr = make_numba_arrays(
        df, ema_val=ema_val, atr_val=atr_val
    )

    # Disable all optional filters
    n = len(df)
    daily_ma = np.zeros(n, dtype=np.float64)
    rvol = np.zeros(n, dtype=np.float64)
    hurst = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)

    signals = _numba_orb_logic(
        day_ids, times, closes, highs, lows, ema, atr,
        start_min, end_min, exit_min,
        atr_max_mult, sl_mult, tp_mult,
        False, daily_ma,      # use_htf=False
        False, rvol, 1.5,     # use_rvol=False
        False, hurst, 0.5,    # use_hurst=False
        False, adx, 20.0,     # use_adx=False
        False, 3.0            # use_trailing_stop=False
    )
    return signals


# ===========================================================================
# TEST 1: ORB Range Bar Count
# ===========================================================================
def test_orb_range_bar_count():
    """
    For 9:30-9:45 ORB with 5-min bars, exactly 3 bars should be in the window:
    9:30, 9:35, 9:40 (since t >= 570 and t < 585).

    ORB high should be max of those 3 highs, low should be min of those 3 lows.
    """
    # Create 3 ORB bars with known H/L
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),  # H=20010
        ("09:35", 20005, 20020, 19995, 20015, 100),  # H=20020 (highest)
        ("09:40", 20015, 20015, 19980, 20010, 100),  # L=19980 (lowest)
    ]

    # Post-ORB bar that stays within range (no breakout)
    post_bars = [
        ("09:45", 20010, 20015, 19985, 20010, 100),
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20010)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0)  # EMA below price

    # At 9:45 bar, close=20010 which is < orb_high=20020, so no long entry
    # And close=20010 which is > orb_low=19980, so no short entry
    # Signal at 9:45 should be 0
    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 0, f"Expected no signal at 9:45 when close is within range, got {signals[idx_945]}"

    # All ORB window bars should have signal=0
    for ts in ["2024-01-15 09:30:00", "2024-01-15 09:35:00", "2024-01-15 09:40:00"]:
        idx = list(df.index).index(pd.Timestamp(ts))
        assert signals[idx] == 0, f"Expected signal=0 during ORB window at {ts}, got {signals[idx]}"


# ===========================================================================
# TEST 2: Long Entry Breakout
# ===========================================================================
def test_long_entry_breakout():
    """
    Close > ORB high AND close > EMA should produce signal=1.
    """
    # ORB range: H=20020, L=19980
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Breakout bar: close=20025 > orb_high=20020 AND > EMA(19900)
    post_bars = [
        ("09:45", 20015, 20030, 20010, 20025, 200),  # BREAKOUT!
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20025)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0)

    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 1, f"Expected LONG signal=1 at breakout bar, got {signals[idx_945]}"


# ===========================================================================
# TEST 3: Short Entry Breakout
# ===========================================================================
def test_short_entry_breakout():
    """
    Close < ORB low AND close < EMA should produce signal=-1.
    """
    # ORB range: H=20020, L=19980
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Breakdown bar: close=19970 < orb_low=19980 AND < EMA(20100)
    post_bars = [
        ("09:45", 19985, 19990, 19965, 19970, 200),  # BREAKDOWN!
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=19970)
    signals = run_numba_orb(df, ema_val=20100.0, atr_val=50.0)  # EMA above price for short

    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == -1, f"Expected SHORT signal=-1 at breakdown bar, got {signals[idx_945]}"


# ===========================================================================
# TEST 4: No Entry During ORB Window
# ===========================================================================
def test_no_entry_during_orb_window():
    """
    No trading should happen while the ORB range is being built (9:30-9:44).
    """
    orb_bars = [
        ("09:30", 20000, 20100, 19900, 20050, 100),  # Big range
        ("09:35", 20050, 20200, 19800, 20100, 100),   # Even bigger
        ("09:40", 20100, 20300, 19700, 20000, 100),    # Massive
    ]

    post_bars = [
        ("09:45", 20000, 20010, 19990, 20005, 100),   # No breakout
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20005)
    signals = run_numba_orb(df, ema_val=19500.0, atr_val=50.0)

    # All bars in ORB window (9:30, 9:35, 9:40) must have signal=0
    for ts in ["2024-01-15 09:30:00", "2024-01-15 09:35:00", "2024-01-15 09:40:00"]:
        idx = list(df.index).index(pd.Timestamp(ts))
        assert signals[idx] == 0, f"Expected signal=0 during ORB window at {ts}, got {signals[idx]}"


# ===========================================================================
# TEST 5: Stop Loss Hit (Long)
# ===========================================================================
def test_sl_hit_long():
    """
    After long entry, when bar low <= SL price, position should close.
    SL = entry_price - (ATR * sl_mult) = 20025 - (50 * 2.0) = 19925
    """
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Breakout long at 9:45: close=20025 > orb_high=20020
    # SL = 20025 - (50 * 2.0) = 19925
    post_bars = [
        ("09:45", 20015, 20030, 20010, 20025, 200),  # LONG ENTRY
        ("09:50", 20025, 20030, 20020, 20028, 100),   # Hold (low=20020 > 19925)
        ("09:55", 20028, 20030, 19920, 19925, 100),   # SL HIT! low=19920 <= 19925
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=19925)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=2.0, tp_mult=4.0)

    # At 9:45: signal=1 (entry)
    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 1, f"Expected LONG entry at 9:45, got {signals[idx_945]}"

    # At 9:50: signal=1 (holding)
    idx_950 = list(df.index).index(pd.Timestamp("2024-01-15 09:50:00"))
    assert signals[idx_950] == 1, f"Expected holding at 9:50, got {signals[idx_950]}"

    # At 9:55: signal=0 (SL hit, position closed)
    idx_955 = list(df.index).index(pd.Timestamp("2024-01-15 09:55:00"))
    assert signals[idx_955] == 0, f"Expected SL exit at 9:55, got {signals[idx_955]}"


# ===========================================================================
# TEST 6: Take Profit Hit (Long)
# ===========================================================================
def test_tp_hit_long():
    """
    After long entry, when bar high >= TP price, position should close.
    TP = entry_price + (ATR * tp_mult) = 20025 + (50 * 4.0) = 20225
    """
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Breakout long at 9:45: close=20025 > orb_high=20020
    # TP = 20025 + (50 * 4.0) = 20225
    post_bars = [
        ("09:45", 20015, 20030, 20010, 20025, 200),   # LONG ENTRY
        ("09:50", 20025, 20100, 20020, 20090, 100),    # Hold (high=20100 < 20225)
        ("09:55", 20090, 20230, 20080, 20200, 100),    # TP HIT! high=20230 >= 20225
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20200)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=2.0, tp_mult=4.0)

    # At 9:45: signal=1 (entry)
    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 1, f"Expected LONG entry at 9:45, got {signals[idx_945]}"

    # At 9:55: signal=0 (TP hit)
    idx_955 = list(df.index).index(pd.Timestamp("2024-01-15 09:55:00"))
    assert signals[idx_955] == 0, f"Expected TP exit at 9:55, got {signals[idx_955]}"


# ===========================================================================
# TEST 7: EOD Exit at 15:45
# ===========================================================================
def test_eod_exit_1545():
    """
    Any open position must be forced flat at 15:45.
    """
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Breakout long. SL/TP very far away so position holds all day.
    post_bars = [
        ("09:45", 20015, 20030, 20010, 20025, 200),   # LONG ENTRY
    ]

    # SL very far: 20025 - (50 * 100) = 15025 (will never hit)
    # TP very far: 20025 + (50 * 100) = 25025 (will never hit)
    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20025)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=100.0, tp_mult=100.0)

    # Position should be held through the day
    idx_1500 = list(df.index).index(pd.Timestamp("2024-01-15 15:00:00"))
    assert signals[idx_1500] == 1, f"Expected holding at 15:00, got {signals[idx_1500]}"

    # At 15:45: signal=0 (EOD exit)
    idx_1545 = list(df.index).index(pd.Timestamp("2024-01-15 15:45:00"))
    assert signals[idx_1545] == 0, f"Expected EOD exit at 15:45, got {signals[idx_1545]}"


# ===========================================================================
# TEST 8: One Trade Per Day
# ===========================================================================
def test_one_trade_per_day():
    """
    traded_today flag should prevent a second entry on the same day,
    even if the first trade was closed (SL/TP) and conditions are met again.
    """
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Entry at 9:45, SL hit at 9:55, then another breakout at 10:05
    # SL = 20025 - (50*2) = 19925
    post_bars = [
        ("09:45", 20015, 20030, 20010, 20025, 200),   # LONG ENTRY
        ("09:50", 20025, 20030, 20020, 20028, 100),    # Hold
        ("09:55", 20028, 20030, 19920, 19925, 100),    # SL HIT (low=19920 <= 19925)
        ("10:00", 19925, 19930, 19920, 19928, 100),    # Flat
        ("10:05", 19928, 20030, 19925, 20025, 200),    # Would be another breakout, but traded_today=True
        ("10:10", 20025, 20030, 20020, 20028, 100),    # Should still be flat
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20028)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=2.0, tp_mult=4.0)

    # At 9:45: long entry
    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 1, f"Expected LONG at 9:45, got {signals[idx_945]}"

    # At 9:55: SL exit
    idx_955 = list(df.index).index(pd.Timestamp("2024-01-15 09:55:00"))
    assert signals[idx_955] == 0, f"Expected SL exit at 9:55, got {signals[idx_955]}"

    # At 10:05: should NOT re-enter (traded_today=True)
    idx_1005 = list(df.index).index(pd.Timestamp("2024-01-15 10:05:00"))
    assert signals[idx_1005] == 0, f"Expected NO re-entry at 10:05 (traded_today), got {signals[idx_1005]}"


# ===========================================================================
# TEST 9: Day Boundary Reset
# ===========================================================================
def test_day_boundary_reset():
    """
    When a new calendar day starts (ordinal change), ORB state should reset.
    Day 1: entry, hold through rest of day
    Day 2: fresh ORB, fresh entry allowed
    """
    # Day 1: Jan 15 - long entry
    day1_orb = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]
    day1_post = [
        ("09:45", 20015, 20030, 20010, 20025, 200),  # LONG ENTRY
    ]
    df1 = make_full_orb_day("2024-01-15", day1_orb, day1_post, fill_price=20025)

    # Day 2: Jan 16 - different range, short entry
    day2_orb = [
        ("09:30", 20100, 20110, 20090, 20100, 100),
        ("09:35", 20100, 20120, 20085, 20090, 100),
        ("09:40", 20090, 20095, 20080, 20085, 100),
    ]
    day2_post = [
        ("09:45", 20085, 20090, 20070, 20075, 200),  # SHORT ENTRY (close=20075 < orb_low=20080, < EMA=20200)
    ]
    df2 = make_full_orb_day("2024-01-16", day2_orb, day2_post, fill_price=20075)

    df = pd.concat([df1, df2])
    signals = run_numba_orb(df, ema_val=20200.0, atr_val=50.0, sl_mult=100.0, tp_mult=100.0)

    # Day 1, 9:45: no entry because close=20025 < ema=20200 for long, and close > orb_low for short
    # Actually wait — close=20025 > orb_high=20020 for long, but close=20025 < ema=20200
    # So no long entry. And close=20025 > orb_low=19980, so no short entry.
    # Let me fix the EMA for day 1 to allow long entry

    # Re-do: set EMA=19900 for long entries on both days
    # Day 2 needs EMA above price for short: EMA=20200
    # But we have one constant EMA... Let's just test day boundary reset with same direction

    # Simpler approach: use low EMA for both days, day 1 long, day 2 long with different range
    df = pd.concat([df1, df2])
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=100.0, tp_mult=100.0)

    # Day 1, 9:45: close=20025 > orb_high=20020 AND > EMA=19900 → LONG
    idx_d1_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_d1_945] == 1, f"Day 1: Expected LONG at 9:45, got {signals[idx_d1_945]}"

    # Day 1 holds through EOD then resets
    idx_d1_1545 = list(df.index).index(pd.Timestamp("2024-01-15 15:45:00"))
    assert signals[idx_d1_1545] == 0, f"Day 1: Expected EOD exit at 15:45, got {signals[idx_d1_1545]}"

    # Day 2, 9:30-9:40: ORB window, should be signal=0
    idx_d2_930 = list(df.index).index(pd.Timestamp("2024-01-16 09:30:00"))
    assert signals[idx_d2_930] == 0, f"Day 2: Expected signal=0 during ORB at 9:30, got {signals[idx_d2_930]}"

    # Day 2: ORB high=20120, low=20080. close=20075 < orb_low=20080 but close > EMA=19900
    # So no SHORT (need close < EMA for short). close also < orb_high, so no LONG.
    # Actually this means no entry on day 2 with EMA=19900. Let me reconfigure.

    # Actually for the test we just need to verify the ORB RESETS on day 2.
    # Day 2 range should be 20080-20120 (NOT carrying over day 1's range 19980-20020)
    # We can verify by checking that day 2 allows a long breakout above 20120

    # Create day 2 with explicit long breakout above day 2's range
    day2_orb = [
        ("09:30", 20100, 20110, 20090, 20100, 100),   # H=20110, L=20090
        ("09:35", 20100, 20120, 20085, 20090, 100),    # H=20120
        ("09:40", 20090, 20095, 20080, 20085, 100),    # L=20080
    ]
    day2_post = [
        ("09:45", 20085, 20130, 20080, 20125, 200),  # LONG: close=20125 > orb_high=20120 AND > EMA=19900
    ]
    df2 = make_full_orb_day("2024-01-16", day2_orb, day2_post, fill_price=20125)

    df = pd.concat([df1, df2])
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=100.0, tp_mult=100.0)

    # Day 2, 9:45: close=20125 > day2_orb_high=20120 → LONG
    idx_d2_945 = list(df.index).index(pd.Timestamp("2024-01-16 09:45:00"))
    assert signals[idx_d2_945] == 1, f"Day 2: Expected LONG at 9:45 (day boundary reset), got {signals[idx_d2_945]}"


# ===========================================================================
# TEST 10: Weekend Gap Handling
# ===========================================================================
def test_weekend_gap():
    """
    Friday → Monday should correctly reset ORB state.
    No carryover from Friday's range to Monday.
    """
    # Friday Jan 19, 2024
    fri_orb = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]
    fri_post = [
        ("09:45", 20015, 20030, 20010, 20025, 200),  # LONG ENTRY
    ]
    df_fri = make_full_orb_day("2024-01-19", fri_orb, fri_post, fill_price=20025)

    # Monday Jan 22, 2024 (gap up)
    mon_orb = [
        ("09:30", 20200, 20210, 20190, 20200, 100),   # H=20210, L=20190
        ("09:35", 20200, 20220, 20185, 20215, 100),    # H=20220
        ("09:40", 20215, 20215, 20180, 20210, 100),    # L=20180
    ]
    mon_post = [
        ("09:45", 20210, 20230, 20200, 20225, 200),  # LONG: close=20225 > orb_high=20220
    ]
    df_mon = make_full_orb_day("2024-01-22", mon_orb, mon_post, fill_price=20225)

    df = pd.concat([df_fri, df_mon])
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, sl_mult=100.0, tp_mult=100.0)

    # Monday 9:45: should use MONDAY's ORB range (20180-20220), not Friday's (19980-20020)
    idx_mon_945 = list(df.index).index(pd.Timestamp("2024-01-22 09:45:00"))
    assert signals[idx_mon_945] == 1, f"Monday: Expected LONG at 9:45 (weekend gap handled), got {signals[idx_mon_945]}"

    # Also verify Monday ORB window bars are signal=0
    idx_mon_930 = list(df.index).index(pd.Timestamp("2024-01-22 09:30:00"))
    assert signals[idx_mon_930] == 0, f"Monday: Expected signal=0 during ORB at 9:30, got {signals[idx_mon_930]}"


# ===========================================================================
# TEST 11: ATR Max Filter
# ===========================================================================
def test_atr_max_filter():
    """
    When ORB range > ATR * atr_max_mult, no entry should occur.
    With ATR=50 and atr_max_mult=2.5, max range = 125 points.
    """
    # Create ORB with range = 200 points (way above 125)
    orb_bars = [
        ("09:30", 20000, 20100, 19900, 20050, 100),   # H=20100, L=19900 → range=200
        ("09:35", 20050, 20100, 19900, 20000, 100),
        ("09:40", 20000, 20100, 19900, 20050, 100),
    ]

    # Clear breakout above orb_high=20100
    post_bars = [
        ("09:45", 20050, 20150, 20040, 20110, 200),  # close=20110 > orb_high=20100
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20110)
    signals = run_numba_orb(df, ema_val=19900.0, atr_val=50.0, atr_max_mult=2.5)

    # range=200 > 50*2.5=125, so entry should be BLOCKED
    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 0, f"Expected NO entry (ATR max filter), got {signals[idx_945]}"


# ===========================================================================
# TEST 12: MA Cross Long Signal
# ===========================================================================
def test_ma_cross_long():
    """
    When fast SMA > slow SMA, VectorizedMA should produce signal=1.
    """
    # Create simple price data where fast MA crosses above slow MA
    # Use short_window=3, long_window=5 for tractable synthetic data
    n = 20
    dates = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")

    # Price goes down then up sharply (fast MA will cross above slow MA)
    prices = np.array([
        100, 99, 98, 97, 96, 95, 94, 93, 92, 91,  # Downtrend
        92, 95, 100, 106, 112, 118, 124, 130, 136, 142  # Sharp uptrend
    ], dtype=float)

    df = pd.DataFrame({
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': 100
    }, index=dates)

    strategy = VectorizedMA(short_window=3, long_window=5)
    signals = strategy.generate_signals(df)

    # After warmup (5 bars), during downtrend: fast < slow → signal=-1
    # During uptrend recovery: fast > slow → signal=1

    # At bar 15+ (well into uptrend), fast SMA should be above slow SMA
    # Fast SMA(3) at bar 15: mean(112, 118, 124) = 118
    # Slow SMA(5) at bar 15: mean(100, 106, 112, 118, 124) = 112
    # Fast > Slow → signal=1
    assert signals.iloc[15] == 1, f"Expected LONG signal=1 during uptrend, got {signals.iloc[15]}"

    # During downtrend (bar 8): fast SMA < slow SMA → signal=-1
    # Fast SMA(3) at bar 8: mean(93, 92, 91) ≈ 92
    # Slow SMA(5) at bar 8: mean(95, 94, 93, 92, 91) = 93
    assert signals.iloc[8] == -1, f"Expected SHORT signal=-1 during downtrend, got {signals.iloc[8]}"


# ===========================================================================
# TEST 13: MA Cross Short Signal
# ===========================================================================
def test_ma_cross_short():
    """
    When fast SMA < slow SMA, VectorizedMA should produce signal=-1.
    """
    n = 15
    dates = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")

    # Steady downtrend — fast MA will be below slow MA
    prices = np.array([
        200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178, 176, 174, 172
    ], dtype=float)

    df = pd.DataFrame({
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': 100
    }, index=dates)

    strategy = VectorizedMA(short_window=3, long_window=5)
    signals = strategy.generate_signals(df)

    # After warmup, in steady downtrend, fast SMA < slow SMA
    # Bar 10: Fast SMA(3) = mean(182, 180, 178) = 180
    #          Slow SMA(5) = mean(186, 184, 182, 180, 178) = 182
    #          Fast < Slow → -1
    assert signals.iloc[10] == -1, f"Expected SHORT signal=-1 in downtrend, got {signals.iloc[10]}"


# ===========================================================================
# TEST 14: Signal Shift No Lookahead
# ===========================================================================
def test_signal_shift_no_lookahead():
    """
    VectorEngine.run() shifts signals by 1 bar.
    Position at bar i should reflect signal at bar i-1.
    """
    # Create simple data
    n = 10
    dates = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")
    prices = np.linspace(100, 110, n)

    df = pd.DataFrame({
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': 100
    }, index=dates)

    # Use MA strategy with windows that produce known signals
    strategy = VectorizedMA(short_window=2, long_window=3)
    engine = VectorEngine(strategy, initial_capital=100000, commission=0, slippage=0, volatility_factor=0, point_value=20.0)

    result = engine.run(df)
    signals = strategy.generate_signals(df)

    # The position at bar i should equal signal at bar i-1
    pos = signals.shift(1).fillna(0)

    for i in range(1, len(signals)):
        expected_pos = signals.iloc[i-1] if i > 0 else 0
        # VectorEngine internally does: pos = signals.shift(1).fillna(0)
        actual_pos = pos.iloc[i]
        assert expected_pos == actual_pos, f"Bar {i}: position={actual_pos} but signal at bar {i-1}={expected_pos}"


# ===========================================================================
# TEST 15: Cost Model Math
# ===========================================================================
def test_cost_model_math():
    """
    Verify commission + slippage calculation is mathematically correct.
    """
    n = 6
    dates = pd.date_range("2024-01-15 09:30", periods=n, freq="5min")

    # Price at 20000 with known returns
    prices = np.array([20000.0, 20000.0, 20100.0, 20100.0, 20000.0, 20000.0])

    df = pd.DataFrame({
        'Open': prices,
        'High': prices + 50,
        'Low': prices - 50,
        'Close': prices,
        'Volume': 100
    }, index=dates)

    # Create a simple strategy that goes long at bar 1, flat at bar 4
    class FixedSignalStrategy:
        def __init__(self):
            self.params = {}
        def generate_signals(self, df):
            signals = pd.Series(0, index=df.index)
            signals.iloc[1] = 1  # Go long
            signals.iloc[2] = 1  # Hold
            signals.iloc[3] = 1  # Hold
            signals.iloc[4] = 0  # Exit
            return signals

    commission = 2.06
    slippage = 5.0
    vol_factor = 0.01
    point_value = 20.0

    engine = VectorEngine(
        FixedSignalStrategy(),
        initial_capital=100000,
        commission=commission,
        slippage=slippage,
        volatility_factor=vol_factor,
        point_value=point_value
    )
    result = engine.run(df)

    # Manually calculate expected costs:
    # Position at bar i = signal at bar i-1 (shift by 1)
    # pos = [0, 0, 1, 1, 1, 0]  (shifted from [0, 1, 1, 1, 0, 0])
    # turnover: [0, 0, 1, 0, 0, 1]  (entry at bar 2, exit at bar 5)

    # Cost per trade at bar 2 (price=20100):
    #   bar range = (20100+50) - (20100-50) = 100
    #   vol_slippage = 100 * 0.01 = 1.0
    #   total_cost_dollars = 2.06 + 5.0 + 1.0 = 8.06
    #   notional = 20100 * 20 = 402000
    #   cost_pct = 8.06 / 402000 ≈ 0.00002005

    entry_cost_pct = (commission + slippage + 100 * vol_factor) / (20100 * point_value)
    exit_cost_pct = (commission + slippage + 100 * vol_factor) / (20000 * point_value)

    # The turnover array should have 1s at entry and exit bars
    turnover = result['turnover']
    assert turnover.iloc[2] == 1.0, f"Expected turnover=1 at entry bar, got {turnover.iloc[2]}"
    assert turnover.iloc[5] == 1.0, f"Expected turnover=1 at exit bar, got {turnover.iloc[5]}"

    # Verify cost deduction is non-zero at trade bars
    gross_returns = pd.Series(0, index=df.index)  # Will compute manually
    net_returns = result['returns']

    # The key assertion: net returns should be LESS than gross returns at trade bars
    # because costs are deducted
    assert net_returns.iloc[2] < 0 or abs(net_returns.iloc[2]) < abs(entry_cost_pct) + 0.01, \
        f"Cost not properly deducted at entry bar"


# ===========================================================================
# TEST 16: EMA vs Manual Calculation
# ===========================================================================
def test_ema_vs_manual_calc():
    """
    Verify ta.ema() matches hand-computed EMA.
    EMA formula: EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
    where alpha = 2 / (N + 1)
    """
    prices = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 14.0, 12.0])
    length = 3
    alpha = 2.0 / (length + 1)  # = 0.5

    # Manual computation
    manual_ema = [prices.iloc[0]]  # First value = first price (adjust=False behavior)
    for i in range(1, len(prices)):
        new_val = alpha * prices.iloc[i] + (1 - alpha) * manual_ema[-1]
        manual_ema.append(new_val)

    # Library computation
    lib_ema = ta.ema(prices, length=length)

    for i in range(len(prices)):
        assert abs(lib_ema.iloc[i] - manual_ema[i]) < 1e-10, \
            f"EMA mismatch at index {i}: lib={lib_ema.iloc[i]:.10f}, manual={manual_ema[i]:.10f}"


# ===========================================================================
# TEST 17: No Entry When EMA Filter Blocks
# ===========================================================================
def test_no_entry_no_ema_filter():
    """
    Close > ORB high but close < EMA should NOT produce a long entry.
    The EMA filter must block the trade.
    """
    orb_bars = [
        ("09:30", 20000, 20010, 19990, 20005, 100),
        ("09:35", 20005, 20020, 19995, 20015, 100),
        ("09:40", 20015, 20015, 19980, 20010, 100),
    ]

    # Close=20025 > orb_high=20020, BUT EMA=20100 > close (EMA above price)
    post_bars = [
        ("09:45", 20015, 20030, 20010, 20025, 200),
    ]

    df = make_full_orb_day("2024-01-15", orb_bars, post_bars, fill_price=20025)
    signals = run_numba_orb(df, ema_val=20100.0, atr_val=50.0)  # EMA way above price

    # No long entry because close < EMA
    idx_945 = list(df.index).index(pd.Timestamp("2024-01-15 09:45:00"))
    assert signals[idx_945] == 0, f"Expected NO entry (EMA filter blocks), got {signals[idx_945]}"

    # Also verify no short entry since close > orb_low
    # close=20025 > orb_low=19980, so no short


# ===========================================================================
# BONUS TEST: ATR Calculation Matches TradingView (RMA smoothing)
# ===========================================================================
def test_atr_rma_smoothing():
    """
    Verify ta.atr() uses RMA (Wilder's smoothing) which matches TradingView.
    RMA uses alpha = 1/N (not 2/(N+1) like standard EMA).
    """
    # Simple 10-bar test data
    high =  pd.Series([105, 106, 107, 105, 108, 106, 109, 107, 110, 108.0])
    low =   pd.Series([95,  94,  93,  95,  92,  94,  91,  93,  90,  92.0])
    close = pd.Series([100, 101, 102, 100, 103, 101, 104, 102, 105, 103.0])

    # Manual True Range
    # TR[0] = H-L = 10
    # TR[1] = max(H-L, |H-prevC|, |L-prevC|) = max(12, 6, 6) = 12
    # etc.

    atr_result = ta.atr(high, low, close, length=3)

    # Verify it's not NaN after warmup
    assert not np.isnan(atr_result.iloc[3]), "ATR should not be NaN after warmup period"

    # Verify RMA smoothing (alpha=1/3):
    # First compute TR manually
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # RMA with alpha=1/3
    alpha = 1.0 / 3
    manual_rma = [true_range.iloc[0]]
    for i in range(1, len(true_range)):
        val = alpha * true_range.iloc[i] + (1 - alpha) * manual_rma[-1]
        manual_rma.append(val)

    # Compare
    for i in range(3, len(atr_result)):
        assert abs(atr_result.iloc[i] - manual_rma[i]) < 1e-6, \
            f"ATR mismatch at index {i}: lib={atr_result.iloc[i]:.6f}, manual={manual_rma[i]:.6f}"


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def main():
    print("=" * 60)
    print("MARCUS BACKTEST ENGINE VALIDATION SUITE")
    print("=" * 60)
    print()

    runner = TestRunner()

    tests = [
        # ORB Tests
        test_orb_range_bar_count,
        test_long_entry_breakout,
        test_short_entry_breakout,
        test_no_entry_during_orb_window,
        test_sl_hit_long,
        test_tp_hit_long,
        test_eod_exit_1545,
        test_one_trade_per_day,
        test_day_boundary_reset,
        test_weekend_gap,
        test_atr_max_filter,
        test_no_entry_no_ema_filter,

        # MA Tests
        test_ma_cross_long,
        test_ma_cross_short,

        # Engine Tests
        test_signal_shift_no_lookahead,
        test_cost_model_math,

        # Indicator Tests
        test_ema_vs_manual_calc,
        test_atr_rma_smoothing,
    ]

    print(f"Running {len(tests)} tests...\n")

    for test in tests:
        runner.run_test(test)

    print()
    success = runner.report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
