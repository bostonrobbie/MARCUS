"""
NQmain Strategy Analyzer
=========================
Models the NQ Main portfolio (Triple NQ Variant) to extract its coverage
profile, time windows, regime filters, and gap analysis.

The Triple NQ Variant runs 6 sub-strategies on NQ E-mini:
  1. Trend NQ     (T2) - MA crossover trend following, 9:45-15:45
  2. Long ORB     (T1) - Opening range breakout (long only), 9:45-15:45
  3. Short ORB    (T1) - Opening range breakout (short only), 9:45-15:45
  4. Simple Short (T1) - Counter-trend short, 9:30-15:45 (VIX-gated)
  5. Overnight Drift (T3) - 18:05-09:25 session mean-reversion/trend
  6. Universal Trend Edge (T2) - EMA crossover, 9:30-15:45

These trade on a SINGLE account with position_size==0 gating (only 1 trade
at a time across all 6 strategies). This means there is implicit gap coverage:
  - RTH session (09:30-15:45): heavily covered by 5 strategies
  - Overnight session (18:05-09:25): covered by Drift only

Marcus should focus on strategies that are:
  1. UNCORRELATED with the portfolio (different regime, different edge)
  2. Active during times NQmain is WEAK (not just inactive)
  3. Trading a DIFFERENT instrument or timeframe if overlapping in time

Key gaps remaining in Triple NQ Variant:
  - Mean reversion regime: All 6 sub-strats are trend/breakout/momentum
  - Low volatility: Requires RVOL>1.5 or directional conviction
  - Lunch hour (11:30-13:30): Covered in theory but rarely triggers mid-day
  - Post-close (15:45-18:05): ~2.3h dead zone
  - Weekend/holiday: No coverage
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from datetime import time as dtime

logger = logging.getLogger(__name__)

# Portfolio constraints from Fund_Manager/constants.py
MAX_PORTFOLIO_CORRELATION = 0.5
MAX_PORTFOLIO_DD = 0.20  # 20% max portfolio drawdown
MAX_STRATEGIES = 7


@dataclass
class NQmainProfile:
    """Coverage profile of the Triple NQ Variant portfolio (6 sub-strategies)."""

    # RTH time windows (ET) -- 5 of 6 strategies trade here
    rth_start: dtime = dtime(9, 30)
    rth_end: dtime = dtime(15, 45)

    # Overnight window (Drift strategy)
    overnight_start: dtime = dtime(18, 5)
    overnight_end: dtime = dtime(9, 25)

    # Sub-strategy active windows (all constrained by position_size==0)
    sub_strategies: Dict = field(default_factory=lambda: {
        "Trend_NQ":   {"start": dtime(9, 45), "end": dtime(15, 45), "type": "trend_following", "regime": "trending"},
        "Long_ORB":   {"start": dtime(9, 45), "end": dtime(15, 45), "type": "breakout", "regime": "trending"},
        "Short_ORB":  {"start": dtime(9, 45), "end": dtime(15, 45), "type": "breakout", "regime": "trending"},
        "Simple_Short": {"start": dtime(9, 30), "end": dtime(15, 45), "type": "counter_trend", "regime": "bearish"},
        "Drift":      {"start": dtime(18, 5), "end": dtime(9, 25), "type": "trend_following", "regime": "all"},
        "Universal_Trend": {"start": dtime(9, 30), "end": dtime(15, 45), "type": "trend_following", "regime": "trending"},
    })

    # Key constraint: only 1 position at a time across all 6 strategies
    max_concurrent_trades: int = 1

    # Regime coverage: ALL sub-strategies require trending/directional conviction
    # Mean-reversion regime is the primary UNCOVERED regime
    regime_types_covered: List[str] = field(default_factory=lambda: [
        "trending", "breakout", "momentum", "bearish",
    ])

    # Entry types used across the portfolio
    entry_types: List[str] = field(default_factory=lambda: [
        "ORB breakout", "MA crossover", "EMA crossover",
        "counter-trend short", "overnight drift",
    ])

    # Sizing: fixed contracts (1/2/3 tiers, overnight 1/1/2)
    base_contracts: int = 1
    max_contracts: int = 3

    # Known gaps / weaknesses in the FULL portfolio
    gap_windows: List[Tuple[dtime, dtime]] = field(default_factory=lambda: [
        (dtime(15, 45), dtime(18, 5)),  # Dead zone: RTH close to overnight start (~2.3h)
        (dtime(11, 30), dtime(13, 30)), # Lunch hour: covered but rarely triggers mid-day
        (dtime(9, 25), dtime(9, 30)),   # Tiny gap: overnight exit to RTH start (5 min)
    ])

    gap_regimes: List[str] = field(default_factory=lambda: [
        "mean_reversion",    # ALL 6 strategies are trend/breakout/momentum
        "choppy_range",      # Needs directional conviction -- range-bound = no edge
        "low_volatility",    # RVOL<1.5 or ATR too low -> filtered out
    ])

    # Portfolio constraints
    max_correlation: float = MAX_PORTFOLIO_CORRELATION
    max_portfolio_dd: float = MAX_PORTFOLIO_DD
    max_strategies: int = MAX_STRATEGIES

    def get_active_minutes(self) -> int:
        """Total minutes the portfolio is actively trading (RTH + overnight)."""
        # RTH: 09:30-15:45 = 375 min
        rth = (self.rth_end.hour * 60 + self.rth_end.minute) - \
              (self.rth_start.hour * 60 + self.rth_start.minute)
        # Overnight: 18:05-09:25 = (1440-1085) + 565 = 920 min
        on_start = self.overnight_start.hour * 60 + self.overnight_start.minute
        on_end = self.overnight_end.hour * 60 + self.overnight_end.minute
        overnight = (1440 - on_start) + on_end
        return rth + overnight  # ~1295 minutes total

    def get_gap_minutes(self) -> int:
        """Total minutes in identified gaps."""
        total = 0
        for start, end in self.gap_windows:
            s = start.hour * 60 + start.minute
            e = end.hour * 60 + end.minute
            if s > e:
                total += (1440 - s) + e
            else:
                total += (e - s)
        return total

    def get_rth_windows(self) -> List[Tuple[dtime, dtime]]:
        """Get all RTH active windows for overlap computation.

        Returns a list of windows because the portfolio has multiple
        sub-strategies that collectively cover the RTH session.
        The primary RTH window is 09:30-15:45 (covers all 5 RTH strategies).
        """
        return [(self.rth_start, self.rth_end)]

    def get_all_active_windows(self) -> List[Tuple[dtime, dtime]]:
        """Get ALL active windows including overnight.

        Used for overlap computation: a new strategy overlaps with NQmain
        if it trades during ANY of these windows.
        """
        return [
            (self.rth_start, self.rth_end),          # RTH: 09:30-15:45
            (self.overnight_start, self.overnight_end),  # Overnight: 18:05-09:25
        ]

    def get_coverage_map(self) -> Dict[str, Dict]:
        """Generate a time-of-day coverage map showing portfolio activity vs gaps."""
        return {
            "09:25-09:30": {
                "nqmain": "TRANSITION",
                "status": "GAP",
                "opportunity": "Drift exits at 09:25, RTH starts at 09:30 -- 5 min gap",
            },
            "09:30-09:45": {
                "nqmain": "ACTIVE_MODERATE",
                "status": "COVERED",
                "opportunity": "Simple Short + Universal Trend active, ORB forming",
            },
            "09:45-10:15": {
                "nqmain": "ACTIVE_HIGH",
                "status": "COVERED",
                "opportunity": "Peak entry zone: 5 strategies competing for position_size==0",
            },
            "10:15-11:30": {
                "nqmain": "ACTIVE_MODERATE",
                "status": "COVERED",
                "opportunity": "If already in trade, held. If flat, strategies can still trigger",
            },
            "11:30-13:30": {
                "nqmain": "ACTIVE_LOW",
                "status": "WEAK",
                "opportunity": "Lunch: covered in theory but volume drops, triggers rare. "
                               "Best gap for MEAN REVERSION / range-bound strategies",
            },
            "13:30-15:30": {
                "nqmain": "ACTIVE_MODERATE",
                "status": "COVERED",
                "opportunity": "Afternoon: Trend/Universal can trigger. Power hour flow",
            },
            "15:30-15:45": {
                "nqmain": "ACTIVE_EXIT",
                "status": "COVERED",
                "opportunity": "Final 15 min -- RTH strategies exit by 15:45",
            },
            "15:45-18:05": {
                "nqmain": "INACTIVE",
                "status": "GAP",
                "opportunity": "Dead zone: 2h20m with no coverage. Post-close strategies",
            },
            "18:05-09:25": {
                "nqmain": "DRIFT_ACTIVE",
                "status": "COVERED",
                "opportunity": "Overnight Drift only (1 strategy). Additional overnight "
                               "strategies MUST be uncorrelated (different entry logic)",
            },
        }


# Strategy archetype time windows for overlap computation
# These are the DEFAULT windows -- actual params may override via get_strategy_time_window()
ARCHETYPE_TIME_WINDOWS = {
    'orb_breakout': (dtime(9, 45), dtime(15, 45)),        # Full ORB trading window
    'orb_vwap': (dtime(9, 45), dtime(15, 45)),             # ORB + VWAP
    'orb_momentum': (dtime(9, 45), dtime(15, 45)),         # ORB + RSI
    'ma_crossover': (dtime(9, 30), dtime(15, 45)),         # All day
    'eod_momentum': (dtime(13, 30), dtime(15, 45)),        # Afternoon only
    'lunch_hour_breakout': (dtime(11, 0), dtime(13, 30)),  # Lunch only
    'gap_fill_fade': (dtime(9, 30), dtime(11, 0)),         # Morning only
    'es_gap_combo': (dtime(9, 30), dtime(11, 0)),          # ES gap strategies
    'power_hour_momentum': (dtime(14, 0), dtime(15, 30)),  # Final 90 min
    'first_hour_fade': (dtime(10, 15), dtime(11, 30)),     # First hour fade
    'lunch_range_fade': (dtime(11, 30), dtime(13, 30)),    # Lunch range
    'overnight': (dtime(18, 0), dtime(8, 0)),              # Overnight session (crosses midnight)
}

# Regime classification for each archetype
# Used to determine if a strategy targets an NQmain GAP regime
ARCHETYPE_REGIME = {
    'orb_breakout': 'breakout',
    'orb_vwap': 'breakout',
    'orb_momentum': 'momentum',
    'ma_crossover': 'trend_following',
    'eod_momentum': 'momentum',
    'lunch_hour_breakout': 'breakout',
    'gap_fill_fade': 'mean_reversion',       # <-- targets NQmain gap regime
    'es_gap_combo': 'mixed',
    'power_hour_momentum': 'momentum',
    'first_hour_fade': 'mean_reversion',     # <-- targets NQmain gap regime
    'lunch_range_fade': 'mean_reversion',    # <-- targets NQmain gap regime
    'overnight': 'mean_reversion',           # <-- BUT NQmain Drift already covers overnight
}


def get_nqmain_profile() -> NQmainProfile:
    """Get the NQmain strategy profile (singleton-like, cached)."""
    return NQmainProfile()


def get_strategy_time_window(archetype: str, params: Dict = None) -> Tuple[dtime, dtime]:
    """Get the time window for a given strategy archetype.

    Uses params override if available (e.g., orb_start/orb_end),
    otherwise uses archetype defaults.
    """
    if params:
        # Try to extract from params (various naming conventions)
        start_str = (params.get('session_start') or params.get('entry_time')
                     or params.get('range_start') or params.get('orb_start'))
        end_str = (params.get('session_end') or params.get('exit_time')
                   or params.get('range_end'))

        if start_str and ':' in str(start_str):
            try:
                parts = str(start_str).split(':')
                start = dtime(int(parts[0]), int(parts[1]))

                if end_str and ':' in str(end_str):
                    parts = str(end_str).split(':')
                    end = dtime(int(parts[0]), int(parts[1]))
                else:
                    # Default: 2 hours after start
                    end_h = start.hour + 2
                    end = dtime(min(end_h, 15), 45)

                return (start, end)
            except (ValueError, TypeError):
                pass

    # Fallback to archetype defaults
    return ARCHETYPE_TIME_WINDOWS.get(archetype, (dtime(9, 30), dtime(15, 45)))


def compute_time_overlap(
    strategy_window: Tuple[dtime, dtime],
    nqmain_windows: List[Tuple[dtime, dtime]] = None
) -> float:
    """Compute fractional time overlap between a strategy and NQmain portfolio.

    Returns a float in [0, 1] where:
    - 0.0 = no overlap (complementary)
    - 1.0 = complete overlap (redundant)

    Now uses the FULL portfolio coverage (RTH + overnight) instead of just
    the old single-ORB window. This means:
    - RTH strategies overlap with the 09:30-15:45 RTH window
    - Overnight strategies overlap with the 18:05-09:25 Drift window
    - The only true gap is 15:45-18:05 (post-close dead zone)

    Handles cross-midnight windows for both strategy and NQmain windows.
    """
    if nqmain_windows is None:
        profile = get_nqmain_profile()
        nqmain_windows = profile.get_all_active_windows()

    strat_start = strategy_window[0].hour * 60 + strategy_window[0].minute
    strat_end = strategy_window[1].hour * 60 + strategy_window[1].minute

    # Handle cross-midnight sessions (e.g., 18:00 -> 08:00)
    if strat_start > strat_end:
        strat_duration = (1440 - strat_start) + strat_end
    else:
        strat_duration = strat_end - strat_start

    if strat_duration <= 0:
        return 0.0

    # Convert windows to minute-sets for accurate overlap with cross-midnight handling
    def window_to_minutes(start_m: int, end_m: int) -> set:
        if start_m > end_m:
            # Cross-midnight: union of [start, 1440) and [0, end)
            return set(range(start_m, 1440)) | set(range(0, end_m))
        return set(range(start_m, end_m))

    strat_minutes = window_to_minutes(strat_start, strat_end)

    nq_minutes = set()
    for nq_start_t, nq_end_t in nqmain_windows:
        nq_start = nq_start_t.hour * 60 + nq_start_t.minute
        nq_end = nq_end_t.hour * 60 + nq_end_t.minute
        nq_minutes |= window_to_minutes(nq_start, nq_end)

    overlap_minutes = len(strat_minutes & nq_minutes)
    return min(1.0, overlap_minutes / strat_duration)


def get_complementary_score(archetype: str, params: Dict = None) -> Dict:
    """Score how complementary a strategy is to the NQmain portfolio.

    The scoring now accounts for the FULL Triple NQ Variant coverage:
    - Time overlap is computed against ALL 6 sub-strategies' windows
    - Regime complement checks if the strategy targets mean_reversion
      (the primary uncovered regime in the portfolio)
    - Gap coverage checks against the updated gap windows (post-close dead zone,
      lunch hour weakness)

    Returns dict with:
    - time_overlap: 0-1 (lower is better for pure time separation)
    - regime_complement: True if targets NQmain gap regime (mean_reversion, choppy_range)
    - gap_coverage: True if covers a known NQmain gap window
    - complementary_score: 0-100 (higher is better)
    """
    profile = get_nqmain_profile()
    window = get_strategy_time_window(archetype, params)

    # Compute time overlap against FULL portfolio (RTH + overnight)
    overlap = compute_time_overlap(window, profile.get_all_active_windows())

    # Check if archetype targets a gap REGIME (most important signal)
    strat_regime = ARCHETYPE_REGIME.get(archetype, '')
    regime_complement = strat_regime in ('mean_reversion', 'choppy_range')

    # Check if covers a gap WINDOW
    gap_coverage = False
    strat_start = window[0].hour * 60 + window[0].minute
    strat_end = window[1].hour * 60 + window[1].minute

    if strat_start > strat_end:
        # Cross-midnight strategy
        # Check if it covers the post-close dead zone (15:45-18:05)
        # A cross-midnight strategy starting before 18:05 covers some of the gap
        if strat_start < 18 * 60 + 5:
            gap_coverage = True
    else:
        for gap_start, gap_end in profile.gap_windows:
            gs = gap_start.hour * 60 + gap_start.minute
            ge = gap_end.hour * 60 + gap_end.minute
            if gs > ge:
                # Cross-midnight gap (shouldn't happen with current gaps, but handle it)
                if strat_end > gs or strat_start < ge:
                    gap_coverage = True
                    break
            else:
                if strat_start < ge and strat_end > gs:
                    gap_coverage = True
                    break

    # Complementary score (0-100)
    # Rebalanced: regime complement is now the MOST important factor because
    # the portfolio covers nearly 24h -- time gaps are small.
    # A mean-reversion strategy during RTH lunch is more valuable than an
    # overnight breakout strategy that duplicates Drift.
    score = 0

    # Time independence (25 pts) -- still useful but less dominant
    score += (1.0 - overlap) * 25

    # Regime complement (35 pts) -- THE most important factor
    # All 6 NQmain strategies are trend/breakout/momentum
    # A mean-reversion strategy is inherently uncorrelated by construction
    score += 35 if regime_complement else 0

    # Gap window coverage (20 pts)
    score += 20 if gap_coverage else 0

    # Bonus: low time overlap (10 pts)
    score += 10 if overlap < 0.5 else 0

    # Bonus: targets specific high-value gaps (10 pts)
    # The lunch hour (11:30-13:30) is the #1 weakness in NQmain
    if strat_start <= 11 * 60 + 30 and strat_end >= 13 * 60 + 30:
        score += 5  # Specifically covers lunch hour
    if strat_start >= 15 * 60 + 45 and strat_end <= 18 * 60 + 5:
        score += 5  # Specifically covers post-close dead zone

    return {
        'time_overlap': overlap,
        'regime_complement': regime_complement,
        'gap_coverage': gap_coverage,
        'complementary_score': min(100, score),
        'strategy_window': f"{window[0].strftime('%H:%M')}-{window[1].strftime('%H:%M')}",
        'archetype': archetype,
    }
