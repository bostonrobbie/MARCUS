"""
NQmain Strategy Analyzer
=========================
Parses the NQmain (NqOrbEnhanced) Pine Script to extract its coverage profile,
time windows, regime filters, and gap analysis.

This allows Marcus to understand WHERE NQmain trades and WHERE the gaps are,
so new strategies can be targeted to fill portfolio holes rather than overlap.

The original NqOrbEnhanced.pine is NEVER modified — only a read-only copy
in reference/ is used for analysis.

Coverage Map:
- NQmain: 9:45-15:45 trading window, trending regime only, max 1 trade/day
- Gap 1: Pre-market / ORB formation (9:30-9:45) — no entries
- Gap 2: Mean reversion regime — NQmain only trades WITH trend (EMA > SMA)
- Gap 3: Low volatility — NQmain requires RVOL > 1.5
- Gap 4: Lunch hour (11:30-13:30) — NQmain active but rarely triggers mid-day
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
    """Extracted coverage profile of the NQmain strategy."""

    # Time windows (ET)
    orb_session_start: dtime = dtime(9, 30)
    orb_session_end: dtime = dtime(9, 45)
    trading_window_start: dtime = dtime(9, 45)
    trading_window_end: dtime = dtime(15, 45)
    hard_exit: dtime = dtime(15, 45)

    # Regime filter
    regime: str = "trending"  # EMA(50) filter + HTF daily SMA(100)
    regime_detail: str = "Long: dailyClose > SMA(100); Short: dailyClose < SMA(100)"

    # Volatility filter
    requires_rvol: bool = True
    rvol_threshold: float = 1.5
    rvol_length: int = 20

    # Entry logic
    entry_type: str = "ORB breakout"
    max_trades_per_day: int = 1
    entry_filters: List[str] = field(default_factory=lambda: [
        "EMA(50) trend alignment",
        "HTF daily SMA(100) direction",
        "RVOL(20) > 1.5",
        "ORB range <= 2.5x ATR(14)",
    ])

    # Risk params
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 4.0
    trailing_atr_mult: float = 2.0
    use_trailing: bool = True

    # Sizing
    base_contracts: int = 1
    max_contracts: int = 3

    # Known gaps / weaknesses
    gap_windows: List[Tuple[dtime, dtime]] = field(default_factory=lambda: [
        (dtime(9, 30), dtime(9, 45)),   # ORB formation — no entries possible
        (dtime(11, 30), dtime(13, 30)), # Lunch hour — NQmain rarely triggers
        (dtime(15, 45), dtime(16, 0)),  # Post hard-exit — 15 min unprotected
    ])

    gap_regimes: List[str] = field(default_factory=lambda: [
        "mean_reversion",    # NQmain only trades WITH trend
        "choppy_range",      # NQmain needs directional conviction
        "low_volatility",    # RVOL < 1.5 filtered out
    ])

    # Portfolio constraints
    max_correlation: float = MAX_PORTFOLIO_CORRELATION
    max_portfolio_dd: float = MAX_PORTFOLIO_DD
    max_strategies: int = MAX_STRATEGIES

    def get_active_minutes(self) -> int:
        """Total minutes NQmain is actively trading."""
        start = self.trading_window_start.hour * 60 + self.trading_window_start.minute
        end = self.trading_window_end.hour * 60 + self.trading_window_end.minute
        return end - start  # 360 minutes (6 hours)

    def get_gap_minutes(self) -> int:
        """Total minutes in identified gaps."""
        total = 0
        for start, end in self.gap_windows:
            s = start.hour * 60 + start.minute
            e = end.hour * 60 + end.minute
            total += (e - s)
        return total

    def get_coverage_map(self) -> Dict[str, Dict]:
        """Generate a time-of-day coverage map showing NQmain activity vs gaps."""
        return {
            "09:30-09:45": {
                "nqmain": "ORB_FORMATION",
                "status": "GAP",
                "opportunity": "No entries — price discovery period",
            },
            "09:45-10:15": {
                "nqmain": "ACTIVE_HIGH",
                "status": "COVERED",
                "opportunity": "Primary NQmain entry zone — breakout window",
            },
            "10:15-11:30": {
                "nqmain": "ACTIVE_LOW",
                "status": "PARTIAL",
                "opportunity": "NQmain traded out, but could add First Hour Fade",
            },
            "11:30-13:30": {
                "nqmain": "ACTIVE_VERY_LOW",
                "status": "GAP",
                "opportunity": "Lunch hour — low NQmain activity, ideal for range strategies",
            },
            "13:30-14:00": {
                "nqmain": "ACTIVE_MODERATE",
                "status": "PARTIAL",
                "opportunity": "Afternoon session restart — possible overlap",
            },
            "14:00-15:30": {
                "nqmain": "ACTIVE_MODERATE",
                "status": "PARTIAL",
                "opportunity": "Power Hour — add momentum strategy if uncorrelated",
            },
            "15:30-15:45": {
                "nqmain": "ACTIVE_EXIT",
                "status": "COVERED",
                "opportunity": "Final 15 min — NQmain exits here",
            },
            "15:45-16:00": {
                "nqmain": "INACTIVE",
                "status": "GAP",
                "opportunity": "Post-exit — unprotected window",
            },
        }


# Strategy time window mapping for overlap computation
ARCHETYPE_TIME_WINDOWS = {
    'orb_breakout': (dtime(9, 45), dtime(15, 45)),    # Full ORB trading window
    'ma_crossover': (dtime(9, 30), dtime(15, 45)),     # All day
    'eod_momentum': (dtime(13, 30), dtime(15, 45)),    # Afternoon only
    'lunch_hour_breakout': (dtime(11, 0), dtime(13, 30)),  # Lunch only
    'gap_fill_fade': (dtime(9, 30), dtime(11, 0)),     # Morning only
    'power_hour_momentum': (dtime(14, 0), dtime(15, 30)),  # Final 90 min
    'first_hour_fade': (dtime(10, 15), dtime(11, 30)),  # First hour fade
    'lunch_range_fade': (dtime(11, 30), dtime(13, 30)),  # Lunch range
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
        start_str = params.get('entry_time') or params.get('range_start') or params.get('orb_start')
        end_str = params.get('exit_time') or params.get('range_end')

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
    """Compute fractional time overlap between a strategy and NQmain.

    Returns a float in [0, 1] where:
    - 0.0 = no overlap (complementary)
    - 1.0 = complete overlap (redundant)
    """
    if nqmain_windows is None:
        profile = get_nqmain_profile()
        # NQmain primary active window
        nqmain_windows = [(profile.trading_window_start, profile.trading_window_end)]

    strat_start = strategy_window[0].hour * 60 + strategy_window[0].minute
    strat_end = strategy_window[1].hour * 60 + strategy_window[1].minute
    strat_duration = strat_end - strat_start
    if strat_duration <= 0:
        return 0.0

    overlap_minutes = 0
    for nq_start_t, nq_end_t in nqmain_windows:
        nq_start = nq_start_t.hour * 60 + nq_start_t.minute
        nq_end = nq_end_t.hour * 60 + nq_end_t.minute

        overlap_start = max(strat_start, nq_start)
        overlap_end = min(strat_end, nq_end)
        if overlap_end > overlap_start:
            overlap_minutes += (overlap_end - overlap_start)

    return min(1.0, overlap_minutes / strat_duration)


def get_complementary_score(archetype: str, params: Dict = None) -> Dict:
    """Score how complementary a strategy is to NQmain.

    Returns dict with:
    - time_overlap: 0-1 (lower is better)
    - regime_complement: True if targets NQmain gap regime
    - gap_coverage: True if covers a known NQmain gap window
    - complementary_score: 0-100 (higher is better)
    """
    profile = get_nqmain_profile()
    window = get_strategy_time_window(archetype, params)
    overlap = compute_time_overlap(window, [(profile.trading_window_start, profile.trading_window_end)])

    # Check if archetype targets a gap regime
    gap_regime_map = {
        'lunch_range_fade': 'choppy_range',
        'first_hour_fade': 'mean_reversion',
        'gap_fill_fade': 'mean_reversion',
    }
    regime_complement = gap_regime_map.get(archetype, '') in profile.gap_regimes

    # Check if covers a gap window
    gap_coverage = False
    strat_start = window[0].hour * 60 + window[0].minute
    strat_end = window[1].hour * 60 + window[1].minute
    for gap_start, gap_end in profile.gap_windows:
        gs = gap_start.hour * 60 + gap_start.minute
        ge = gap_end.hour * 60 + gap_end.minute
        # Strategy covers gap if it overlaps with gap window
        if strat_start < ge and strat_end > gs:
            gap_coverage = True
            break

    # Complementary score (0-100)
    score = 0
    score += (1.0 - overlap) * 40  # Less overlap = better (40 pts)
    score += 30 if regime_complement else 0  # Gap regime = 30 pts
    score += 20 if gap_coverage else 0  # Gap window coverage = 20 pts
    score += 10 if overlap < 0.3 else 0  # Bonus for low overlap

    return {
        'time_overlap': overlap,
        'regime_complement': regime_complement,
        'gap_coverage': gap_coverage,
        'complementary_score': min(100, score),
        'strategy_window': f"{window[0].strftime('%H:%M')}-{window[1].strftime('%H:%M')}",
        'archetype': archetype,
    }
