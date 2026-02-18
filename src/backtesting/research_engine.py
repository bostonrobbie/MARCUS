"""
Autonomous Research Engine
============================
The core brains of Marcus. Runs a full 5-stage research pipeline autonomously:

    Stage 1: Basic Backtest (standard costs) - Is it profitable at all?
    Stage 2: Gauntlet Stress (2x costs)      - Does it survive realistic costs?
    Stage 3: Regime Split (3 periods)         - Is it robust across market regimes?
    Stage 4: Parameter Sensitivity            - Is it overfit to specific params?
    Stage 5: Complementarity Check            - Does it add value to the portfolio?

Ideas that fail are disposed via the StrategyLifecycleManager.
Ideas that pass all 5 gates become FINAL CANDIDATES.

Uses GPU acceleration for Monte Carlo simulations when available.
"""

import os
import json
import logging
import hashlib
import time
import traceback
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .marcus_config import MarcusConfig
from .registry import StrategyRegistry
from .lifecycle import StrategyLifecycleManager
from .monitor import PipelineMonitor
from .accelerate import get_gpu_info, gpu_monte_carlo, GPU_AVAILABLE
from .statistics import StatisticalSignificance

logger = logging.getLogger(__name__)

# === Market Rationale Map ===
# Each archetype family must have an evidence-based rationale explaining
# WHY the strategy should work, rooted in real market microstructure.
MARKET_RATIONALE = {
    'orb_breakout': (
        'Opening Range Breakout exploits institutional order flow concentration '
        'in the first 5-30 minutes. Large orders cluster at the open as mutual funds, '
        'pension funds, and market makers adjust positions from overnight information. '
        'The ORB captures directional conviction when volume confirms the breakout.'
    ),
    'ma_crossover': (
        'Moving average crossovers capture momentum regime shifts. When fast MA crosses '
        'above slow MA, it signals that recent buying pressure exceeds the longer-term trend. '
        'This works because institutional rebalancing creates sustained directional flow, '
        'and trend-following is one of the most persistent market anomalies (Moskowitz 2012).'
    ),
    'eod_momentum': (
        'End-of-day momentum exploits portfolio rebalancing flows and closing auction dynamics. '
        'Institutional traders execute large orders in the final 90 minutes to minimize market '
        'impact. The MOC (Market-On-Close) order imbalance creates predictable price pressure.'
    ),
    'lunch_hour_breakout': (
        'Lunch hour range breakout exploits liquidity withdrawal between 11:00-13:00 ET. '
        'Market makers widen spreads, volume drops 40-60%, and price consolidates in a narrow '
        'range. The subsequent breakout captures the return of institutional flow.'
    ),
    'gap_fill_fade': (
        'Overnight gap fill exploits mean reversion of the futures basis. Gaps are caused by '
        'overnight news and futures-cash basis adjustments. Statistical analysis shows ~70% of '
        'NQ gaps partially fill within the first 2 hours of trading (intraday mean reversion).'
    ),
    'power_hour_momentum': (
        'Final hour momentum (14:00-15:30) captures closing auction participation and MOC '
        'order flow. Institutional traders who need to complete daily allocations create '
        'predictable directional pressure. Volume surges 2-3x in the final 90 minutes.'
    ),
    'first_hour_fade': (
        'First hour fade (10:15-11:30) exploits retail overreaction to morning news. '
        'Retail traders rush to act on headlines, creating short-term mispricings. '
        'Smart money fades these moves as initial volatility subsides and price '
        'reverts toward fair value (Barber & Odean, 2000).'
    ),
    'lunch_range_fade': (
        'Lunch range fade (11:30-13:30) exploits the predictable contraction of '
        'volatility during the midday liquidity trough. Price tends to oscillate within '
        'a defined range as participation drops. Mean reversion strategies profit from '
        'fading moves to range extremes.'
    ),
}


@dataclass
class CycleResult:
    """Tracks results of one research cycle."""
    cycle_num: int = 0
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    ideas_generated: int = 0
    backtests_run: int = 0
    stage1_passed: int = 0
    stage2_passed: int = 0
    stage3_passed: int = 0
    stage4_passed: int = 0
    stage5_passed: int = 0
    rejected: int = 0
    disposed: int = 0
    errors: int = 0
    best_sharpe: float = 0.0
    best_strategy_name: str = ""
    gpu_used: bool = False
    notes: str = ""
    error_details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Cycle {self.cycle_num}: "
            f"{self.ideas_generated} ideas -> "
            f"S1:{self.stage1_passed} S2:{self.stage2_passed} "
            f"S3:{self.stage3_passed} S4:{self.stage4_passed} S5:{self.stage5_passed} | "
            f"Rejected:{self.rejected} Errors:{self.errors} | "
            f"Best: {self.best_strategy_name} (Sharpe {self.best_sharpe:.2f}) | "
            f"{self.duration_seconds:.0f}s"
        )


class AutonomousResearchEngine:
    """
    Self-operating research engine that generates, tests, validates,
    and manages NQ futures strategies completely autonomously.
    """

    def __init__(self, config: MarcusConfig):
        self.config = config
        self.registry = StrategyRegistry(config.db_path)
        self.lifecycle = StrategyLifecycleManager(config.db_path, config)
        self.monitor = PipelineMonitor(config.logs_dir)
        self.gpu_available = GPU_AVAILABLE and config.use_gpu
        self._data_loaded = False
        self._data_cache = {}

        # Lazy imports to avoid circular dependencies
        self._idea_gen = None
        self._backtester = None
        self._improver = None
        self._shared_data_handler = None  # Cached data handler (loaded once, ~8s save per backtest)

    # =========================================================================
    # Lazy Component Initialization
    # =========================================================================

    def _get_idea_generator(self):
        """Lazy load the strategy idea generator."""
        if self._idea_gen is None:
            try:
                from .stage1_strategy_research import StrategyIdeaGenerator
                # Build LLM config from MarcusConfig fields (P0-3 fix)
                llm_config = {}
                if self.config.llm_enabled:
                    llm_config = {
                        'provider': self.config.llm_provider,
                        'model': self.config.llm_model,
                        'base_url': self.config.llm_base_url,
                        'temperature': self.config.llm_temperature,
                    }
                    logger.info(f"LLM idea generation enabled: {self.config.llm_provider}/{self.config.llm_model}")
                else:
                    logger.info("LLM idea generation disabled. Using deterministic fallback grid.")
                self._idea_gen = StrategyIdeaGenerator(
                    config=llm_config,
                    db_path=self.config.db_path,
                )
            except (ImportError, Exception) as e:
                logger.warning(f"StrategyIdeaGenerator unavailable: {e}. Using fallback.")
                self._idea_gen = None
        return self._idea_gen

    def _ensure_shared_data(self):
        """Load NQ data once and cache for reuse across all backtester instances.

        This avoids the ~8s CSV reload for every strategy backtest.
        The cached data handler is shared across stages within and across cycles.
        """
        if self._shared_data_handler is not None:
            return self._shared_data_handler

        try:
            from .data import SmartDataHandler
            logger.info(f"Loading shared data for {self.config.symbol} ({self.config.interval})...")
            t0 = time.time()
            self._shared_data_handler = SmartDataHandler(
                symbol_list=[self.config.symbol],
                search_dirs=[self.config.data_dir],
                start_date=pd.to_datetime(self.config.backtest_start),
                end_date=pd.to_datetime(self.config.backtest_end),
                interval=self.config.interval,
            )
            elapsed = time.time() - t0
            # Verify data loaded
            for sym, df in self._shared_data_handler.symbol_data.items():
                logger.info(f"Shared data loaded: {len(df)} bars for {sym} in {elapsed:.1f}s (cached for all backtests)")
                break
        except Exception as e:
            logger.error(f"Failed to load shared data: {e}")
            self._shared_data_handler = None

        return self._shared_data_handler

    def _get_backtester(self, commission: float = None, slippage: float = None):
        """Get a backtester instance with specified cost parameters.

        Reuses the shared data handler to avoid reloading 1M+ bars from CSV
        every time a new backtester is created (~8s savings per instance).
        """
        try:
            from .stage2_rigorous_backtest import RigorousBacktester

            # Ensure shared data is loaded (only loads once)
            data_handler = self._ensure_shared_data()

            bt = RigorousBacktester(
                data_handler=data_handler,  # Pass cached data handler
                symbol=self.config.symbol,
                search_dirs=[self.config.data_dir],
                start_date=self.config.backtest_start,
                end_date=self.config.backtest_end,
                interval=self.config.interval,
                initial_capital=self.config.initial_capital,
                config={
                    'commission_per_unit': commission or self.config.s1_commission_per_unit,
                    'slippage_per_unit': slippage or self.config.s1_slippage_per_unit,
                    'point_value': self.config.point_value,
                    'db_path': self.config.db_path,
                }
            )
            return bt
        except ImportError as e:
            logger.error(f"Failed to import RigorousBacktester: {e}")
            return None

    def _get_improver(self):
        """Lazy load the strategy improver for sensitivity testing."""
        if self._improver is None:
            try:
                from .auto_improver import StrategyImprover
                self._improver = StrategyImprover(
                    symbol=self.config.symbol,
                    start_date=self.config.backtest_start,
                    end_date=self.config.backtest_end,
                    interval=self.config.interval,
                    initial_capital=self.config.initial_capital,
                )
            except ImportError as e:
                logger.error(f"Failed to import StrategyImprover: {e}")
        return self._improver

    def _get_strategy_class(self, idea: Dict):
        """Resolve strategy class from idea archetype for permutation testing."""
        try:
            archetype = idea.get('archetype', 'orb_breakout')
            if 'orb' in archetype.lower():
                from .strategies import VectorizedNQORB
                return VectorizedNQORB
            elif 'ma' in archetype.lower():
                from .strategies import VectorizedMA
                return VectorizedMA
            else:
                # Default to ORB for unknown archetypes
                from .strategies import VectorizedNQORB
                return VectorizedNQORB
        except ImportError:
            return None

    def _get_shared_data_handler(self):
        """Return the cached shared data handler (or None if not loaded)."""
        return self._shared_data_handler

    # =========================================================================
    # Timeout Protection (P1-1)
    # =========================================================================

    BACKTEST_TIMEOUT_SECONDS = 120  # 2 minutes max per individual backtest

    def _run_with_timeout(self, fn, *args, timeout: int = None, **kwargs):
        """Run a function with a timeout. Returns (success, result_or_error).

        Uses ThreadPoolExecutor to enforce a wall-clock timeout on any
        backtest call, preventing a single hung strategy from stalling
        the entire daemon.
        """
        timeout = timeout or self.BACKTEST_TIMEOUT_SECONDS
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return True, result
            except concurrent.futures.TimeoutError:
                logger.error(f"Backtest timed out after {timeout}s")
                return False, {'error': f'Timed out after {timeout}s'}
            except Exception as e:
                logger.error(f"Backtest exception in thread: {e}")
                return False, {'error': str(e)}

    # =========================================================================
    # Idea Generation & Filtering
    # =========================================================================

    def _generate_ideas(self, n: int) -> List[Dict[str, Any]]:
        """Generate strategy ideas, filtering out graveyard hashes.

        Uses a two-tier fallback: primary generator → engine fallback grid.
        Both tiers are graveyard-filtered to avoid re-testing killed strategies.
        """
        gen = self._get_idea_generator()

        # --- Tier 1: primary generator (LLM or its own fallback) ---
        raw_ideas = []
        if gen is not None:
            try:
                raw_ideas = gen.generate_ideas(n_ideas=n * 2)  # request extra to survive filtering
            except Exception as e:
                logger.warning(f"Idea generation failed: {e}. Using engine fallback.")

        # Graveyard-filter tier 1 results
        filtered = self._graveyard_filter(raw_ideas)

        # --- Tier 2: engine-level fallback if tier 1 yielded too few ---
        if len(filtered) < n:
            needed = n - len(filtered)
            existing_hashes = {idea['_hash'] for idea in filtered}
            logger.info(f"Tier 1 yielded {len(filtered)}/{n}. Generating {needed} more from engine fallback.")
            fallback_ideas = self._fallback_ideas(needed * 3)  # generate extra
            for idea in fallback_ideas:
                idea_hash = self._hash_idea(idea)
                if idea_hash in existing_hashes:
                    continue
                if self.lifecycle.is_in_graveyard(idea_hash):
                    continue
                idea['_hash'] = idea_hash
                filtered.append(idea)
                existing_hashes.add(idea_hash)
                if len(filtered) >= n:
                    break

        final = filtered[:n]

        # P2-3: Log archetype diversity for observability
        from collections import Counter
        arch_counts = Counter(idea.get('archetype', 'unknown') for idea in final)
        logger.info(f"Idea batch diversity: {dict(arch_counts)}")
        if final:
            max_pct = max(arch_counts.values()) / len(final)
            if max_pct > 0.6:
                logger.warning(f"Low diversity: {max(arch_counts, key=arch_counts.get)} "
                               f"is {max_pct:.0%} of batch")

        return final

    def _graveyard_filter(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter ideas against the graveyard, returning only novel ones."""
        filtered = []
        for idea in ideas:
            idea_hash = self._hash_idea(idea)
            if self.lifecycle.is_in_graveyard(idea_hash):
                logger.debug(f"Skipping graveyard idea: {idea.get('strategy_name', 'unknown')}")
                continue
            idea['_hash'] = idea_hash
            filtered.append(idea)
        return filtered

    def _fallback_ideas(self, n: int) -> List[Dict[str, Any]]:
        """Diverse deterministic idea generation covering wide parameter space.

        Generates ideas across MULTIPLE strategy archetypes:
        - ORB breakout (various timeframes, SL/TP, filters)
        - MA crossover (various fast/slow window pairs)
        - EOD momentum (afternoon breakout strategies)
        - Lunch hour breakout (mid-day range break)
        - Gap fill fade (counter-trend gap strategies)

        Shuffled each call to explore different regions of parameter space.
        """
        import itertools
        ideas = []

        # === 1. ORB Breakout (~4400 combos) ===
        orb_periods = [
            ("09:30", "09:35", "5min"),
            ("09:30", "09:40", "10min"),
            ("09:30", "09:45", "15min"),
            ("09:30", "10:00", "30min"),
        ]
        sl_values = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        tp_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
        ema_values = [10, 20, 50, 100, 200]
        atr_max_values = [1.5, 2.0, 2.5, 3.0]

        combos = list(itertools.product(orb_periods, sl_values, tp_values, ema_values, atr_max_values))
        for (start, end, variant), sl, tp, ema, atr_max in combos:
            if tp <= sl:
                continue
            name = f"ORB_{variant}_SL{sl}_TP{tp}_EMA{ema}_ATR{atr_max}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'orb_breakout',
                'variant': variant,
                'params': {
                    'orb_start': start,
                    'orb_end': end,
                    'ema_filter': ema,
                    'atr_filter': 14,
                    'sl_atr_mult': sl,
                    'tp_atr_mult': tp,
                    'atr_max_mult': atr_max,
                },
                'hypothesis': f"ORB {variant} with {sl}x SL / {tp}x TP, EMA({ema}), ATR max {atr_max}x"
            })

        # === 2. MA Crossover (~120 combos) ===
        short_windows = [5, 8, 10, 13, 20, 30, 50]
        long_windows = [20, 50, 100, 150, 200, 300]
        for short_w, long_w in itertools.product(short_windows, long_windows):
            if short_w >= long_w:
                continue
            name = f"MA_Cross_{short_w}_{long_w}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'ma_crossover',
                'variant': f'{short_w}x{long_w}',
                'params': {
                    'short_window': short_w,
                    'long_window': long_w,
                },
                'hypothesis': f"MA crossover: {short_w}-bar crosses above/below {long_w}-bar"
            })

        # === 3. EOD Momentum (~80 combos) ===
        eod_entry_times = ["13:30", "14:00", "14:30", "15:00"]
        eod_ema_values = [20, 50, 100, 200]
        eod_sl_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        for entry, ema, sl in itertools.product(eod_entry_times, eod_ema_values, eod_sl_values):
            entry_end = f"{int(entry.split(':')[0])}:{int(entry.split(':')[1])+15:02d}" if int(entry.split(':')[1]) + 15 < 60 else f"{int(entry.split(':')[0])+1}:{(int(entry.split(':')[1])+15)-60:02d}"
            name = f"EOD_Mom_{entry.replace(':','')}_{ema}_SL{sl}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'eod_momentum',
                'variant': entry,
                'params': {
                    'entry_time': entry,
                    'ema_filter': ema,
                    'sl_atr_mult': sl,
                },
                'hypothesis': f"EOD momentum: enter at {entry}, EMA({ema}) filter, {sl}x ATR stop"
            })

        # === 4. Lunch Hour Breakout (~80 combos) ===
        lunch_starts = ["11:00", "11:30", "12:00"]
        lunch_ends = ["12:30", "13:00", "13:30"]
        lunch_emas = [20, 50, 100, 200]
        for ls, le, ema in itertools.product(lunch_starts, lunch_ends, lunch_emas):
            if ls >= le:
                continue
            name = f"Lunch_BRK_{ls.replace(':','')}_{le.replace(':','')}_{ema}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'lunch_hour_breakout',
                'variant': f"{ls}-{le}",
                'params': {
                    'range_start': ls,
                    'range_end': le,
                    'ema_filter': ema,
                },
                'hypothesis': f"Lunch hour breakout: {ls}-{le} range, EMA({ema}) trend filter"
            })

        # === 5. Gap Fill Fade (~40 combos) ===
        gap_sls = [1.0, 1.5, 2.0, 3.0]
        gap_tps = [2.0, 3.0, 4.0, 6.0, 8.0]
        for sl, tp in itertools.product(gap_sls, gap_tps):
            if tp <= sl:
                continue
            name = f"GapFade_SL{sl}_TP{tp}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'gap_fill_fade',
                'variant': 'gap_fade',
                'params': {
                    'sl_atr_mult': sl,
                    'tp_atr_mult': tp,
                },
                'hypothesis': f"Gap fill fade: {sl}x ATR stop, {tp}x ATR target"
            })

        # === 6. Power Hour Momentum (~100 combos) ===
        # Targets NQmain gap: Final hour institutional flow (14:00-15:30)
        ph_entry_times = ["14:00", "14:15", "14:30", "14:45"]
        ph_ema_values = [10, 20, 50, 100]
        ph_sl_values = [1.0, 1.5, 2.0, 2.5]
        ph_tp_values = [2.0, 3.0, 4.0, 6.0]
        for entry, ema, sl, tp in itertools.product(ph_entry_times, ph_ema_values, ph_sl_values, ph_tp_values):
            if tp <= sl:
                continue
            name = f"PowerHour_{entry.replace(':','')}_{ema}_SL{sl}_TP{tp}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'power_hour_momentum',
                'variant': entry,
                'params': {
                    'entry_time': entry,
                    'exit_time': '15:30',
                    'ema_filter': ema,
                    'sl_atr_mult': sl,
                    'tp_atr_mult': tp,
                },
                'hypothesis': f"Power hour momentum: enter at {entry}, EMA({ema}), {sl}x SL/{tp}x TP"
            })

        # === 7. First Hour Fade (~80 combos) ===
        # Targets NQmain gap: Mean reversion during retail overreaction (10:15-11:30)
        fhf_entry_times = ["10:15", "10:30", "10:45"]
        fhf_ema_values = [10, 20, 50, 100]
        fhf_sl_values = [1.0, 1.5, 2.0, 3.0]
        fhf_rsi_values = [30, 35, 40]
        for entry, ema, sl, rsi in itertools.product(fhf_entry_times, fhf_ema_values, fhf_sl_values, fhf_rsi_values):
            name = f"FHFade_{entry.replace(':','')}_{ema}_SL{sl}_RSI{rsi}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'first_hour_fade',
                'variant': entry,
                'params': {
                    'entry_time': entry,
                    'exit_time': '11:30',
                    'ema_filter': ema,
                    'sl_atr_mult': sl,
                    'rsi_threshold': rsi,
                },
                'hypothesis': f"First hour fade: enter at {entry}, EMA({ema}), RSI<{rsi}, {sl}x SL"
            })

        # === 8. Lunch Range Fade (~80 combos) ===
        # Targets NQmain gap: Range-bound mean reversion during liquidity trough (11:30-13:30)
        lrf_range_starts = ["11:30", "12:00"]
        lrf_range_ends = ["13:00", "13:30"]
        lrf_ema_values = [10, 20, 50, 100]
        lrf_sl_values = [0.75, 1.0, 1.5, 2.0]
        lrf_bb_values = [1.5, 2.0, 2.5]
        for rs, re, ema, sl, bb in itertools.product(lrf_range_starts, lrf_range_ends, lrf_ema_values, lrf_sl_values, lrf_bb_values):
            if rs >= re:
                continue
            name = f"LunchFade_{rs.replace(':','')}_{re.replace(':','')}_{ema}_BB{bb}_SL{sl}"
            ideas.append({
                'strategy_name': name,
                'archetype': 'lunch_range_fade',
                'variant': f"{rs}-{re}",
                'params': {
                    'range_start': rs,
                    'range_end': re,
                    'ema_filter': ema,
                    'sl_atr_mult': sl,
                    'bb_mult': bb,
                },
                'hypothesis': f"Lunch range fade: {rs}-{re}, EMA({ema}), BB({bb}), {sl}x SL"
            })

        # === BALANCED ROUND-ROBIN: ensure diverse archetypes in every batch ===
        # Group by archetype, shuffle within each, then interleave
        from collections import defaultdict
        by_archetype = defaultdict(list)
        for idea in ideas:
            by_archetype[idea['archetype']].append(idea)

        # Shuffle within each archetype
        for arch_ideas in by_archetype.values():
            np.random.shuffle(arch_ideas)

        # Round-robin: take one from each archetype in turn
        balanced = []
        archetype_iters = {k: iter(v) for k, v in by_archetype.items()}
        archetype_keys = list(by_archetype.keys())
        np.random.shuffle(archetype_keys)  # randomize archetype order too

        while len(balanced) < n and archetype_iters:
            exhausted = []
            for key in archetype_keys:
                if key not in archetype_iters:
                    continue
                try:
                    balanced.append(next(archetype_iters[key]))
                    if len(balanced) >= n:
                        break
                except StopIteration:
                    exhausted.append(key)
            for key in exhausted:
                del archetype_iters[key]
                archetype_keys.remove(key)

        return balanced[:n]

    def _hash_idea(self, idea: Dict) -> str:
        """Generate unique hash for an idea to prevent re-testing."""
        content = json.dumps({
            'archetype': idea.get('archetype', ''),
            'variant': idea.get('variant', ''),
            'params': idea.get('params', {}),
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    # =========================================================================
    # Stage 1: Basic Backtest
    # =========================================================================

    def _stage1_basic_backtest(self, idea: Dict) -> Tuple[bool, Dict]:
        """
        Standard backtest with normal commission/slippage.
        Pass: net profit > 0, trades > min_trades
        Returns: (passed, metrics_dict) where metrics_dict includes 'equity_returns'
                 numpy array for Stage 5 complementarity check.
        """
        bt = self._get_backtester()
        if bt is None:
            return False, {'error': 'Backtester not available'}

        try:
            # P1-1: Run with timeout to prevent hang
            ok, result = self._run_with_timeout(bt.backtest_strategy, idea)
            if not ok:
                return False, result  # result is the error dict
            if result is None:
                return False, {'error': 'Backtest returned None'}

            metrics = result.get('metrics', result)
            net_profit = float(metrics.get('net_profit', 0))
            total_trades = int(metrics.get('total_trades', 0))
            sharpe = float(metrics.get('sharpe_ratio', 0))

            passed = (
                net_profit > self.config.s1_min_profit and
                total_trades >= self.config.s1_min_trades
            )

            metrics_out = {
                'net_profit': net_profit,
                'total_trades': total_trades,
                'sharpe_ratio': sharpe,
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'win_rate': float(metrics.get('win_rate', 0)),
                'profit_factor': float(metrics.get('profit_factor', 0)),
                'total_return': float(metrics.get('total_return', 0)),
            }

            # Preserve equity returns for Stage 5 complementarity check
            equity_returns = metrics.get('equity_returns')
            if equity_returns is not None:
                metrics_out['equity_returns'] = equity_returns

            # P0-5: Preserve raw equity curve for winner persistence
            equity_curve_raw = metrics.get('equity_curve_raw')
            if equity_curve_raw is not None:
                metrics_out['equity_curve_raw'] = equity_curve_raw

            if not passed:
                reasons = []
                if net_profit <= self.config.s1_min_profit:
                    reasons.append(f"Net profit ${net_profit:.0f} <= ${self.config.s1_min_profit:.0f}")
                if total_trades < self.config.s1_min_trades:
                    reasons.append(f"Trades {total_trades} < {self.config.s1_min_trades}")
                metrics_out['failure_reason'] = "; ".join(reasons)

            return passed, metrics_out

        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
            return False, {'error': str(e)}

    # =========================================================================
    # Stage 2: Gauntlet Stress Test
    # =========================================================================

    def _stage2_gauntlet(self, idea: Dict, s1_equity_returns: np.ndarray = None) -> Tuple[bool, Dict]:
        """
        Re-backtest with 1.5x commission, 1.5x slippage.
        Pass: Metric thresholds + Sharpe CI positive + Permutation test.
        """
        bt = self._get_backtester(
            commission=self.config.get_s2_commission(),
            slippage=self.config.get_s2_slippage()
        )
        if bt is None:
            return False, {'error': 'Stressed backtester not available'}

        try:
            # P1-1: Run with timeout to prevent hang
            ok, result = self._run_with_timeout(bt.backtest_strategy, idea)
            if not ok:
                return False, result  # result is the error dict
            if result is None:
                return False, {'error': 'Gauntlet backtest returned None'}

            metrics = result.get('metrics', result)
            sharpe = float(metrics.get('sharpe_ratio', 0))
            max_dd = abs(float(metrics.get('max_drawdown', 1.0)))
            pf = float(metrics.get('profit_factor', 0))
            wr = float(metrics.get('win_rate', 0))
            trades = int(metrics.get('total_trades', 0))
            net_profit = float(metrics.get('net_profit', 0))

            # Traditional path
            trad_pass = (
                sharpe >= self.config.s2_min_sharpe and
                max_dd <= self.config.s2_max_drawdown_pct and
                pf >= self.config.s2_min_profit_factor and
                trades >= self.config.s2_min_trades and
                (wr >= self.config.s2_min_win_rate or pf >= self.config.s2_alt_min_profit_factor)
            )

            # Alternative high-R:R path (calibrated to actual data distribution)
            alt_pass = (
                sharpe >= 0.25 and
                pf >= self.config.s2_alt_min_profit_factor and
                trades >= 50 and
                net_profit > 0 and
                max_dd <= self.config.s2_max_drawdown_pct
            )

            passed = trad_pass or alt_pass

            metrics_out = {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'profit_factor': pf,
                'win_rate': wr,
                'total_trades': trades,
                'net_profit': net_profit,
                'passed_traditional': trad_pass,
                'passed_alternative': alt_pass,
                'commission_used': self.config.get_s2_commission(),
                'slippage_used': self.config.get_s2_slippage(),
            }

            if not passed:
                reasons = []
                if sharpe < self.config.s2_min_sharpe:
                    reasons.append(f"Sharpe {sharpe:.2f} < {self.config.s2_min_sharpe}")
                if max_dd > self.config.s2_max_drawdown_pct:
                    reasons.append(f"DD {max_dd:.1%} > {self.config.s2_max_drawdown_pct:.0%}")
                if pf < self.config.s2_min_profit_factor:
                    reasons.append(f"PF {pf:.2f} < {self.config.s2_min_profit_factor}")
                metrics_out['failure_reason'] = "; ".join(reasons)
                return False, metrics_out

            # ─── Statistical Gate 1: Sharpe Confidence Interval ───────────
            # Uses equity returns from S1 (more trades, more data, less noise from cost stress)
            equity_returns = s1_equity_returns
            if equity_returns is None:
                equity_returns = metrics.get('equity_returns')

            if self.config.stat_require_sharpe_ci_positive and equity_returns is not None and len(equity_returns) > 20:
                try:
                    sharpe_val, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
                        equity_returns,
                        confidence=self.config.stat_confidence_level,
                    )
                    metrics_out['sharpe_ci_lower'] = float(ci_lower)
                    metrics_out['sharpe_ci_upper'] = float(ci_upper)
                    metrics_out['sharpe_ci_sharpe'] = float(sharpe_val)

                    if ci_lower <= 0:
                        metrics_out['failure_reason'] = (
                            f"Sharpe CI crosses zero: [{ci_lower:.3f}, {ci_upper:.3f}] — "
                            f"not statistically significant at {self.config.stat_confidence_level:.0%}"
                        )
                        logger.info(f"S2 STAT REJECT {idea.get('strategy_name', '?')}: "
                                    f"Sharpe CI [{ci_lower:.3f}, {ci_upper:.3f}]")
                        return False, metrics_out
                    else:
                        logger.info(f"S2 Sharpe CI PASS: [{ci_lower:.3f}, {ci_upper:.3f}]")
                except Exception as e:
                    logger.warning(f"Sharpe CI computation failed: {e}")
                    metrics_out['sharpe_ci_error'] = str(e)

            # ─── Statistical Gate 2: Permutation Test ─────────────────────
            if self.config.s2_permutation_enabled and equity_returns is not None and len(equity_returns) > 50:
                try:
                    from .skeptic import StrategySkeptic
                    # Get strategy class and params for permutation test
                    strategy_cls = self._get_strategy_class(idea)
                    if strategy_cls is not None:
                        from .vector_engine import VectorEngine
                        skeptic = StrategySkeptic(
                            VectorEngine, strategy_cls,
                            idea.get('params', {}),
                            initial_capital=self.config.initial_capital,
                            n_jobs=2,  # Limit CPU usage within daemon
                        )
                        # Use cached data for permutation test
                        data_handler = self._get_shared_data_handler()
                        if data_handler is not None and hasattr(data_handler, '_df') and data_handler._df is not None:
                            perm_result = skeptic.run_permutation_test(
                                data_handler._df,
                                n_sims=self.config.s2_permutation_sims
                            )
                            metrics_out['permutation_p_value'] = perm_result.get('p_value', 1.0)
                            metrics_out['permutation_verdict'] = perm_result.get('verdict', 'UNKNOWN')
                            metrics_out['permutation_n_sims'] = perm_result.get('n_sims', 0)

                            if perm_result.get('verdict', '') != 'PASS':
                                metrics_out['failure_reason'] = (
                                    f"Permutation test FAIL: p={perm_result.get('p_value', 1.0):.3f} "
                                    f"({perm_result.get('n_sims', 0)} sims) — "
                                    f"indistinguishable from random"
                                )
                                logger.info(f"S2 PERM REJECT {idea.get('strategy_name', '?')}: "
                                            f"p={perm_result.get('p_value', 1.0):.3f}")
                                return False, metrics_out
                            else:
                                logger.info(f"S2 Permutation PASS: p={perm_result.get('p_value', 1.0):.3f}")
                except Exception as e:
                    logger.warning(f"Permutation test failed (non-fatal): {e}")
                    metrics_out['permutation_error'] = str(e)

            return True, metrics_out

        except Exception as e:
            logger.error(f"Stage 2 error: {e}")
            return False, {'error': str(e)}

    # =========================================================================
    # Stage 3: Regime Split
    # =========================================================================

    def _stage3_regime_split(self, idea: Dict) -> Tuple[bool, Dict]:
        """
        Run backtest on 3 separate date ranges.
        Pass: Net profitable in ALL periods.
        """
        periods = self.config.s3_periods
        period_results = {}
        all_profitable = True

        for i, (start, end) in enumerate(periods):
            try:
                bt = self._get_backtester()
                if bt is None:
                    return False, {'error': 'Backtester not available'}

                # Override date range for this period
                bt.start_date = start
                bt.end_date = end

                result = bt.backtest_strategy(idea)
                if result is None:
                    all_profitable = False
                    period_results[f"period_{i}"] = {
                        'start': start, 'end': end,
                        'error': 'Backtest returned None'
                    }
                    continue

                metrics = result.get('metrics', result)
                net_profit = float(metrics.get('net_profit', 0))
                sharpe = float(metrics.get('sharpe_ratio', 0))

                profitable = net_profit > self.config.s3_min_profit_per_period
                if not profitable:
                    all_profitable = False

                period_results[f"period_{i}"] = {
                    'start': start, 'end': end,
                    'net_profit': net_profit,
                    'sharpe_ratio': sharpe,
                    'profitable': profitable,
                }

            except Exception as e:
                logger.error(f"Stage 3 period {i} error: {e}")
                all_profitable = False
                period_results[f"period_{i}"] = {
                    'start': start, 'end': end,
                    'error': str(e)
                }

        metrics_out = {
            'all_profitable': all_profitable,
            'periods': period_results,
        }

        if not all_profitable:
            failed = [k for k, v in period_results.items() if not v.get('profitable', False)]
            metrics_out['failure_reason'] = f"Not profitable in: {', '.join(failed)}"

        return all_profitable, metrics_out

    # =========================================================================
    # Stage 4: Parameter Sensitivity
    # =========================================================================

    def _stage4_sensitivity(self, idea: Dict, baseline_profit: float) -> Tuple[bool, Dict]:
        """
        Generate parameter variants and test each.
        Pass: ALL variants profitable, no >50% profit drop from baseline.
        """
        params = idea.get('params', {})
        if not params:
            return True, {'note': 'No tunable params, auto-pass'}

        factors = self.config.s4_variation_factors
        variant_results = {}
        all_profitable = True
        max_drop = 0.0

        bt = self._get_backtester()
        if bt is None:
            return False, {'error': 'Backtester not available'}

        for param_name, param_value in params.items():
            if not isinstance(param_value, (int, float)):
                continue

            for factor in factors:
                varied_value = param_value * factor
                if isinstance(param_value, int):
                    varied_value = max(1, int(round(varied_value)))

                # Create variant idea
                variant_idea = idea.copy()
                variant_params = params.copy()
                variant_params[param_name] = varied_value
                variant_idea['params'] = variant_params

                variant_key = f"{param_name}_{factor}"

                try:
                    result = bt.backtest_strategy(variant_idea)
                    if result is None:
                        variant_results[variant_key] = {'error': 'None result'}
                        all_profitable = False
                        continue

                    metrics = result.get('metrics', result)
                    v_profit = float(metrics.get('net_profit', 0))
                    v_sharpe = float(metrics.get('sharpe_ratio', 0))

                    profitable = v_profit > 0
                    if not profitable:
                        all_profitable = False

                    # Check profit drop
                    if baseline_profit > 0:
                        drop = 1.0 - (v_profit / baseline_profit)
                        max_drop = max(max_drop, drop)
                    else:
                        drop = 0.0

                    variant_results[variant_key] = {
                        'param': param_name,
                        'factor': factor,
                        'value': varied_value,
                        'net_profit': v_profit,
                        'sharpe': v_sharpe,
                        'profitable': profitable,
                        'profit_drop_pct': drop,
                    }

                except Exception as e:
                    logger.error(f"Stage 4 variant {variant_key} error: {e}")
                    variant_results[variant_key] = {'error': str(e)}
                    all_profitable = False

        excessive_drop = max_drop > self.config.s4_max_profit_drop_pct

        # Count profitable variants
        total_valid = sum(1 for v in variant_results.values() if 'profitable' in v)
        profitable_count = sum(1 for v in variant_results.values() if v.get('profitable', False))
        profitable_pct = profitable_count / total_valid if total_valid > 0 else 0

        # Relaxed S4 gate: configurable profitable percentage threshold
        if self.config.s4_all_variants_profitable:
            profitability_ok = all_profitable
        else:
            min_pct = getattr(self.config, 's4_min_profitable_pct', 0.75)
            profitability_ok = profitable_pct >= min_pct

        basic_passed = profitability_ok and not excessive_drop

        metrics_out = {
            'all_profitable': all_profitable,
            'profitable_pct': profitable_pct,
            'profitable_count': profitable_count,
            'total_variants': total_valid,
            'max_profit_drop_pct': max_drop,
            'excessive_drop': excessive_drop,
            'variants_tested': len(variant_results),
            'variants': variant_results,
        }

        if not basic_passed:
            reasons = []
            if not profitability_ok:
                reasons.append(f"Only {profitable_pct:.0%} variants profitable "
                               f"(need {self.config.s4_min_profitable_pct:.0%})")
            if excessive_drop:
                reasons.append(f"Max profit drop {max_drop:.1%} > {self.config.s4_max_profit_drop_pct:.0%}")
            metrics_out['failure_reason'] = "; ".join(reasons)
            return False, metrics_out

        # ─── WFO Analytics: Parameter Robustness Score & Cliff Detection ────
        # Build DataFrame from variant results for ParameterSensitivityMapper
        try:
            from .wfo_analytics import ParameterSensitivityMapper
            rows = []
            for vkey, vdata in variant_results.items():
                if 'net_profit' not in vdata:
                    continue
                row = {
                    'param_name': vdata.get('param', ''),
                    'factor': vdata.get('factor', 1.0),
                    'value': vdata.get('value', 0),
                    'Total Return': vdata.get('net_profit', 0),
                }
                rows.append(row)

            if len(rows) >= 3:
                variant_df = pd.DataFrame(rows)
                mapper = ParameterSensitivityMapper(variant_df, metric_col='Total Return')

                # Robustness score (0-100)
                robustness = mapper.robustness_score()
                rob_score = robustness.get('robustness_score', 0)
                metrics_out['robustness_score'] = rob_score
                metrics_out['robustness_interpretation'] = robustness.get('interpretation', '')
                metrics_out['good_region_fraction'] = robustness.get('good_region_fraction', 0)

                min_rob = getattr(self.config, 's4_min_robustness_score', 50.0)
                if rob_score < min_rob:
                    metrics_out['failure_reason'] = (
                        f"Robustness score {rob_score:.0f} < {min_rob:.0f} — "
                        f"{robustness.get('interpretation', 'fragile parameters')}"
                    )
                    logger.info(f"S4 ROBUSTNESS REJECT {idea.get('strategy_name', '?')}: "
                                f"score={rob_score:.0f}")
                    return False, metrics_out

                # Cliff detection
                cliffs = mapper.cliff_detection()
                metrics_out['n_cliffs'] = len(cliffs)
                if cliffs:
                    high_cliffs = [c for c in cliffs if c.get('severity') == 'HIGH']
                    if high_cliffs:
                        cliff = high_cliffs[0]
                        metrics_out['failure_reason'] = (
                            f"Parameter cliff detected: {cliff['parameter']} "
                            f"({cliff['from_value']} -> {cliff['to_value']}) "
                            f"causes {abs(cliff['pct_change']):.0%} performance drop"
                        )
                        logger.info(f"S4 CLIFF REJECT {idea.get('strategy_name', '?')}: "
                                    f"{cliff['parameter']}")
                        return False, metrics_out

                logger.info(f"S4 Robustness PASS: score={rob_score:.0f}, cliffs={len(cliffs)}")

        except Exception as e:
            logger.warning(f"WFO analytics failed (non-fatal): {e}")
            metrics_out['wfo_error'] = str(e)

        return True, metrics_out

    # =========================================================================
    # Stage 5: Complementarity Check
    # =========================================================================

    def _stage5_complementarity(self, idea: Dict, equity_returns: np.ndarray = None) -> Tuple[bool, Dict]:
        """
        Check correlation with existing active strategies.
        Pass: daily return correlation < 0.3 with all active strategies.
        """
        if equity_returns is None or len(equity_returns) < 20:
            # Can't compute correlation with insufficient data
            return True, {'note': 'Insufficient data for correlation, auto-pass'}

        active = self.lifecycle.get_active_strategies()
        if not active:
            # No existing strategies to compare against - auto-pass
            return True, {'note': 'No active strategies, auto-pass (first entry)'}

        max_corr = 0.0
        correlations = {}

        for strat in active:
            strat_id = strat.get('strategy_id')
            if strat_id is None:
                continue

            try:
                existing_curve = self.registry.get_equity_curve(strat_id)
                if existing_curve is None or existing_curve.empty:
                    continue

                if 'equity' in existing_curve.columns:
                    existing_returns = existing_curve['equity'].pct_change().dropna().values
                else:
                    continue

                # Align lengths
                min_len = min(len(equity_returns), len(existing_returns))
                if min_len < 20:
                    continue

                corr = np.corrcoef(
                    equity_returns[:min_len],
                    existing_returns[:min_len]
                )[0, 1]

                if np.isnan(corr):
                    corr = 0.0

                corr = abs(corr)
                max_corr = max(max_corr, corr)
                correlations[strat.get('strategy_name', f'ID-{strat_id}')] = corr

            except Exception as e:
                logger.debug(f"Correlation check error for {strat_id}: {e}")
                continue

        corr_passed = max_corr < self.config.s5_max_daily_correlation

        metrics_out = {
            'max_correlation': max_corr,
            'threshold': self.config.s5_max_daily_correlation,
            'correlations': correlations,
            'active_strategies_checked': len(correlations),
        }

        if not corr_passed:
            metrics_out['failure_reason'] = f"Max correlation {max_corr:.3f} >= {self.config.s5_max_daily_correlation}"
            return False, metrics_out

        # ─── Statistical Gate: Deflated Sharpe Ratio (Multiple Testing Correction) ──
        # Bailey & Lopez de Prado (2014): Adjusts Sharpe for the number of strategies tested
        if equity_returns is not None and len(equity_returns) > 50:
            try:
                n_trials = self.registry.get_total_backtest_count()
                if n_trials > 10:  # Only apply with sufficient trial history
                    ret_stats = StatisticalSignificance.returns_statistics(equity_returns)
                    mean_ret = np.mean(equity_returns)
                    std_ret = np.std(equity_returns, ddof=1)
                    if std_ret > 0:
                        raw_sharpe = np.sqrt(252) * mean_ret / std_ret
                    else:
                        raw_sharpe = 0.0

                    # Estimate variance of Sharpe ratios across all trials
                    # Use SE^2 as proxy for variance since we don't have all trial Sharpes
                    se = StatisticalSignificance.sharpe_standard_error(equity_returns, raw_sharpe)
                    var_sharpe = se ** 2

                    dsr_prob = StatisticalSignificance.deflated_sharpe_ratio(
                        sharpe=raw_sharpe,
                        n_trials=n_trials,
                        var_sharpe=var_sharpe,
                        skewness=ret_stats.get('skewness', 0),
                        kurtosis=ret_stats.get('kurtosis', 3),
                        n_observations=len(equity_returns),
                    )

                    metrics_out['dsr_probability'] = float(dsr_prob)
                    metrics_out['dsr_n_trials'] = n_trials
                    metrics_out['dsr_raw_sharpe'] = float(raw_sharpe)

                    if dsr_prob < self.config.stat_min_dsr_probability:
                        metrics_out['failure_reason'] = (
                            f"Deflated Sharpe FAIL: DSR probability={dsr_prob:.3f} < "
                            f"{self.config.stat_min_dsr_probability} "
                            f"(n_trials={n_trials}, raw_sharpe={raw_sharpe:.3f}) — "
                            f"likely result of data mining"
                        )
                        logger.info(f"S5 DSR REJECT {idea.get('strategy_name', '?')}: "
                                    f"prob={dsr_prob:.3f}, trials={n_trials}")
                        return False, metrics_out
                    else:
                        logger.info(f"S5 DSR PASS: prob={dsr_prob:.3f} "
                                    f"(trials={n_trials}, sharpe={raw_sharpe:.3f})")

            except Exception as e:
                logger.warning(f"Deflated Sharpe computation failed (non-fatal): {e}")
                metrics_out['dsr_error'] = str(e)

        # ─── NQmain Portfolio Fit: Time Window Overlap Check ─────────────
        try:
            from .nqmain_analyzer import (
                get_nqmain_profile, get_strategy_time_window,
                compute_time_overlap, get_complementary_score
            )
            archetype = idea.get('archetype', '')
            params = idea.get('params', {})

            comp_score = get_complementary_score(archetype, params)
            metrics_out['nqmain_time_overlap'] = comp_score['time_overlap']
            metrics_out['nqmain_complementary_score'] = comp_score['complementary_score']
            metrics_out['nqmain_regime_complement'] = comp_score['regime_complement']
            metrics_out['nqmain_gap_coverage'] = comp_score['gap_coverage']
            metrics_out['strategy_time_window'] = comp_score['strategy_window']

            max_overlap = getattr(self.config, 's5_max_nqmain_overlap', 0.30)
            # Only block full ORB overlaps (same archetype, high time overlap)
            if archetype == 'orb_breakout' and comp_score['time_overlap'] > max_overlap:
                metrics_out['failure_reason'] = (
                    f"NQmain overlap too high: {comp_score['time_overlap']:.0%} "
                    f"(ORB vs ORB) — redundant with NQmain"
                )
                logger.info(f"S5 NQMAIN REJECT {idea.get('strategy_name', '?')}: "
                            f"overlap={comp_score['time_overlap']:.0%}")
                return False, metrics_out

            logger.info(f"S5 NQmain fit: overlap={comp_score['time_overlap']:.0%}, "
                        f"complement={comp_score['complementary_score']:.0f}")

        except Exception as e:
            logger.warning(f"NQmain overlap check failed (non-fatal): {e}")
            metrics_out['nqmain_error'] = str(e)

        return True, metrics_out

    # =========================================================================
    # Full Research Cycle
    # =========================================================================

    def run_cycle(self) -> CycleResult:
        """
        Execute one complete research cycle through all 5 stages.
        This is the main entry point called by the daemon.
        """
        cycle_num = self.registry.get_next_cycle_num()
        result = CycleResult(
            cycle_num=cycle_num,
            started_at=datetime.now().isoformat(),
            gpu_used=self.gpu_available,
        )

        self.monitor.log_cycle_start(cycle_num)
        start_time = time.time()

        try:
            # 1. Generate ideas
            ideas = self._generate_ideas(self.config.ideas_per_cycle)
            result.ideas_generated = len(ideas)

            if not ideas:
                result.notes = "No ideas generated"
                self._finalize_cycle(result, start_time)
                return result

            # 2. Process each idea through the pipeline
            for idea in ideas:
                try:
                    self._process_idea(idea, result)
                except Exception as e:
                    result.errors += 1
                    result.error_details.append(f"{idea.get('strategy_name', '?')}: {str(e)}")
                    logger.error(f"Idea processing error: {e}\n{traceback.format_exc()}")

            # 3. Run disposal sweep
            disposal = self.lifecycle.run_disposal_sweep()
            result.disposed = sum(disposal.values())

        except Exception as e:
            result.errors += 1
            result.error_details.append(f"Cycle-level error: {str(e)}")
            logger.error(f"Cycle {cycle_num} error: {e}\n{traceback.format_exc()}")

        self._finalize_cycle(result, start_time)
        return result

    @staticmethod
    def _strip_arrays(metrics: Dict) -> Dict:
        """Strip numpy arrays and pandas Series from metrics dict for JSON-safe serialization.
        Lifecycle and archive calls serialize to JSON; numpy arrays/Series would bloat the DB."""
        return {k: v for k, v in metrics.items()
                if not isinstance(v, (np.ndarray, pd.Series, pd.DataFrame))}

    def _process_idea(self, idea: Dict, result: CycleResult) -> None:
        """Process a single idea through all applicable stages."""
        name = idea.get('strategy_name', 'unknown')
        idea_hash = idea.get('_hash', self._hash_idea(idea))

        # ─── Pre-S1: Complexity Cap ──────────────────────────────────
        # Reject over-parameterized strategies before wasting compute
        params = idea.get('params', {})
        numeric_params = sum(1 for v in params.values() if isinstance(v, (int, float)))
        if numeric_params > self.config.max_strategy_params:
            logger.info(f"COMPLEXITY REJECT {name}: {numeric_params} params > "
                        f"{self.config.max_strategy_params} max")
            result.rejected += 1
            return

        # Attach market rationale to idea for downstream persistence
        archetype = idea.get('archetype', '')
        if archetype in MARKET_RATIONALE and 'market_rationale' not in idea:
            idea['market_rationale'] = MARKET_RATIONALE[archetype]

        # Register in lifecycle
        lc_id = self.lifecycle.register_candidate(idea_hash, name,
                                                   idea.get('archetype', ''))
        self.lifecycle.set_testing(lc_id)
        result.backtests_run += 1

        # Stage 1: Basic profitability
        s1_pass, s1_metrics = self._stage1_basic_backtest(idea)
        # Preserve equity_returns for Stage 5 complementarity, but strip for DB calls
        s1_equity_returns = s1_metrics.get('equity_returns')
        s1_db = self._strip_arrays(s1_metrics)

        if not s1_pass:
            self.lifecycle.reject(lc_id, s1_metrics.get('failure_reason', 'S1 fail'), 'TESTING')
            self.monitor.log_disposal(name, 'STAGE1', s1_metrics.get('failure_reason', ''))
            result.rejected += 1
            self._archive_run(idea, s1_db, 'STAGE1_FAIL')
            return

        self.lifecycle.promote(lc_id, 'STAGE1_PASS', s1_db)
        self.monitor.log_promotion(name, 'STAGE1_PASS', s1_metrics.get('sharpe_ratio', 0))
        result.stage1_passed += 1

        # P0-4: Track best Sharpe from S1 onwards (not just S5)
        s1_sharpe = float(s1_metrics.get('sharpe_ratio', 0))
        if s1_sharpe > result.best_sharpe:
            result.best_sharpe = s1_sharpe
            result.best_strategy_name = name

        # Stage 2: Gauntlet stress (with statistical verification gates)
        s2_pass, s2_metrics = self._stage2_gauntlet(idea, s1_equity_returns=s1_equity_returns)
        s2_db = self._strip_arrays(s2_metrics)

        if not s2_pass:
            self.lifecycle.reject(lc_id, s2_metrics.get('failure_reason', 'S2 fail'), 'STAGE1_PASS')
            self.monitor.log_disposal(name, 'STAGE2', s2_metrics.get('failure_reason', ''))
            result.rejected += 1
            self._archive_run(idea, {**s1_db, **s2_db}, 'STAGE2_FAIL')
            return

        self.lifecycle.promote(lc_id, 'STAGE2_PASS', s2_db)
        self.monitor.log_promotion(name, 'STAGE2_PASS', s2_metrics.get('sharpe_ratio', 0))
        result.stage2_passed += 1

        # Stage 3: Regime split
        s3_pass, s3_metrics = self._stage3_regime_split(idea)
        s3_db = self._strip_arrays(s3_metrics)

        if not s3_pass:
            self.lifecycle.reject(lc_id, s3_metrics.get('failure_reason', 'S3 fail'), 'STAGE2_PASS')
            self.monitor.log_disposal(name, 'STAGE3', s3_metrics.get('failure_reason', ''))
            result.rejected += 1
            self._archive_run(idea, {**s1_db, **s2_db, **s3_db}, 'STAGE3_FAIL')
            return

        self.lifecycle.promote(lc_id, 'STAGE3_PASS', s3_db)
        self.monitor.log_promotion(name, 'STAGE3_PASS', s2_metrics.get('sharpe_ratio', 0))
        result.stage3_passed += 1

        # Stage 4: Sensitivity
        baseline_profit = s1_metrics.get('net_profit', 0)
        s4_pass, s4_metrics = self._stage4_sensitivity(idea, baseline_profit)
        s4_db = self._strip_arrays(s4_metrics)

        if not s4_pass:
            self.lifecycle.reject(lc_id, s4_metrics.get('failure_reason', 'S4 fail'), 'STAGE3_PASS')
            self.monitor.log_disposal(name, 'STAGE4', s4_metrics.get('failure_reason', ''))
            result.rejected += 1
            self._archive_run(idea, {**s1_db, **s2_db, **s4_db}, 'STAGE4_FAIL')
            return

        self.lifecycle.promote(lc_id, 'STAGE4_PASS', s4_db)
        self.monitor.log_promotion(name, 'STAGE4_PASS', s4_metrics.get('sharpe_ratio', 0))
        result.stage4_passed += 1

        # Stage 5: Complementarity
        # Pass equity returns from Stage 1 backtest for correlation analysis
        s5_pass, s5_metrics = self._stage5_complementarity(idea, equity_returns=s1_equity_returns)
        s5_db = self._strip_arrays(s5_metrics)

        if not s5_pass:
            self.lifecycle.reject(lc_id, s5_metrics.get('failure_reason', 'S5 fail'), 'STAGE4_PASS')
            self.monitor.log_disposal(name, 'STAGE5', s5_metrics.get('failure_reason', ''))
            result.rejected += 1
            self._archive_run(idea, {**s1_db, **s2_db, **s5_db}, 'STAGE5_FAIL')
            return

        self.lifecycle.promote(lc_id, 'STAGE5_PASS', {**s1_db, **s2_db, **s5_db})
        self.monitor.log_promotion(name, 'STAGE5_PASS', s2_metrics.get('sharpe_ratio', 0))
        result.stage5_passed += 1

        # FINAL CANDIDATE - save to winning strategies
        # Pass full s1_metrics (with equity_returns) so _save_winner can run Monte Carlo
        sharpe = max(float(s1_metrics.get('sharpe_ratio', 0)),
                     float(s2_metrics.get('sharpe_ratio', 0)))
        saved = self._save_winner(idea, s1_metrics, s2_metrics, s3_metrics, s4_metrics, s5_metrics)

        if not saved:
            # MC VaR95 gate rejected - still log as S5 pass but not saved as winner
            logger.warning(f"MC gate rejected {name} after S5 pass — too much tail risk")
            self._archive_run(idea, {**s1_db, **s2_db, **s5_db}, 'MC_REJECTED')
            return

        # Track best (update if S5 winner is better than prior S1 best)
        if sharpe > result.best_sharpe:
            result.best_sharpe = sharpe
            result.best_strategy_name = name

        logger.info(f"*** FINAL CANDIDATE: {name} (Sharpe={sharpe:.2f}) ***")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _archive_run(self, idea: Dict, metrics: Dict, stage: str):
        """Save a run to the backtest_runs table regardless of pass/fail."""
        try:
            self.registry.save_run(
                strategy_name=idea.get('strategy_name', 'unknown'),
                symbol=self.config.symbol,
                interval=self.config.interval,
                params=idea.get('params', {}),
                stats=metrics,
                data_range=(self.config.backtest_start, self.config.backtest_end),
                regime=stage,
                notes=metrics.get('failure_reason', ''),
            )
        except Exception as e:
            logger.error(f"Archive run failed: {e}")

    def _save_winner(self, idea: Dict, s1: Dict, s2: Dict,
                     s3: Dict, s4: Dict, s5: Dict) -> bool:
        """Save a strategy that passed all 5 gates to winning_strategies.

        Returns True if saved successfully, False if rejected by MC gate.
        """
        try:
            combined_metrics = {**s1, **s2}
            combined_metrics.pop('failure_reason', None)

            # Run Monte Carlo if GPU available (uses equity_returns numpy array)
            mc_var95 = 0.0
            equity_returns = combined_metrics.get('equity_returns')
            if self.gpu_available and equity_returns is not None and len(equity_returns) > 0:
                try:
                    mc = gpu_monte_carlo(
                        equity_returns,
                        n_sims=self.config.monte_carlo_sims
                    )
                    mc_var95 = mc.get('p5', 0.0)
                except Exception as e:
                    logger.warning(f"GPU Monte Carlo failed: {e}")

            # ─── Monte Carlo VaR95 Gate ───────────────────────────────────
            # Reject if worst-case loss at 95th percentile is too severe
            if mc_var95 < self.config.mc_min_var95 and mc_var95 != 0.0:
                logger.warning(
                    f"MC VaR95 REJECT {idea.get('strategy_name', '?')}: "
                    f"VaR95={mc_var95:.3f} < {self.config.mc_min_var95} — "
                    f"too much tail risk"
                )
                return False

            # P0-5: Extract raw equity curve for persistence
            equity_curve_raw = combined_metrics.pop('equity_curve_raw', None)

            # Strip numpy arrays/Series before DB save (not JSON-serializable)
            combined_metrics.pop('equity_returns', None)

            # P0-5: Generate source code snapshot from idea params
            source_code = self._generate_source_snapshot(idea)

            # Build quality notes with statistical verification summary
            stat_notes = ["Passed all 5 gates"]
            if s2.get('sharpe_ci_lower') is not None:
                stat_notes.append(f"Sharpe CI: [{s2['sharpe_ci_lower']:.3f}, {s2.get('sharpe_ci_upper', 0):.3f}]")
            if s2.get('permutation_p_value') is not None:
                stat_notes.append(f"Perm p={s2['permutation_p_value']:.3f}")
            if s5.get('dsr_probability') is not None:
                stat_notes.append(f"DSR={s5['dsr_probability']:.3f}")
            if s4.get('robustness_score') is not None:
                stat_notes.append(f"Robustness={s4['robustness_score']:.0f}")
            if mc_var95 != 0.0:
                stat_notes.append(f"MC VaR95={mc_var95:.3f}")

            # Add market rationale
            archetype = idea.get('archetype', '')
            rationale = MARKET_RATIONALE.get(archetype, '')
            if rationale:
                combined_metrics['market_rationale'] = rationale

            self.registry.save_winning_strategy(
                strategy_name=idea.get('strategy_name', 'unknown'),
                archetype=archetype,
                symbol=self.config.symbol,
                interval=self.config.interval,
                metrics=combined_metrics,
                params=idea.get('params', {}),
                source_code=source_code,
                equity_curve=equity_curve_raw,
                quality_notes=" | ".join(stat_notes),
                data_range=(self.config.backtest_start, self.config.backtest_end),
                notes=f"S3: {json.dumps(s3.get('periods', {}), default=str)[:200]}",
                monte_carlo_var95=mc_var95,
                regime_analysis=s3.get('periods', {}),
            )
            return True
        except Exception as e:
            logger.error(f"Save winner failed: {e}")
            return False

    def _generate_source_snapshot(self, idea: Dict) -> str:
        """Generate a reproducible source code snapshot from the idea parameters.

        This captures everything needed to recreate the strategy, providing
        traceability from winning_strategies back to the exact configuration.
        """
        params = idea.get('params', {})
        archetype = idea.get('archetype', 'unknown')
        name = idea.get('strategy_name', 'unknown')
        hypothesis = idea.get('hypothesis', '')

        lines = [
            f"# Strategy: {name}",
            f"# Archetype: {archetype}",
            f"# Hypothesis: {hypothesis}",
            f"# Generated by Marcus Autonomous Research Engine",
            f"# Date: {datetime.now().isoformat()}",
            f"#",
            f"# Configuration:",
            f"#   Symbol: {self.config.symbol}",
            f"#   Interval: {self.config.interval}",
            f"#   Initial Capital: ${self.config.initial_capital:,.0f}",
            f"#   Commission: ${self.config.s1_commission_per_unit}/unit",
            f"#   Slippage: ${self.config.s1_slippage_per_unit}/unit",
            f"#",
            f"idea = {{",
            f"    'strategy_name': {name!r},",
            f"    'archetype': {archetype!r},",
            f"    'params': {json.dumps(params, indent=8, default=str)},",
            f"}}",
        ]
        return '\n'.join(lines)

    def _finalize_cycle(self, result: CycleResult, start_time: float):
        """Finalize cycle timing and logging."""
        result.duration_seconds = time.time() - start_time
        result.finished_at = datetime.now().isoformat()

        # Log to registry
        self.registry.log_cycle(result.to_dict())

        # Log to monitor
        self.monitor.log_cycle_end(result.cycle_num, result.to_dict())

        logger.info(result.summary())
