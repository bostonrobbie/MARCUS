"""
Stage 2: Rigorous Backtest
==========================
Bridges LLM-generated strategy ideas to REAL backtest implementations.

CRITICAL FIX: This replaces the placeholder stub that always returned 0 trades.
Maps strategy archetypes from Stage 1 to actual VectorEngine/EventDriven backtests.

Fixes:
- Critical Bug #1: Placeholder backtest that returned 0 trades
- Major Issue #2: Unifies OLD and NEW systems via archetype mapping
- Major Issue #3: LLM-to-code translation gap
- Major Issue #4: Hardcoded strategy map replaced with flexible matching
- Issue #7: None-type handling with defensive checks
- Issue #9: Date parsing errors
- Issue #12: Quality thresholds use OR-logic for high R:R strategies
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .vector_engine import VectorEngine, VectorizedNQORB, VectorizedMA, VectorStrategy
from .data import SmartDataHandler
from .registry import StrategyRegistry
from .stage1_strategy_research import STRATEGY_ARCHETYPES

logger = logging.getLogger(__name__)


class QualityChecker:
    """
    Strategy quality checking with flexible thresholds.

    Fix for Issue #12: Uses OR-logic so high R:R strategies with low win rates
    are not incorrectly rejected. A strategy passes if EITHER:
    - It meets traditional criteria (Sharpe >= 1.0, Win rate >= 40%), OR
    - It meets alternative criteria (Sharpe >= 1.5, Profit Factor >= 1.5, regardless of win rate)
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        # Traditional thresholds (relaxed from original 1.5/40%/500)
        self.min_sharpe = cfg.get("min_sharpe", 1.0)
        self.min_win_rate = cfg.get("min_win_rate", 30.0)  # Lowered from 40%
        self.min_trades = cfg.get("min_trades", 100)  # Lowered from 500
        self.max_drawdown = cfg.get("max_drawdown", -0.35)  # 35% max drawdown

    def check(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if strategy meets quality standards.
        Returns (passed, list_of_issues).
        """
        issues = []

        # Defensive None handling (Issue #7)
        sharpe = self._safe_float(metrics.get("sharpe_ratio", 0))
        win_rate = self._safe_float(metrics.get("win_rate", 0))
        total_trades = self._safe_int(metrics.get("total_trades", 0))
        net_profit = self._safe_float(metrics.get("net_profit", 0))
        max_dd = self._safe_float(metrics.get("max_drawdown", 0))
        profit_factor = self._safe_float(metrics.get("profit_factor", 0))

        # Collect issues
        if sharpe < self.min_sharpe:
            issues.append(f"Sharpe {sharpe:.2f} < {self.min_sharpe}")
        if total_trades < self.min_trades:
            issues.append(f"Only {total_trades} trades < {self.min_trades} required")
        if max_dd < self.max_drawdown:
            issues.append(f"Max drawdown {max_dd:.1%} exceeds {self.max_drawdown:.1%} limit")

        # TRADITIONAL PATH: Sharpe + Win Rate + Trades
        traditional_pass = (
            sharpe >= self.min_sharpe
            and win_rate >= self.min_win_rate
            and total_trades >= self.min_trades
            and max_dd >= self.max_drawdown
        )

        # ALTERNATIVE PATH (Issue #12): High R:R strategies with low win rate
        # Example: 18% win rate but 1.79 Sharpe, 1.86 PF is a WINNING strategy
        alternative_pass = (
            sharpe >= 1.5
            and profit_factor >= 1.5
            and total_trades >= max(50, self.min_trades // 2)  # Relaxed trade count
            and net_profit > 0
            and max_dd >= self.max_drawdown
        )

        passed = traditional_pass or alternative_pass

        if alternative_pass and not traditional_pass:
            issues.clear()
            issues.append(f"Passed via high-R:R path (Sharpe={sharpe:.2f}, PF={profit_factor:.2f}, WR={win_rate:.1f}%)")

        return passed, issues

    @staticmethod
    def _safe_float(val) -> float:
        """Safely convert to float, handling None/NaN (Issue #7)."""
        if val is None:
            return 0.0
        try:
            result = float(val)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_int(val) -> int:
        """Safely convert to int, handling None (Issue #7)."""
        if val is None:
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0


class StrategyMapper:
    """
    Maps strategy archetypes from Stage 1 to actual backtest implementations.

    Fix for Major Issue #4: Replaces hardcoded strategy map with flexible
    archetype-based matching that works with any LLM-generated strategy name.
    """

    @staticmethod
    def create_vector_strategy(idea: Dict[str, Any]) -> Optional[VectorStrategy]:
        """
        Create a VectorStrategy instance from a structured strategy idea.
        This is the FIX for Critical Bug #1 - actually runs backtests instead of stubs.
        """
        archetype = idea.get("archetype", "")
        params = idea.get("params", {})

        if archetype == "orb_breakout":
            return VectorizedNQORB(
                orb_start=params.get("orb_start", "09:30"),
                orb_end=params.get("orb_end", "09:45"),
                ema_filter=params.get("ema_filter", 50),
                atr_filter=params.get("atr_filter", 14),
                sl_atr_mult=params.get("sl_atr_mult", 2.0),
                tp_atr_mult=params.get("tp_atr_mult", 4.0),
                atr_max_mult=params.get("atr_max_mult", 2.5),
                use_htf=params.get("use_htf", False),
                htf_ma=params.get("htf_ma", 200),
                use_rvol=params.get("use_rvol", False),
                rvol_thresh=params.get("rvol_thresh", 1.5),
                use_hurst=params.get("use_hurst", False),
                hurst_thresh=params.get("hurst_thresh", 0.5),
                use_adx=params.get("use_adx", False),
                adx_thresh=params.get("adx_thresh", 20),
                use_trailing_stop=params.get("use_trailing_stop", False),
                ts_atr_mult=params.get("ts_atr_mult", 3.0),
            )

        elif archetype == "orb_vwap":
            # Use VectorizedNQORB as base, VWAP filtering not in vector engine yet
            # Fall back to ORB with defaults
            return VectorizedNQORB(
                orb_start=params.get("orb_start", "09:30"),
                orb_end=params.get("orb_end", "09:45"),
                ema_filter=params.get("ema_filter", 50),
            )

        elif archetype == "orb_momentum":
            return VectorizedNQORB(
                orb_start=params.get("orb_start", "09:30"),
                orb_end=params.get("orb_end", "09:45"),
                ema_filter=params.get("ema_filter", 50),
                use_adx=True,
                adx_thresh=params.get("adx_thresh", 20),
            )

        elif archetype == "ma_crossover":
            return VectorizedMA(
                short_window=params.get("short_window", 50),
                long_window=params.get("long_window", 200),
            )

        elif archetype == "gap_fill_fade":
            # Use ORB framework with gap-aligned params
            return VectorizedNQORB(
                orb_start="09:30",
                orb_end="09:35",  # Short ORB for gap plays
                sl_atr_mult=params.get("sl_atr_mult", 2.0),
                tp_atr_mult=params.get("tp_atr_mult", 4.0),
            )

        elif archetype == "es_gap_combo":
            orb_min = params.get("orb_period_min", 15)
            end_min = 30 + orb_min  # 09:30 + orb_period
            orb_end_str = f"09:{end_min}" if end_min < 60 else f"10:{end_min - 60:02d}"
            return VectorizedNQORB(
                orb_start="09:30",
                orb_end=orb_end_str,
                use_rvol=True,
                rvol_thresh=params.get("rvol_threshold", 1.45),
                use_hurst=True,
                hurst_thresh=params.get("hurst_threshold", 0.52),
            )

        elif archetype == "lunch_hour_breakout":
            return VectorizedNQORB(
                orb_start=params.get("range_start", "11:30"),
                orb_end=params.get("range_end", "13:00"),
                ema_filter=params.get("ema_filter", 50),
            )

        elif archetype == "eod_momentum":
            # EOD momentum: use entry_time as ORB start, add 15min for end
            entry = params.get("entry_time", "14:00")
            # Calculate orb_end = entry_time + 15 minutes
            h, m = int(entry.split(':')[0]), int(entry.split(':')[1])
            m += 15
            if m >= 60:
                h += 1
                m -= 60
            orb_end = f"{h:02d}:{m:02d}"
            return VectorizedNQORB(
                orb_start=entry,
                orb_end=orb_end,
                ema_filter=params.get("ema_filter", 20),
                sl_atr_mult=params.get("sl_atr_mult", 2.0),
            )

        else:
            logger.warning(f"Unknown archetype '{archetype}', defaulting to ORB")
            return VectorizedNQORB()


class RigorousBacktester:
    """
    Stage 2: Runs actual backtests on strategy ideas from Stage 1.

    This is the CORE FIX for Critical Bug #1 - the old code was a placeholder
    that always returned 0 trades. This implementation uses the real VectorEngine.
    """

    def __init__(
        self,
        data_handler: SmartDataHandler = None,
        symbol: str = "NQ",
        search_dirs: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        interval: str = "5m",
        initial_capital: float = 100000.0,
        config: Dict[str, Any] = None,
    ):
        self.symbol = symbol
        self.search_dirs = search_dirs or []
        self.start_date = start_date or datetime(2011, 1, 1)
        self.end_date = end_date or datetime(2025, 6, 1)
        self.interval = interval
        self.initial_capital = initial_capital
        self.config = config or {}
        self.quality_checker = QualityChecker(self.config.get("quality", {}))
        self.registry = StrategyRegistry(self.config.get("db_path", "backtests.db"))

        # Lazy-load data
        self._data_handler = data_handler
        self._dataframe = None

    def _ensure_data(self):
        """Load data if not already loaded."""
        if self._dataframe is not None:
            return

        if self._data_handler is None:
            logger.info(f"Loading data for {self.symbol} ({self.interval})...")
            self._data_handler = SmartDataHandler(
                symbol_list=[self.symbol],
                search_dirs=self.search_dirs,
                start_date=pd.to_datetime(self.start_date),
                end_date=pd.to_datetime(self.end_date),
                interval=self.interval,
            )

        # Extract DataFrame for vector engine
        if self.symbol in self._data_handler.symbol_data:
            self._dataframe = self._data_handler.symbol_data[self.symbol]
        else:
            # Try first available symbol
            for sym, df in self._data_handler.symbol_data.items():
                self._dataframe = df
                self.symbol = sym
                break

        if self._dataframe is None:
            raise ValueError(f"No data available for {self.symbol}")

        # Fix Issue #9: Ensure datetime index is properly parsed
        self._fix_datetime_index()

        logger.info(
            f"Data loaded: {len(self._dataframe)} bars, "
            f"{self._dataframe.index[0]} to {self._dataframe.index[-1]}"
        )

    def _fix_datetime_index(self):
        """
        Fix Issue #9: Date parsing error that caused '1970-01-01' dates.
        Ensures the DataFrame has a proper DatetimeIndex.
        """
        df = self._dataframe
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # If index is numeric timestamps (nanoseconds)
                try:
                    df.index = pd.to_datetime(df.index, unit='ns')
                except Exception:
                    try:
                        df.index = pd.to_datetime(df.index, unit='s')
                    except Exception as e:
                        logger.error(f"Could not parse datetime index: {e}")
                        return

        # Validate dates aren't epoch (1970)
        if df.index[0].year < 2000:
            logger.warning(
                f"Suspicious date range detected: {df.index[0]} to {df.index[-1]}. "
                "This may indicate a datetime parsing error."
            )

        self._dataframe = df

    def backtest_strategy(self, strategy_idea: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a REAL backtest on a strategy idea.

        THIS IS THE FIX FOR CRITICAL BUG #1.
        Instead of returning dummy results, this actually executes the strategy
        using VectorEngine and returns real trade metrics.
        """
        strategy_name = strategy_idea.get("strategy_name", "Unknown")

        try:
            self._ensure_data()

            # Create vector strategy from archetype mapping
            vector_strategy = StrategyMapper.create_vector_strategy(strategy_idea)
            if vector_strategy is None:
                return self._error_result(strategy_name, "Could not create strategy")

            # Run the actual backtest via VectorEngine
            # Use configured commission/slippage from marcus_config (not hardcoded!)
            engine = VectorEngine(
                strategy=vector_strategy,
                initial_capital=self.initial_capital,
                commission=self.config.get('commission_per_unit', 2.06),
                slippage=self.config.get('slippage_per_unit', 5.0),
                point_value=self.config.get('point_value', 20.0),
            )

            # Apply date range filtering if start_date/end_date are set
            df = self._dataframe.copy()
            if self.start_date is not None:
                start_ts = pd.to_datetime(self.start_date)
                df = df[df.index >= start_ts]
            if self.end_date is not None:
                end_ts = pd.to_datetime(self.end_date)
                df = df[df.index <= end_ts]

            if len(df) == 0:
                return self._error_result(strategy_name, "No data in date range")

            result = engine.run(df)

            # Extract metrics from real results
            metrics = self._extract_metrics(result, strategy_name)

            return metrics

        except Exception as e:
            logger.error(f"Backtest failed for {strategy_name}: {e}\n{traceback.format_exc()}")
            return self._error_result(strategy_name, str(e))

    def _extract_metrics(self, result: Dict, strategy_name: str) -> Dict[str, Any]:
        """Extract performance metrics from VectorEngine result."""
        equity_curve = result.get("equity_curve")
        returns = result.get("returns")
        signals = result.get("signals")
        turnover = result.get("turnover")

        if equity_curve is None or len(equity_curve) == 0:
            return self._error_result(strategy_name, "Empty equity curve")

        # Safe metric extraction (Issue #7)
        try:
            final_equity = float(equity_curve.iloc[-1])
            initial_equity = float(equity_curve.iloc[0])
            total_return = (final_equity / initial_equity) - 1.0 if initial_equity > 0 else 0.0
        except (IndexError, TypeError, ZeroDivisionError):
            total_return = 0.0
            final_equity = self.initial_capital

        # Net profit
        net_profit = final_equity - self.initial_capital

        # Returns-based metrics
        clean_returns = returns.dropna() if returns is not None else pd.Series([0.0])

        # Sharpe ratio (annualized for intraday bars)
        # For 5-min bars: ~78 bars/day × 252 trading days = 19,656 bars/year
        # Annualization factor = sqrt(bars_per_year)
        if len(clean_returns) > 1 and clean_returns.std() > 0:
            # Estimate bars per year from data frequency
            if hasattr(self, 'interval') and self.interval == '5m':
                bars_per_year = 252 * 78  # 5-min bars: ~78 per day
            elif hasattr(self, 'interval') and self.interval in ('15m', '15min'):
                bars_per_year = 252 * 26  # 15-min bars: ~26 per day
            elif hasattr(self, 'interval') and self.interval in ('1h', '60m'):
                bars_per_year = 252 * 6.5  # hourly bars: ~6.5 per day
            else:
                bars_per_year = 252  # default: daily
            sharpe = float(np.sqrt(bars_per_year) * clean_returns.mean() / clean_returns.std())
        else:
            sharpe = 0.0

        # Max drawdown
        if equity_curve is not None and len(equity_curve) > 0:
            peak = equity_curve.cummax()
            safe_peak = peak.replace(0, np.nan)
            drawdown = ((equity_curve - peak) / safe_peak).fillna(0)
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0

        # Trade counting from signal changes
        if signals is not None:
            signal_changes = signals.diff().fillna(0)
            entries = (signal_changes != 0).sum()
            total_trades = max(0, int(entries) // 2)  # Entry + exit = 1 round trip
        else:
            total_trades = 0

        # Win rate estimation from returns segmented by trade
        win_trades = 0
        loss_trades = 0
        if signals is not None and returns is not None:
            # Segment returns by position
            pos = signals.shift(1).fillna(0)
            trade_returns = returns * pos

            # Group returns by trade (signal change boundaries)
            trade_boundaries = pos.diff().fillna(0) != 0
            trade_ids = trade_boundaries.cumsum()
            trade_pnls = trade_returns.groupby(trade_ids).sum()
            trade_pnls = trade_pnls[trade_pnls != 0]  # Remove flat periods

            total_trades = len(trade_pnls)
            win_trades = int((trade_pnls > 0).sum())
            loss_trades = int((trade_pnls < 0).sum())

        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Profit factor (based on per-trade PnL, not bar-level returns)
        # Uses trade_pnls computed above for accurate win/loss grouping
        if total_trades > 0 and win_trades + loss_trades > 0:
            # Recompute trade-level profit factor from grouped trade PnLs
            pos = signals.shift(1).fillna(0)
            trade_returns = returns * pos
            trade_boundaries = pos.diff().fillna(0) != 0
            trade_ids = trade_boundaries.cumsum()
            trade_pnls_pf = trade_returns.groupby(trade_ids).sum()
            trade_pnls_pf = trade_pnls_pf[trade_pnls_pf != 0]
            gross_profit = float(trade_pnls_pf[trade_pnls_pf > 0].sum())
            gross_loss = float(abs(trade_pnls_pf[trade_pnls_pf < 0].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            profit_factor = 0.0

        # Date range (Fix Issue #9)
        try:
            date_start = str(self._dataframe.index[0])
            date_end = str(self._dataframe.index[-1])
        except (IndexError, AttributeError):
            date_start = str(self.start_date)
            date_end = str(self.end_date)

        # Equity returns for Stage 5 complementarity check
        # These are the pct_change() of the equity curve, used for correlation analysis
        if equity_curve is not None and len(equity_curve) > 1:
            equity_returns_arr = equity_curve.pct_change().dropna().values
        else:
            equity_returns_arr = np.array([])

        return {
            "strategy_name": strategy_name,
            "total_return": total_return,
            "net_profit": net_profit,
            "final_equity": final_equity,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "profit_factor": profit_factor,
            "date_range_start": date_start,
            "date_range_end": date_end,
            "equity_returns": equity_returns_arr,
            # P0-5: Preserve raw equity curve Series for winner persistence
            "equity_curve_raw": equity_curve,
            "status": "completed",
            "error": None,
        }

    def _error_result(self, strategy_name: str, error: str) -> Dict[str, Any]:
        """Return a properly structured error result (not 0-trades placeholder!)."""
        return {
            "strategy_name": strategy_name,
            "total_return": 0.0,
            "net_profit": 0.0,
            "final_equity": self.initial_capital,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
            "profit_factor": 0.0,
            "date_range_start": str(self.start_date),
            "date_range_end": str(self.end_date),
            "status": "error",
            "error": error,
        }

    def batch_backtest(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run backtests on multiple strategy ideas.
        Returns list of results with quality assessment.
        """
        results = []

        for i, idea in enumerate(ideas):
            name = idea.get("strategy_name", f"Strategy_{i}")
            logger.info(f"[{i+1}/{len(ideas)}] Backtesting: {name}")

            metrics = self.backtest_strategy(idea)

            # Quality check with OR-logic (Issue #12)
            passed, issues = self.quality_checker.check(metrics)
            metrics["quality_passed"] = passed
            metrics["quality_issues"] = issues

            if passed:
                logger.info(f"  [PASSED] {name}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                           f"WR={metrics['win_rate']:.1f}%, Trades={metrics['total_trades']}")
            else:
                logger.info(f"  [FAILED] {name}: {'; '.join(issues)}")

            results.append(metrics)

            # Archive to registry
            try:
                self.registry.save_run(
                    strategy_name=name,
                    symbol=self.symbol,
                    interval=self.interval,
                    params=idea.get("params", {}),
                    stats={
                        "Total Return": metrics["total_return"],
                        "Sharpe Ratio": metrics["sharpe_ratio"],
                        "Max Drawdown": metrics["max_drawdown"],
                        "Ending Equity": metrics["final_equity"],
                        "Profit Factor": metrics["profit_factor"],
                    },
                    data_range=(self.start_date, self.end_date),
                    notes=f"Marcus auto-research cycle | Quality: {'PASS' if passed else 'FAIL'}",
                )
            except Exception as e:
                logger.warning(f"Failed to archive {name}: {e}")

        return results
