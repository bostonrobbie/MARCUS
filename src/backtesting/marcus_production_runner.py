"""
Marcus Production Runner
========================
Unified autonomous strategy research loop that connects:
  Stage 1 (LLM idea generation) -> Stage 2 (Real backtesting) -> Quality Gate -> Registry

Fixes:
- Major Issue #2: Unifies OLD and NEW systems into single pipeline
- Major Issue #3: Bridges LLM ideas to real backtest code
- Issue #8: Windows Unicode encoding
- Issue #11: Progress persistence across restarts
- Issue #13: Error recovery with try-except around cycle execution
"""

import sys
import os
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# Fix Issue #8: Windows Unicode encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

from .stage1_strategy_research import StrategyIdeaGenerator
from .stage2_rigorous_backtest import RigorousBacktester, QualityChecker
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)


class CycleResult:
    """Result of one research cycle."""
    def __init__(self, cycle_num: int):
        self.cycle_num = cycle_num
        self.ideas_generated = 0
        self.backtests_run = 0
        self.strategies_passed = 0
        self.strategies_failed = 0
        self.best_sharpe = 0.0
        self.best_strategy = ""
        self.errors: List[str] = []
        self.duration_seconds = 0.0
        self.timestamp = datetime.now()

    def summary(self) -> str:
        """Human-readable cycle summary."""
        status = "OK" if not self.errors else f"ERRORS({len(self.errors)})"
        return (
            f"Cycle {self.cycle_num} [{status}] - "
            f"Ideas: {self.ideas_generated}, "
            f"Backtests: {self.backtests_run}, "
            f"Passed: {self.strategies_passed}/{self.backtests_run}, "
            f"Best Sharpe: {self.best_sharpe:.2f} ({self.best_strategy}), "
            f"Duration: {self.duration_seconds:.1f}s"
        )


class MarcusProductionRunner:
    """
    The unified Marcus autonomous strategy research runner.

    Runs continuous cycles of:
    1. Generate strategy ideas (LLM or fallback grid)
    2. Backtest each idea with real VectorEngine
    3. Quality gate check (with OR-logic for high R:R strategies)
    4. Archive results to registry
    5. Report best strategies

    Features error recovery (Issue #13), progress persistence (Issue #11),
    and Windows Unicode support (Issue #8).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self._setup_logging()

        # Core components
        self.idea_generator = StrategyIdeaGenerator(
            config=self.config,
            db_path=self.config.get("db_path", "backtests.db"),
        )
        self.backtester = RigorousBacktester(
            symbol=self.config.get("symbol", "NQ"),
            search_dirs=self.config.get("data", {}).get("search_dirs", []),
            start_date=datetime.fromisoformat(self.config["backtest"]["start_date"])
                if "backtest" in self.config and "start_date" in self.config.get("backtest", {})
                else datetime(2011, 1, 1),
            end_date=datetime.fromisoformat(self.config["backtest"]["end_date"])
                if "backtest" in self.config and "end_date" in self.config.get("backtest", {})
                else datetime(2025, 6, 1),
            interval=self.config.get("interval", "5m"),
            initial_capital=self.config.get("initial_capital", 100000.0),
            config=self.config,
        )
        self.registry = StrategyRegistry(self.config.get("db_path", "backtests.db"))

        # State
        self.cycle_count = 0
        self.total_passed = 0
        self.all_results: List[CycleResult] = []
        self._running = False

        # Load persisted state (Issue #11)
        self._load_state()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "symbol": "NQ",
            "interval": "5m",
            "initial_capital": 100000.0,
            "ideas_per_cycle": 5,
            "max_cycles": 100,
            "cycle_delay_seconds": 5,
            "db_path": "backtests.db",
            "state_file": "marcus_state.json",
            "log_file": "marcus.log",
            "data": {"search_dirs": []},
            "backtest": {
                "start_date": "2011-01-01",
                "end_date": "2025-06-01",
            },
        }

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        log_file = self.config.get("log_file", "marcus.log")

        # Create formatter
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_handler.setLevel(logging.INFO)

        # Root logger
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.INFO)

    def _load_state(self):
        """Load persisted state from disk (Issue #11: progress persistence)."""
        state_file = self.config.get("state_file", "marcus_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                self.cycle_count = state.get("cycle_count", 0)
                self.total_passed = state.get("total_passed", 0)
                logger.info(
                    f"Resumed from cycle {self.cycle_count} "
                    f"({self.total_passed} strategies passed previously)"
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load state: {e}, starting fresh")

    def _save_state(self):
        """Persist state to disk (Issue #11)."""
        state_file = self.config.get("state_file", "marcus_state.json")
        state = {
            "cycle_count": self.cycle_count,
            "total_passed": self.total_passed,
            "last_run": datetime.now().isoformat(),
            "tested_strategies": self.idea_generator.get_tested_count(),
        }
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except IOError as e:
            logger.warning(f"Could not save state: {e}")

    def run_cycle(self) -> CycleResult:
        """
        Run a single research cycle.
        Wrapped in try-except for error recovery (Issue #13).
        """
        self.cycle_count += 1
        result = CycleResult(self.cycle_count)
        start_time = time.time()

        logger.info(f"{'='*60}")
        logger.info(f"MARCUS RESEARCH CYCLE {self.cycle_count}")
        logger.info(f"{'='*60}")

        try:
            # Stage 1: Generate ideas
            logger.info("[Stage 1] Generating strategy ideas...")
            ideas_per_cycle = self.config.get("ideas_per_cycle", 5)
            ideas = self.idea_generator.generate_ideas(n_ideas=ideas_per_cycle)
            result.ideas_generated = len(ideas)
            logger.info(f"  Generated {len(ideas)} ideas")

            if not ideas:
                logger.warning("No ideas generated, skipping cycle")
                result.errors.append("No ideas generated")
                return result

            # Stage 2: Backtest each idea
            logger.info("[Stage 2] Running backtests...")
            backtest_results = self.backtester.batch_backtest(ideas)
            result.backtests_run = len(backtest_results)

            # Stage 3: Evaluate results
            logger.info("[Stage 3] Evaluating results...")
            for bt_result in backtest_results:
                if bt_result.get("quality_passed", False):
                    result.strategies_passed += 1
                    self.total_passed += 1

                    sharpe = bt_result.get("sharpe_ratio", 0)
                    if sharpe > result.best_sharpe:
                        result.best_sharpe = sharpe
                        result.best_strategy = bt_result.get("strategy_name", "")
                else:
                    result.strategies_failed += 1

            # Log cycle summary
            logger.info(f"[Summary] {result.summary()}")

            if result.strategies_passed > 0:
                logger.info(
                    f"  ** {result.strategies_passed} strategies PASSED quality gate **"
                )

        except Exception as e:
            # Issue #13: Error recovery - don't crash the entire system
            error_msg = f"Cycle {self.cycle_count} error: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time

        # Persist state after each cycle (Issue #11)
        self._save_state()
        self.all_results.append(result)

        return result

    def run(self, max_cycles: int = None):
        """
        Run the autonomous research loop.
        Continues until max_cycles reached or manually stopped.
        """
        max_cycles = max_cycles or self.config.get("max_cycles", 100)
        delay = self.config.get("cycle_delay_seconds", 5)
        self._running = True

        logger.info(f"Marcus Production Runner starting (max {max_cycles} cycles)")
        logger.info(f"Previously tested: {self.idea_generator.get_tested_count()} strategies")

        cycles_run = 0

        try:
            while self._running and cycles_run < max_cycles:
                cycle_result = self.run_cycle()
                cycles_run += 1

                # Brief delay between cycles
                if self._running and cycles_run < max_cycles:
                    time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Marcus stopped by user (KeyboardInterrupt)")
        finally:
            self._running = False
            self._save_state()
            self._print_final_report()

    def stop(self):
        """Gracefully stop the runner."""
        self._running = False

    def _print_final_report(self):
        """Print a summary of all cycles."""
        logger.info(f"\n{'='*60}")
        logger.info("MARCUS SESSION REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Total Cycles: {len(self.all_results)}")
        logger.info(f"Total Strategies Tested: {self.idea_generator.get_tested_count()}")
        logger.info(f"Total Passed Quality Gate: {self.total_passed}")

        # Best overall
        best_sharpe = 0
        best_name = ""
        for r in self.all_results:
            if r.best_sharpe > best_sharpe:
                best_sharpe = r.best_sharpe
                best_name = r.best_strategy

        if best_name:
            logger.info(f"Best Strategy: {best_name} (Sharpe: {best_sharpe:.2f})")

        # Errors summary
        total_errors = sum(len(r.errors) for r in self.all_results)
        if total_errors > 0:
            logger.warning(f"Total Errors: {total_errors}")

        logger.info(f"{'='*60}")

    def get_leaderboard(self, limit: int = 10):
        """Get top strategies from the registry."""
        return self.registry.get_leaderboard(limit=limit)


def main():
    """CLI entry point for Marcus."""
    import argparse

    parser = argparse.ArgumentParser(description="Marcus Autonomous Strategy Research")
    parser.add_argument("--config", default="marcus_config.json", help="Config file path")
    parser.add_argument("--cycles", type=int, default=10, help="Number of research cycles")
    parser.add_argument("--symbol", default="NQ", help="Symbol to research")
    parser.add_argument("--interval", default="5m", help="Data interval")
    parser.add_argument("--single", action="store_true", help="Run single cycle only")

    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)

    # CLI overrides
    config["symbol"] = args.symbol
    config["interval"] = args.interval

    runner = MarcusProductionRunner(config=config)

    if args.single:
        result = runner.run_cycle()
        print(f"\n{result.summary()}")
    else:
        runner.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
