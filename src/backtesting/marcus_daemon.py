"""
Marcus Daemon - 24/7 Autonomous Research Service
==================================================
The main entry point for the Marcus autonomous agent. Runs as a Windows Service
via NSSM, executing research cycles on a configurable schedule.

Features:
    - Scheduled research cycles (default: every 4 hours)
    - Dashboard auto-rebuild (default: every 15 minutes)
    - Heartbeat monitoring (default: every 5 minutes)
    - Crash recovery with state persistence
    - GPU resource validation on startup
    - Data integrity checks
    - Graceful shutdown handling

Usage:
    # Continuous mode (for Windows Service / NSSM)
    python marcus_daemon.py

    # Single cycle mode (for testing)
    python marcus_daemon.py --once

    # With custom config file
    python marcus_daemon.py --config path/to/config.json

    # Dashboard-only rebuild (no research)
    python marcus_daemon.py --dashboard-only
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import traceback
from datetime import datetime
from typing import Optional

# Fix imports when running as __main__
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
_project_dir = os.path.dirname(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from backtesting.marcus_config import MarcusConfig
from backtesting.research_engine import AutonomousResearchEngine
from backtesting.marcus_dashboard import MarcusDashboard
from backtesting.monitor import PipelineMonitor


def _setup_logging(config: MarcusConfig):
    """Configure logging for the daemon."""
    log_dir = config.logs_dir
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "marcus_daemon.log")

    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level, logging.INFO))

    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            log_file,
            maxBytes=config.log_rotate_bytes,
            backupCount=config.max_log_files,
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root.addHandler(fh)
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(ch)

    return logging.getLogger("MarcusDaemon")


class MarcusDaemon:
    """
    24/7 autonomous research daemon.
    Manages the research cycle scheduler, dashboard rebuilds, and health monitoring.
    """

    def __init__(self, config: MarcusConfig):
        self.config = config
        self.engine = AutonomousResearchEngine(config)
        self.dashboard = MarcusDashboard(config)
        self.monitor = PipelineMonitor(config.logs_dir)
        self.logger = logging.getLogger("MarcusDaemon")

        self._running = False
        self._state = {
            'last_cycle_at': None,
            'last_dashboard_at': None,
            'last_heartbeat_at': None,
            'total_cycles': 0,
            'total_errors': 0,
            'started_at': None,
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Main daemon loop. Runs until stopped."""
        self._running = True
        self._state['started_at'] = datetime.now().isoformat()
        self._load_state()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("=" * 60)
        self.logger.info("MARCUS DAEMON STARTING")
        self.logger.info("=" * 60)
        self.monitor.log_event("Daemon", "STARTUP", "Marcus daemon starting", "INFO")

        # Startup checks
        self._check_gpu()
        self._validate_data()

        # Initial dashboard build
        try:
            self.dashboard.rebuild()
            self._state['last_dashboard_at'] = datetime.now().isoformat()
            self.logger.info("Initial dashboard build complete.")
        except Exception as e:
            self.logger.error(f"Initial dashboard build failed: {e}")

        # Main loop
        self.logger.info(f"Entering main loop (cycle every {self.config.cycle_interval_minutes}min, "
                         f"dashboard every {self.config.dashboard_refresh_minutes}min)")
        self._cycle_running = False  # Prevent concurrent cycles

        while self._running:
            try:
                now = datetime.now()

                # 1. Research cycle (with concurrency guard)
                if not self._cycle_running and self._should_run_cycle(now):
                    self._cycle_running = True
                    try:
                        self._run_research_cycle()
                    finally:
                        self._cycle_running = False

                # 2. Dashboard refresh
                if self._should_refresh_dashboard(now):
                    self._refresh_dashboard()

                # 3. Heartbeat
                if self._should_heartbeat(now):
                    self._heartbeat()

                # P1-2: Check for stale heartbeat (>30 min since last)
                last_hb = self._state.get('last_heartbeat_at')
                if last_hb:
                    try:
                        hb_age = (now - datetime.fromisoformat(last_hb)).total_seconds()
                        if hb_age > 1800:  # 30 minutes
                            self.logger.warning(f"STALE HEARTBEAT: {hb_age:.0f}s since last beat")
                            self.monitor.log_event("Daemon", "STALE_HEARTBEAT",
                                                   f"Heartbeat {hb_age:.0f}s old", "WARNING")
                    except (ValueError, TypeError):
                        pass

                # 4. Sleep (check every 30 seconds)
                time.sleep(30)

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt received.")
                self.stop()
                break
            except Exception as e:
                self._cycle_running = False  # Reset guard on error
                self._state['total_errors'] += 1
                self.logger.error(f"Main loop error: {e}\n{traceback.format_exc()}")
                self.monitor.log_event("Daemon", "LOOP_ERROR", str(e), "ERROR")
                self._save_state()
                # Back off on errors (60s, not 300s to reduce downtime)
                time.sleep(60)

        self.logger.info("Main loop exited.")

    def stop(self):
        """Graceful shutdown."""
        self.logger.info("MARCUS DAEMON STOPPING")
        self._running = False
        self._save_state()
        self.monitor.log_event("Daemon", "SHUTDOWN", "Marcus daemon stopping gracefully", "INFO")
        # Final dashboard update
        try:
            self.dashboard.rebuild()
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        """Handle OS signals for graceful shutdown."""
        self.logger.info(f"Signal {signum} received. Initiating shutdown...")
        self.stop()

    # =========================================================================
    # Scheduled Tasks
    # =========================================================================

    def _should_run_cycle(self, now: datetime) -> bool:
        """Check if enough time has passed since last research cycle."""
        last = self._state.get('last_cycle_at')
        if last is None:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            elapsed_min = (now - last_dt).total_seconds() / 60
            return elapsed_min >= self.config.cycle_interval_minutes
        except (ValueError, TypeError):
            return True

    def _should_refresh_dashboard(self, now: datetime) -> bool:
        """Check if dashboard needs rebuilding."""
        last = self._state.get('last_dashboard_at')
        if last is None:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            elapsed_min = (now - last_dt).total_seconds() / 60
            return elapsed_min >= self.config.dashboard_refresh_minutes
        except (ValueError, TypeError):
            return True

    def _should_heartbeat(self, now: datetime) -> bool:
        """Check if heartbeat is due."""
        last = self._state.get('last_heartbeat_at')
        if last is None:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            elapsed_min = (now - last_dt).total_seconds() / 60
            return elapsed_min >= self.config.health_check_minutes
        except (ValueError, TypeError):
            return True

    def _run_research_cycle(self):
        """Execute one research cycle."""
        self.logger.info("=" * 40)
        self.logger.info("STARTING RESEARCH CYCLE")
        self.logger.info("=" * 40)

        cycle_start = time.time()
        try:
            result = self.engine.run_cycle()
            self._state['last_cycle_at'] = datetime.now().isoformat()
            self._state['total_cycles'] += 1

            # P0-2: Propagate in-cycle errors to daemon state
            # CycleResult.errors tracks errors caught within idea processing,
            # but these never reached _state['total_errors'] before this fix.
            if result.errors > 0:
                self._state['total_errors'] += result.errors
                self.logger.warning(f"Cycle had {result.errors} internal errors: "
                                    f"{'; '.join(result.error_details[:3])}")

            self.logger.info(f"Cycle complete: {result.summary()}")

            # Rebuild dashboard after each cycle
            self._refresh_dashboard()

        except Exception as e:
            self._state['total_errors'] += 1
            self.logger.error(f"Research cycle failed: {e}\n{traceback.format_exc()}")
            self.monitor.log_event("Daemon", "CYCLE_ERROR", str(e), "ERROR")

        # P1-2: Stall detection — warn if cycle took too long
        cycle_elapsed = time.time() - cycle_start
        if cycle_elapsed > 600:  # 10 minutes
            self.logger.warning(f"SLOW CYCLE: took {cycle_elapsed:.0f}s (>600s threshold)")
            self.monitor.log_event("Daemon", "SLOW_CYCLE",
                                   f"Cycle took {cycle_elapsed:.0f}s", "WARNING")

        self._save_state()

    def _refresh_dashboard(self):
        """Rebuild the HTML dashboard."""
        try:
            self.dashboard.rebuild()
            self._state['last_dashboard_at'] = datetime.now().isoformat()
        except Exception as e:
            self.logger.error(f"Dashboard rebuild failed: {e}")

    def _heartbeat(self):
        """Write heartbeat for liveness monitoring."""
        self.monitor.log_heartbeat()
        self._state['last_heartbeat_at'] = datetime.now().isoformat()

    # =========================================================================
    # Startup Checks
    # =========================================================================

    def _check_gpu(self):
        """Verify GPU availability on startup."""
        try:
            from backtesting.accelerate import get_gpu_info, GPU_AVAILABLE
            if GPU_AVAILABLE:
                info = get_gpu_info()
                self.logger.info(f"GPU ACTIVE: {info.get('device_name', 'CUDA device')}")
                self.logger.info(f"  Memory: {info.get('total_memory_mb', '?')} MB")
                self.monitor.log_gpu_status(True, f"GPU active: {info.get('device_name', 'CUDA')}")
            else:
                self.logger.warning("GPU not available. Using CPU fallback.")
                self.monitor.log_gpu_status(False, "GPU not available, CPU mode")
        except ImportError:
            self.logger.warning("accelerate module not available. CPU only.")
            self.monitor.log_gpu_status(False, "accelerate module import failed")

    def _validate_data(self):
        """Check that required data files exist."""
        data_dir = self.config.data_dir
        if not os.path.exists(data_dir):
            self.logger.warning(f"Data directory does not exist: {data_dir}")
            self.monitor.log_event("Daemon", "DATA_WARNING",
                                   f"Data dir missing: {data_dir}", "WARNING")
            return

        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            self.logger.warning(f"No CSV files found in {data_dir}")
            self.monitor.log_event("Daemon", "DATA_WARNING",
                                   "No CSV files found", "WARNING")
        else:
            self.logger.info(f"Data directory OK: {len(csv_files)} CSV files in {data_dir}")
            self.monitor.log_data_loading(
                symbol=self.config.symbol,
                source=data_dir,
                success=True
            )

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _save_state(self):
        """Persist daemon state to JSON for crash recovery."""
        try:
            os.makedirs(os.path.dirname(self.config.state_file), exist_ok=True)
            with open(self.config.state_file, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"State save failed: {e}")

    def _load_state(self):
        """Restore daemon state from previous run."""
        try:
            if os.path.exists(self.config.state_file):
                with open(self.config.state_file, 'r') as f:
                    saved = json.load(f)
                # Merge saved state (preserving cycle count, last timestamps)
                self._state['last_cycle_at'] = saved.get('last_cycle_at')
                self._state['last_dashboard_at'] = saved.get('last_dashboard_at')
                self._state['total_cycles'] = saved.get('total_cycles', 0)
                self._state['total_errors'] = saved.get('total_errors', 0)
                self.logger.info(f"Restored state: {self._state['total_cycles']} previous cycles")
        except Exception as e:
            self.logger.warning(f"State restore failed (starting fresh): {e}")

    # =========================================================================
    # Single-Cycle Mode
    # =========================================================================

    def run_once(self):
        """Execute a single research cycle and exit. For testing."""
        self.logger.info("=" * 60)
        self.logger.info("MARCUS SINGLE CYCLE MODE")
        self.logger.info("=" * 60)

        self._check_gpu()
        self._validate_data()

        result = self.engine.run_cycle()
        self.logger.info(f"Cycle result: {result.summary()}")

        # Rebuild dashboard
        self.dashboard.rebuild()
        self.logger.info(f"Dashboard: {self.config.dashboard_path}")

        return result

    def dashboard_only(self):
        """Just rebuild the dashboard and exit."""
        self.logger.info("Dashboard-only mode")
        self.dashboard.rebuild()
        self.logger.info(f"Dashboard written to: {self.config.dashboard_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Marcus Autonomous Research Agent - 24/7 Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python marcus_daemon.py                  # Run continuously (service mode)
  python marcus_daemon.py --once           # Single research cycle
  python marcus_daemon.py --dashboard-only # Just rebuild dashboard
  python marcus_daemon.py --config my.json # Use custom config
        """
    )
    parser.add_argument('--once', action='store_true',
                        help='Run a single research cycle and exit')
    parser.add_argument('--dashboard-only', action='store_true',
                        help='Only rebuild the dashboard, no research')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file')

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config = MarcusConfig.from_file(args.config)
    else:
        config = MarcusConfig.default()

    # Setup logging
    logger = _setup_logging(config)

    # Create daemon
    daemon = MarcusDaemon(config)

    if args.dashboard_only:
        daemon.dashboard_only()
    elif args.once:
        daemon.run_once()
    else:
        daemon.start()


if __name__ == "__main__":
    main()
