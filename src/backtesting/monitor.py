
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any

class PipelineMonitor:
    """
    Centralized Health Monitor for the Strategy Pipeline.
    Tracks critical system events, resource usage, and error rates.
    """
    _instance = None

    def __new__(cls, log_dir="logs"):
        if cls._instance is None:
            cls._instance = super(PipelineMonitor, cls).__new__(cls)
            cls._instance._initialize(log_dir)
        elif cls._instance.log_dir != log_dir and log_dir != "logs":
            # P1-3: Allow re-initialization when log_dir changes from default
            # The singleton may have been created with default "logs" path before
            # the daemon passes the real log path from MarcusConfig.
            cls._instance._initialize(log_dir)
        return cls._instance

    def _initialize(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "system_health.jsonl")
        
        # Setup standard Python logging as well for console output
        self.logger = logging.getLogger("PipelineMonitor")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_event(self, component: str, event_type: str, message: str, severity: str = "INFO", metadata: Dict[str, Any] = None):
        """
        Logs a health event.
        - severity: INFO, WARNING, ERROR, CRITICAL
        """
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "component": component,
            "type": event_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        }
        
        # 1. Write to JSONL
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        # 2. Console (Filtered)
        if severity in ["WARNING", "ERROR", "CRITICAL"]:
            self.logger.warning(f"[{component}] {message}")
        else:
            self.logger.info(f"[{component}] {message}")

    def log_gpu_status(self, success: bool, details: str):
        severity = "INFO" if success else "WARNING"
        status = "GPU_ACTIVE" if success else "GPU_FALLBACK"
        self.log_event("VectorEngine", status, details, severity)

    def log_data_loading(self, symbol: str, source: str, success: bool):
        severity = "INFO" if success else "ERROR"
        status = "DATA_LOAD_SUCCESS" if success else "DATA_LOAD_FAIL"
        msg = f"Loaded {symbol} from {source}" if success else f"Failed to load {symbol} from {source}"
        self.log_event("DataHandler", status, msg, severity, {"symbol": symbol, "source": source})

    # =========================================================================
    # Marcus Daemon Extensions
    # =========================================================================

    def log_cycle_start(self, cycle_num: int):
        """Log the start of a research cycle."""
        self.log_event("Marcus", "CYCLE_START", f"Cycle {cycle_num} starting", "INFO",
                       {"cycle_num": cycle_num})

    def log_cycle_end(self, cycle_num: int, result: Dict[str, Any]):
        """Log the end of a research cycle with results."""
        msg = (f"Cycle {cycle_num} complete: "
               f"{result.get('ideas_generated', 0)} ideas, "
               f"{result.get('stage1_passed', 0)} S1, "
               f"{result.get('stage5_passed', 0)} S5 | "
               f"Best Sharpe: {result.get('best_sharpe', 0):.2f}")
        self.log_event("Marcus", "CYCLE_END", msg, "INFO", result)

    def log_disposal(self, strategy_name: str, stage: str, reason: str):
        """Log a strategy disposal/rejection."""
        msg = f"DISPOSED: {strategy_name} at {stage} - {reason}"
        self.log_event("Lifecycle", "DISPOSAL", msg, "INFO",
                       {"strategy": strategy_name, "stage": stage, "reason": reason})

    def log_promotion(self, strategy_name: str, to_stage: str, sharpe: float = 0.0):
        """Log a strategy promotion through pipeline stages."""
        msg = f"PROMOTED: {strategy_name} -> {to_stage} (Sharpe: {sharpe:.2f})"
        self.log_event("Lifecycle", "PROMOTION", msg, "INFO",
                       {"strategy": strategy_name, "stage": to_stage, "sharpe": sharpe})

    def log_heartbeat(self):
        """Write daemon heartbeat for liveness monitoring."""
        self.log_event("Daemon", "HEARTBEAT", "alive", "INFO")
        # Also write a simple timestamp file for fast external checks
        try:
            hb_path = os.path.join(self.log_dir, "heartbeat.txt")
            with open(hb_path, "w") as f:
                f.write(datetime.now().isoformat())
        except Exception:
            pass

    def get_recent_events(self, limit: int = 50) -> list:
        """Read recent events from the JSONL log. Returns list of dicts."""
        events = []
        try:
            if not os.path.exists(self.log_file):
                return events
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        events.reverse()  # Most recent first
        return events

    def get_last_heartbeat(self) -> str:
        """Read last heartbeat timestamp. Returns ISO string or 'UNKNOWN'."""
        try:
            hb_path = os.path.join(self.log_dir, "heartbeat.txt")
            if os.path.exists(hb_path):
                with open(hb_path, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        return "UNKNOWN"

    def get_error_count(self, hours: int = 24) -> int:
        """Count errors in the last N hours."""
        count = 0
        cutoff = datetime.now().timestamp() - (hours * 3600)
        try:
            events = self.get_recent_events(limit=500)
            for evt in events:
                if evt.get('severity') in ('ERROR', 'CRITICAL'):
                    try:
                        ts = datetime.fromisoformat(evt.get('timestamp', '')).timestamp()
                        if ts >= cutoff:
                            count += 1
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        return count


if __name__ == "__main__":
    monitor = PipelineMonitor()
    monitor.log_event("Test", "INIT", "System initializing...")
    monitor.log_gpu_status(False, "Numba import failed. Switched to CPU.")
