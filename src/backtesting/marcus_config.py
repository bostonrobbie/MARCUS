"""
Marcus Autonomous Agent - Central Configuration

All tunable values for the Marcus daemon, research engine, lifecycle manager,
and dashboard in one place. No magic numbers scattered across files.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


# === Base Paths ===
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_QUANT_LAB = os.path.abspath(os.path.join(_BASE_DIR, "..", "..", ".."))
_MARCUS_DIR = os.path.join(_QUANT_LAB, "Marcus_Research")


@dataclass
class MarcusConfig:
    """Central configuration for the entire Marcus autonomous system."""

    # === Paths ===
    data_dir: str = os.path.join(_QUANT_LAB, "data")
    db_path: str = os.path.join(_BASE_DIR, "marcus_registry.db")
    reports_dir: str = os.path.join(_MARCUS_DIR, "reports")
    dashboard_path: str = os.path.join(_MARCUS_DIR, "dashboard", "marcus_live.html")
    logs_dir: str = os.path.join(_MARCUS_DIR, "logs")
    state_file: str = os.path.join(_MARCUS_DIR, "marcus_daemon_state.json")
    pine_dir: str = os.path.join(_MARCUS_DIR, "strategies")

    # === Schedule ===
    cycle_interval_minutes: int = 1          # Continuous research cycles (start next ~30s after previous ends)
    dashboard_refresh_minutes: int = 15     # Rebuild dashboard HTML
    health_check_minutes: int = 5           # Heartbeat interval

    # === Research ===
    symbol: str = "NQ"
    interval: str = "5m"
    point_value: float = 20.0              # NQ E-mini: 1 index point = $20
    ideas_per_cycle: int = 10               # Strategy ideas per cycle
    max_active_strategies: int = 20         # Cap on STAGE5_PASS + DEPLOYED
    variants_per_baseline: int = 4          # Sensitivity variants per passing baseline
    initial_capital: float = 100000.0

    # === Quality Gates ===
    # Stage 1: Basic profitability (standard costs)
    s1_min_profit: float = 0.0
    s1_min_trades: int = 200
    s1_commission_per_unit: float = 2.06    # $2.06 per order = $4.12 RT
    s1_slippage_per_unit: float = 5.0       # 1 tick on NQ

    # Stage 2: Gauntlet stress test (elevated costs)
    # Calibrated 2026-02-17 against actual data distribution:
    #   Best S1 Sharpe = 0.31, top 18% of S1 passers have Sharpe >= 0.15
    #   With 1.5x costs, Sharpe drops ~20-30%, so 0.15 threshold is achievable
    s2_min_sharpe: float = 0.15
    s2_max_drawdown_pct: float = 0.30       # 30%
    s2_min_trades: int = 150
    s2_min_profit_factor: float = 1.0      # Must be profitable (PF > 1.0)
    s2_min_win_rate: float = 30.0            # 30% win rate (stored as percentage), OR condition with PF
    s2_alt_min_profit_factor: float = 1.1   # Alt path: PF >= 1.1
    s2_commission_mult: float = 1.5         # 1.5x commission
    s2_slippage_mult: float = 1.5           # 1.5x slippage

    # Stage 3: Regime split periods
    s3_periods: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("2011-01-01", "2015-12-31"),   # Post-GFC recovery
        ("2016-01-01", "2020-12-31"),   # Mixed + COVID
        ("2021-01-01", "2026-12-31"),   # Rate hikes + AI
    ])
    s3_min_profit_per_period: float = 0.0

    # Stage 4: Parameter sensitivity (tightened 2026-02-17)
    # ±20% variation, max 50% profit drop, 60% must be profitable
    # WFO robustness score integrated for cliff/stability detection
    s4_max_profit_drop_pct: float = 0.50    # Max 50% profit drop (was 200% — way too lax)
    s4_variation_factors: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.1, 1.2])  # ±20%
    s4_all_variants_profitable: bool = False  # Allow some variants to fail
    s4_min_profitable_pct: float = 0.60     # At least 60% of variants must be profitable
    s4_min_robustness_score: float = 50.0   # Minimum robustness score (0-100, from wfo_analytics)

    # Stage 5: Complementarity
    s5_max_daily_correlation: float = 0.3
    s5_require_combined_sharpe_improvement: bool = True

    # === Disposal ===
    max_degradation_strikes: int = 3        # Strikes before forced archive
    delete_stage1_failures: bool = True     # Immediate delete for garbage
    graveyard_retention_days: int = -1      # -1 = keep graveyard hashes forever
    rejected_cleanup_days: int = 7          # Delete REJECTED data after N days

    # === LLM (for idea generation, optional) ===
    llm_provider: str = os.environ.get("MARCUS_LLM_PROVIDER", "ollama")
    llm_model: str = os.environ.get("MARCUS_LLM_MODEL", "mistral")
    llm_base_url: str = os.environ.get("MARCUS_LLM_URL", "http://localhost:11434")
    llm_temperature: float = 0.7
    llm_enabled: bool = False               # Disabled by default; fallback grid is reliable

    # === Statistical Verification Gates ===
    # Sharpe CI gate (S2): reject if 95% CI lower bound crosses zero
    stat_require_sharpe_ci_positive: bool = True
    stat_confidence_level: float = 0.95     # 95% confidence interval

    # Deflated Sharpe Ratio gate (S5): reject if DSR probability too low
    # Bailey & Lopez de Prado (2014) - accounts for multiple testing
    stat_min_dsr_probability: float = 0.95  # Must be 95%+ likely skill not luck

    # Permutation test gate (S2): reject if strategy indistinguishable from random
    s2_permutation_enabled: bool = True
    s2_permutation_sims: int = 50           # 50 sims for reasonable cycle time

    # Monte Carlo gate (winner save): reject if VaR95 too severe
    mc_min_var95: float = -0.30             # Max 30% loss at 95th percentile

    # Complexity caps: reject over-parameterized strategies
    max_strategy_params: int = 8            # Max tunable parameters
    max_strategy_indicators: int = 5        # Max technical indicators

    # === NQmain Portfolio Fit ===
    nqmain_pine_path: str = os.path.join(_BASE_DIR, "reference", "NqOrbEnhanced_READONLY.pine")
    s5_max_nqmain_overlap: float = 0.30     # Max 30% time window overlap with NQmain

    # === GPU ===
    use_gpu: bool = True
    monte_carlo_sims: int = 2000
    gpu_memory_pool_mb: int = 512           # CuPy memory pool limit

    # === Backtest date range ===
    backtest_start: str = "2011-01-01"
    backtest_end: str = "2026-12-31"

    # === Logging ===
    log_level: str = "INFO"
    log_rotate_bytes: int = 10 * 1024 * 1024  # 10MB per log file
    max_log_files: int = 10

    @classmethod
    def default(cls) -> 'MarcusConfig':
        """Create config with all defaults. Ensures directories exist."""
        config = cls()
        config._ensure_dirs()
        return config

    @classmethod
    def from_file(cls, path: str) -> 'MarcusConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        config._ensure_dirs()
        return config

    def save(self, path: Optional[str] = None):
        """Save config to JSON file."""
        target = path or os.path.join(_MARCUS_DIR, "marcus_config.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return asdict(self)

    def _ensure_dirs(self):
        """Create all required directories."""
        for d in [self.data_dir, self.reports_dir, self.logs_dir,
                  self.pine_dir, os.path.dirname(self.dashboard_path)]:
            os.makedirs(d, exist_ok=True)

    def get_s2_commission(self) -> float:
        """Stage 2 stressed commission per unit."""
        return self.s1_commission_per_unit * self.s2_commission_mult

    def get_s2_slippage(self) -> float:
        """Stage 2 stressed slippage per unit."""
        return self.s1_slippage_per_unit * self.s2_slippage_mult

    def __repr__(self):
        return (
            f"MarcusConfig(\n"
            f"  symbol={self.symbol}, interval={self.interval}\n"
            f"  cycle_interval={self.cycle_interval_minutes}min\n"
            f"  max_active={self.max_active_strategies}\n"
            f"  gpu={self.use_gpu}, mc_sims={self.monte_carlo_sims}\n"
            f"  data_dir={self.data_dir}\n"
            f"  db_path={self.db_path}\n"
            f")"
        )
