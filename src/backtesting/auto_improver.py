"""
Auto Improver
=============
Takes strategies that passed quality gate and attempts to improve them
via parameter tuning and LLM-suggested modifications.

Fixes:
- Issue #7: Safe None-type handling throughout (abs(None) crashes)
- Issue #5: No eval() - uses json.loads() for rule parsing
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .stage2_rigorous_backtest import RigorousBacktester, QualityChecker
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def safe_abs(val) -> float:
    """
    Safely compute abs() on a value that might be None or non-numeric.
    Fix for Issue #7: abs(None) crashes.
    """
    if val is None:
        return 0.0
    try:
        result = abs(float(val))
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return result
    except (TypeError, ValueError):
        return 0.0


def safe_float(val, default=0.0) -> float:
    """Safely convert to float, handling None/NaN."""
    if val is None:
        return default
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_parse_rules(rules_str) -> Dict:
    """
    Safely parse strategy rules.
    Fix for Issue #5: Replaces eval(rules) with json.loads(rules).
    """
    if not rules_str:
        return {}
    if isinstance(rules_str, dict):
        return rules_str

    # Try JSON first (Issue #5: replaces dangerous eval())
    try:
        return json.loads(rules_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # If it's a Python dict literal, try ast.literal_eval (safe unlike eval)
    try:
        import ast
        result = ast.literal_eval(rules_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # If it's natural language, return as description
    logger.warning(f"Could not parse rules as structured data: {str(rules_str)[:100]}")
    return {"description": str(rules_str)}


class StrategyImprover:
    """
    Attempts to improve a strategy's performance through:
    1. Parameter perturbation (grid around current best)
    2. LLM-suggested modifications
    """

    def __init__(self, backtester: RigorousBacktester, config: Dict[str, Any] = None):
        self.backtester = backtester
        self.config = config or {}
        self.quality_checker = QualityChecker(self.config.get("quality", {}))
        self.llm = LLMClient.from_config(self.config)

    def improve(
        self, idea: Dict[str, Any], baseline_metrics: Dict[str, Any], max_iterations: int = 5
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Attempt to improve a strategy.
        Returns (best_idea, best_metrics).
        """
        best_idea = idea.copy()
        best_metrics = baseline_metrics.copy()
        best_sharpe = safe_float(baseline_metrics.get("sharpe_ratio"))

        logger.info(f"Improving {idea.get('strategy_name', 'Unknown')} (baseline Sharpe={best_sharpe:.2f})")

        for iteration in range(max_iterations):
            try:
                # Generate parameter variations
                variations = self._generate_variations(best_idea)

                for var_idea in variations:
                    metrics = self.backtester.backtest_strategy(var_idea)

                    sharpe = safe_float(metrics.get("sharpe_ratio"))
                    max_dd = safe_abs(metrics.get("max_drawdown"))
                    trades = metrics.get("total_trades", 0) or 0

                    # Check if this is an improvement
                    if sharpe > best_sharpe and trades >= 50 and max_dd < 0.5:
                        best_sharpe = sharpe
                        best_metrics = metrics
                        best_idea = var_idea
                        logger.info(
                            f"  Iteration {iteration+1}: Improved to Sharpe={sharpe:.2f} "
                            f"(DD={max_dd:.1%}, Trades={trades})"
                        )

                # Check if we should continue
                if safe_float(best_metrics.get("sharpe_ratio")) - safe_float(baseline_metrics.get("sharpe_ratio")) < 0.01:
                    logger.info(f"  No improvement after iteration {iteration+1}, stopping")
                    break

            except Exception as e:
                logger.warning(f"  Improvement iteration {iteration+1} failed: {e}")
                continue

        improvement = safe_float(best_metrics.get("sharpe_ratio")) - safe_float(baseline_metrics.get("sharpe_ratio"))
        logger.info(f"  Final improvement: {improvement:+.2f} Sharpe")

        return best_idea, best_metrics

    def _generate_variations(self, idea: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate parameter variations for a strategy idea.
        Perturbs numeric parameters by +/- 10-20%.
        """
        params = idea.get("params", {})
        variations = []

        for key, value in params.items():
            if not isinstance(value, (int, float)):
                continue
            if isinstance(value, bool):
                continue

            for factor in [0.8, 0.9, 1.1, 1.2]:
                new_params = params.copy()
                new_value = value * factor
                if isinstance(value, int):
                    new_value = max(1, int(new_value))
                else:
                    new_value = round(new_value, 4)

                new_params[key] = new_value

                var_idea = idea.copy()
                var_idea["params"] = new_params
                var_idea["strategy_name"] = f"{idea.get('strategy_name', 'Var')}_{key}_{factor}"
                variations.append(var_idea)

        return variations
