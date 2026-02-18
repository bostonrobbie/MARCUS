"""
Parameter Agent
===============
LLM-powered agent that analyzes optimization results and suggests
intelligent parameter grid refinements. Replaces blind grid expansion
with pattern-based reasoning.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Describes a parameter's current and suggested range."""
    name: str
    current_min: float
    current_max: float
    current_step: float
    suggested_min: Optional[float] = None
    suggested_max: Optional[float] = None
    suggested_step: Optional[float] = None
    reasoning: str = ""

    @property
    def should_refine(self) -> bool:
        """Whether this parameter should be refined."""
        return self.suggested_min is not None

    def to_grid_values(self) -> List[float]:
        """Generate grid values for this parameter."""
        if self.suggested_min is not None:
            min_val = self.suggested_min
            max_val = self.suggested_max or self.current_max
            step = self.suggested_step or self.current_step
        else:
            min_val = self.current_min
            max_val = self.current_max
            step = self.current_step

        if step <= 0:
            return [min_val]

        values = []
        val = min_val
        while val <= max_val + 1e-9:  # Small epsilon for float comparison
            values.append(round(val, 6))
            val += step
        return values


@dataclass
class GridAnalysis:
    """Analysis of optimization results."""
    total_combinations: int = 0
    best_sharpe: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    worst_sharpe: float = 0.0
    sharpe_std: float = 0.0

    # Per-parameter analysis
    param_correlations: Dict[str, float] = field(default_factory=dict)
    param_best_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Overfitting indicators
    is_isolated_peak: bool = False
    neighbor_degradation: float = 0.0
    stability_score: float = 0.0


@dataclass
class GridSuggestion:
    """Suggested next parameter grid."""
    parameters: Dict[str, ParameterRange] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.5
    action: str = "REFINE"  # REFINE, EXPAND, STOP
    timestamp: datetime = field(default_factory=datetime.now)

    def to_grid_dict(self) -> Dict[str, List[float]]:
        """Convert suggestions to grid dictionary for optimizer."""
        return {
            name: param.to_grid_values()
            for name, param in self.parameters.items()
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Action: {self.action}",
            f"Confidence: {self.confidence:.0%}",
            f"Reasoning: {self.reasoning}",
            "\nParameter Suggestions:",
        ]
        for name, param in self.parameters.items():
            if param.should_refine:
                lines.append(
                    f"  {name}: [{param.suggested_min}, {param.suggested_max}] "
                    f"step={param.suggested_step} ({param.reasoning})"
                )
            else:
                lines.append(f"  {name}: Keep current range")
        return "\n".join(lines)


class ParameterAgent:
    """
    Analyzes optimization results and suggests intelligent parameter refinements.

    Capabilities:
    - Identify which parameter ranges performed best
    - Detect overfitting patterns (isolated peaks)
    - Suggest refinement vs expansion
    - Recommend stopping when no improvement likely
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_config = self.config.get('param_agent', {})
        self.refinement_factor = self.agent_config.get('refinement_factor', 0.5)
        self.expansion_threshold = self.agent_config.get('expansion_threshold', 0.8)
        self.llm_client = LLMClient.from_config(self.config)

    def analyze_results(self, results: Any, param_grid: Dict[str, List[float]]) -> GridAnalysis:
        """
        Analyze optimization results to understand parameter space.

        Args:
            results: DataFrame or list of result dicts with 'params' and 'sharpe' keys
            param_grid: Current parameter grid used

        Returns:
            GridAnalysis with insights about the optimization
        """
        if not HAS_PANDAS:
            logger.warning("pandas not available, using basic analysis")
            return self._basic_analysis(results, param_grid)

        # Convert to DataFrame if needed
        if isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            df = results.copy()

        analysis = GridAnalysis(total_combinations=len(df))

        # Basic stats
        sharpe_col = 'sharpe' if 'sharpe' in df.columns else 'Sharpe'
        if sharpe_col not in df.columns:
            # Try to find any sharpe-like column
            sharpe_cols = [c for c in df.columns if 'sharpe' in c.lower()]
            sharpe_col = sharpe_cols[0] if sharpe_cols else None

        if sharpe_col is None:
            logger.warning("No Sharpe column found in results")
            return analysis

        analysis.best_sharpe = df[sharpe_col].max()
        analysis.worst_sharpe = df[sharpe_col].min()
        analysis.sharpe_std = df[sharpe_col].std()

        # Get best params
        best_idx = df[sharpe_col].idxmax()
        if 'params' in df.columns:
            analysis.best_params = df.loc[best_idx, 'params']
        else:
            # Extract param columns
            param_cols = [c for c in df.columns if c in param_grid.keys()]
            analysis.best_params = df.loc[best_idx, param_cols].to_dict()

        # Analyze each parameter
        for param_name, values in param_grid.items():
            if param_name not in df.columns:
                continue

            # Correlation with Sharpe
            if len(df[param_name].unique()) > 1:
                corr = df[param_name].corr(df[sharpe_col])
                analysis.param_correlations[param_name] = corr

            # Find best performing range
            top_pct = df.nlargest(max(1, len(df) // 10), sharpe_col)
            if len(top_pct) > 0:
                analysis.param_best_ranges[param_name] = (
                    top_pct[param_name].min(),
                    top_pct[param_name].max()
                )

        # Check for isolated peak (overfitting indicator)
        analysis.is_isolated_peak, analysis.neighbor_degradation = self._check_isolated_peak(
            df, sharpe_col, analysis.best_params, param_grid
        )

        # Calculate stability score
        analysis.stability_score = self._calculate_stability(df, sharpe_col, param_grid)

        return analysis

    def _basic_analysis(self, results: List[Dict], param_grid: Dict[str, List]) -> GridAnalysis:
        """Basic analysis without pandas."""
        analysis = GridAnalysis(total_combinations=len(results))

        if not results:
            return analysis

        # Find best and worst
        sharpes = [r.get('sharpe', r.get('Sharpe', 0)) for r in results]
        analysis.best_sharpe = max(sharpes) if sharpes else 0
        analysis.worst_sharpe = min(sharpes) if sharpes else 0

        best_idx = sharpes.index(analysis.best_sharpe)
        analysis.best_params = results[best_idx].get('params', {})

        return analysis

    def _check_isolated_peak(self, df: 'pd.DataFrame', sharpe_col: str,
                              best_params: Dict, param_grid: Dict) -> Tuple[bool, float]:
        """Check if the best result is an isolated peak (overfitting risk)."""
        if not best_params:
            return False, 0.0

        # Get neighbors (adjacent parameter values)
        neighbor_mask = pd.Series([True] * len(df))

        for param, best_val in best_params.items():
            if param not in df.columns or param not in param_grid:
                continue

            values = sorted(param_grid[param])
            if best_val not in values:
                continue

            idx = values.index(best_val)
            neighbors = []
            if idx > 0:
                neighbors.append(values[idx - 1])
            neighbors.append(best_val)
            if idx < len(values) - 1:
                neighbors.append(values[idx + 1])

            neighbor_mask &= df[param].isin(neighbors)

        neighbors_df = df[neighbor_mask]

        if len(neighbors_df) <= 1:
            return False, 0.0

        # Calculate degradation from peak to neighbors
        neighbor_sharpes = neighbors_df[sharpe_col]
        best_sharpe = df[sharpe_col].max()
        neighbor_mean = neighbor_sharpes[neighbor_sharpes < best_sharpe].mean()

        if pd.isna(neighbor_mean) or best_sharpe == 0:
            return False, 0.0

        degradation = (best_sharpe - neighbor_mean) / abs(best_sharpe)

        # If neighbors are >30% worse, consider it isolated
        is_isolated = degradation > 0.3

        return is_isolated, degradation

    def _calculate_stability(self, df: 'pd.DataFrame', sharpe_col: str,
                             param_grid: Dict) -> float:
        """Calculate parameter stability score (0-1, higher is more stable)."""
        if len(df) < 10:
            return 0.5

        # Stability = how consistent is performance across parameter space
        sharpe_std = df[sharpe_col].std()
        sharpe_mean = df[sharpe_col].mean()

        if sharpe_mean == 0:
            return 0.5

        # Coefficient of variation (lower = more stable)
        cv = sharpe_std / abs(sharpe_mean)

        # Convert to 0-1 score (higher = better)
        stability = max(0, min(1, 1 - cv))

        return stability

    def suggest_next_grid(self, analysis: GridAnalysis,
                          current_grid: Dict[str, List[float]],
                          iteration: int = 1) -> GridSuggestion:
        """
        Suggest the next parameter grid based on analysis.

        Args:
            analysis: Results from analyze_results()
            current_grid: Current parameter grid
            iteration: Current iteration number (for stopping logic)

        Returns:
            GridSuggestion with recommended next grid
        """
        # Build context for LLM
        if self.llm_client.is_available():
            suggestion = self._llm_suggest(analysis, current_grid, iteration)
        else:
            suggestion = self._heuristic_suggest(analysis, current_grid, iteration)

        return suggestion

    def _llm_suggest(self, analysis: GridAnalysis, current_grid: Dict[str, List],
                     iteration: int) -> GridSuggestion:
        """Use LLM to suggest next grid."""
        prompt = f"""Analyze these optimization results and suggest the next parameter grid.

## Current Iteration: {iteration}

## Results Summary
- Total combinations tested: {analysis.total_combinations}
- Best Sharpe: {analysis.best_sharpe:.3f}
- Best Parameters: {json.dumps(analysis.best_params)}
- Sharpe Std Dev: {analysis.sharpe_std:.3f}
- Stability Score: {analysis.stability_score:.2f} (1.0 = very stable)
- Isolated Peak Warning: {analysis.is_isolated_peak}
- Neighbor Degradation: {analysis.neighbor_degradation:.1%}

## Current Grid
{json.dumps({k: [round(v, 4) for v in vals] for k, vals in current_grid.items()}, indent=2)}

## Best Performing Ranges (top 10% of results)
{json.dumps({k: [round(v[0], 4), round(v[1], 4)] for k, v in analysis.param_best_ranges.items()}, indent=2)}

## Parameter-Sharpe Correlations
{json.dumps({k: round(v, 3) for k, v in analysis.param_correlations.items()}, indent=2)}

---

Based on this, recommend ONE of these actions:
1. REFINE - Narrow grid around best ranges (use when stable pattern found)
2. EXPAND - Widen grid if best values are at edges (use when best at boundary)
3. STOP - Stop optimization (use when isolated peak detected or stability low)

Respond with ONLY valid JSON:
{{
    "action": "REFINE" | "EXPAND" | "STOP",
    "confidence": 0.0-1.0,
    "reasoning": "2-3 sentence explanation",
    "parameters": {{
        "param_name": {{
            "suggested_min": number,
            "suggested_max": number,
            "suggested_step": number,
            "reasoning": "why this range"
        }}
    }}
}}"""

        response = self.llm_client.call(
            prompt=prompt,
            system="You are a quantitative researcher optimizing trading strategy parameters. Be conservative - avoid overfitting.",
            response_format="json",
        )

        if response.success and response.parsed_json:
            return self._parse_llm_suggestion(response.parsed_json, current_grid)

        # Fallback to heuristic
        return self._heuristic_suggest(analysis, current_grid, iteration)

    def _parse_llm_suggestion(self, data: Dict, current_grid: Dict) -> GridSuggestion:
        """Parse LLM response into GridSuggestion."""
        parameters = {}

        for param_name, grid_vals in current_grid.items():
            current_min = min(grid_vals)
            current_max = max(grid_vals)
            current_step = (current_max - current_min) / max(1, len(grid_vals) - 1)

            param_data = data.get('parameters', {}).get(param_name, {})

            parameters[param_name] = ParameterRange(
                name=param_name,
                current_min=current_min,
                current_max=current_max,
                current_step=current_step,
                suggested_min=param_data.get('suggested_min'),
                suggested_max=param_data.get('suggested_max'),
                suggested_step=param_data.get('suggested_step'),
                reasoning=param_data.get('reasoning', ''),
            )

        return GridSuggestion(
            parameters=parameters,
            reasoning=data.get('reasoning', ''),
            confidence=data.get('confidence', 0.5),
            action=data.get('action', 'REFINE'),
        )

    def _heuristic_suggest(self, analysis: GridAnalysis, current_grid: Dict[str, List],
                           iteration: int) -> GridSuggestion:
        """Heuristic-based suggestion without LLM."""
        parameters = {}
        action = "REFINE"
        reasoning_parts = []

        # Stop conditions
        if analysis.is_isolated_peak:
            action = "STOP"
            reasoning_parts.append("Isolated peak detected - further optimization risks overfitting")
        elif analysis.stability_score < 0.3:
            action = "STOP"
            reasoning_parts.append("Low stability score - results are too sensitive to parameters")
        elif iteration >= 3 and analysis.sharpe_std < 0.1:
            action = "STOP"
            reasoning_parts.append("Convergence detected - parameter space well explored")

        for param_name, grid_vals in current_grid.items():
            current_min = min(grid_vals)
            current_max = max(grid_vals)
            current_step = (current_max - current_min) / max(1, len(grid_vals) - 1)

            param_range = ParameterRange(
                name=param_name,
                current_min=current_min,
                current_max=current_max,
                current_step=current_step,
            )

            if action == "STOP":
                # Don't modify parameters
                pass
            elif param_name in analysis.param_best_ranges:
                best_min, best_max = analysis.param_best_ranges[param_name]
                best_val = analysis.best_params.get(param_name)

                # Check if best is at boundary
                if best_val is not None:
                    at_lower = abs(best_val - current_min) < current_step * 0.5
                    at_upper = abs(best_val - current_max) < current_step * 0.5

                    if at_lower or at_upper:
                        # Expand in that direction
                        action = "EXPAND"
                        expand_amount = (current_max - current_min) * 0.5
                        if at_lower:
                            param_range.suggested_min = current_min - expand_amount
                            param_range.suggested_max = current_max
                            param_range.reasoning = "Best at lower boundary - expanding down"
                        else:
                            param_range.suggested_min = current_min
                            param_range.suggested_max = current_max + expand_amount
                            param_range.reasoning = "Best at upper boundary - expanding up"
                        param_range.suggested_step = current_step
                    else:
                        # Refine around best range
                        range_size = best_max - best_min
                        padding = range_size * 0.2
                        param_range.suggested_min = max(current_min, best_min - padding)
                        param_range.suggested_max = min(current_max, best_max + padding)
                        param_range.suggested_step = current_step * self.refinement_factor
                        param_range.reasoning = f"Refining around best range [{best_min:.2f}, {best_max:.2f}]"

            parameters[param_name] = param_range

        if not reasoning_parts:
            if action == "EXPAND":
                reasoning_parts.append("Best parameters at grid boundary - expanding search")
            else:
                reasoning_parts.append("Refining parameter ranges around best performing region")

        return GridSuggestion(
            parameters=parameters,
            reasoning=" ".join(reasoning_parts),
            confidence=0.6 if action != "STOP" else 0.8,
            action=action,
        )

    def detect_overfitting(self, analysis: GridAnalysis) -> Tuple[bool, str]:
        """
        Check if optimization shows signs of overfitting.

        Returns:
            (is_overfit, explanation)
        """
        warnings = []

        if analysis.is_isolated_peak:
            warnings.append(
                f"Best result is an isolated peak with {analysis.neighbor_degradation:.0%} "
                f"degradation to neighbors"
            )

        if analysis.stability_score < 0.4:
            warnings.append(
                f"Low stability score ({analysis.stability_score:.2f}) - "
                f"performance varies wildly across parameter space"
            )

        if analysis.best_sharpe > 3.0 and analysis.sharpe_std < 0.5:
            warnings.append(
                "Suspiciously high Sharpe with low variance - possible data snooping"
            )

        is_overfit = len(warnings) > 0
        explanation = "; ".join(warnings) if warnings else "No overfitting indicators detected"

        return is_overfit, explanation


def analyze_and_suggest(results: Any, param_grid: Dict[str, List],
                        config: Dict[str, Any] = None,
                        iteration: int = 1) -> GridSuggestion:
    """
    Convenience function to analyze results and get next grid suggestion.

    Args:
        results: Optimization results (DataFrame or list of dicts)
        param_grid: Current parameter grid
        config: Application configuration
        iteration: Current iteration number

    Returns:
        GridSuggestion for next optimization round
    """
    agent = ParameterAgent(config)
    analysis = agent.analyze_results(results, param_grid)
    return agent.suggest_next_grid(analysis, param_grid, iteration)
