"""
Validation & Decision Agent
============================
LLM-powered agent that interprets validation results and makes deployment decisions.
Makes DEPLOY, ITERATE, or ABANDON decisions with detailed reasoning.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from .llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class Decision(Enum):
    """Strategy deployment decision."""
    DEPLOY = "DEPLOY"
    ITERATE = "ITERATE"
    ABANDON = "ABANDON"


@dataclass
class DecisionContext:
    """
    Aggregates all validation data for decision making.
    Populated from backtest results, validation suite, and robustness checks.
    """
    # Strategy identification
    strategy_name: str = ""
    symbol: str = ""
    current_params: Dict[str, Any] = field(default_factory=dict)

    # Core Performance Metrics
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0

    # Statistical Validation
    sharpe_pvalue: float = 1.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    sample_size_adequate: bool = False

    # Monte Carlo Results
    mc_risk_of_ruin: float = 1.0
    mc_var_95_drawdown: float = 0.0
    mc_median_equity: float = 0.0
    mc_simulations: int = 0

    # Regime Analysis
    regime_stats: Dict[str, Dict] = field(default_factory=dict)
    regimes_profitable: int = 0

    # Parameter Stability
    param_stability_score: float = 0.0
    is_lucky_peak: bool = True

    # Skeptic/Permutation Tests
    permutation_pvalue: float = 1.0
    detrended_profitable: bool = False

    # Walk-Forward (if available)
    wfo_oos_return: Optional[float] = None
    wfo_degradation: Optional[float] = None
    wfo_consistency: Optional[float] = None

    @classmethod
    def from_stats(cls, stats: Dict[str, Any], strategy_name: str = "",
                   symbol: str = "", params: Dict[str, Any] = None) -> 'DecisionContext':
        """Create context from backtest stats dictionary."""
        ctx = cls(
            strategy_name=strategy_name,
            symbol=symbol,
            current_params=params or {},
        )

        # Map stats to context fields
        ctx.total_return = stats.get('Total Return', stats.get('total_return', 0.0))
        ctx.cagr = stats.get('CAGR', stats.get('cagr', 0.0))
        ctx.sharpe_ratio = stats.get('Sharpe', stats.get('sharpe_ratio', 0.0))
        ctx.sortino_ratio = stats.get('Sortino', stats.get('sortino_ratio', 0.0))
        ctx.max_drawdown = stats.get('Max Drawdown', stats.get('max_drawdown', 0.0))
        ctx.profit_factor = stats.get('Profit Factor', stats.get('profit_factor', 0.0))
        ctx.win_rate = stats.get('Win Rate', stats.get('win_rate', 0.0))
        ctx.total_trades = stats.get('Total Trades', stats.get('total_trades', 0))

        # Statistical validation
        stat_sig = stats.get('statistical_significance', {})
        ctx.sharpe_pvalue = stat_sig.get('sharpe_pvalue', stats.get('sharpe_pvalue', 1.0))
        ci = stat_sig.get('sharpe_confidence_interval', (0, 0))
        if isinstance(ci, (list, tuple)) and len(ci) >= 2:
            ctx.sharpe_ci_lower, ctx.sharpe_ci_upper = ci[0], ci[1]

        # Monte Carlo
        mc = stats.get('monte_carlo', {})
        ctx.mc_risk_of_ruin = mc.get('risk_of_ruin', stats.get('mc_risk_of_ruin', 1.0))
        ctx.mc_var_95_drawdown = mc.get('var_95_drawdown', stats.get('mc_var_95_drawdown', 0.0))
        ctx.mc_median_equity = mc.get('median_final_equity', stats.get('mc_median_equity', 0.0))

        # Regime stats
        ctx.regime_stats = stats.get('regime_stats', stats.get('regime_performance', {}))
        ctx.regimes_profitable = sum(
            1 for r in ctx.regime_stats.values()
            if isinstance(r, dict) and r.get('return', r.get('total_return', 0)) > 0
        )

        # Parameter stability
        ctx.param_stability_score = stats.get('param_stability_score',
                                               stats.get('parameter_stability', 0.0))
        ctx.is_lucky_peak = ctx.param_stability_score < 0.7

        # Skeptic tests
        skeptic = stats.get('skeptic', stats.get('skeptic_results', {}))
        ctx.permutation_pvalue = skeptic.get('permutation_pvalue',
                                              skeptic.get('p_value', 1.0))
        ctx.detrended_profitable = skeptic.get('detrended_profitable', False)

        # Walk-forward
        wfo = stats.get('wfo', stats.get('walk_forward', {}))
        if wfo:
            ctx.wfo_oos_return = wfo.get('oos_return', wfo.get('out_of_sample_return'))
            ctx.wfo_degradation = wfo.get('degradation')
            ctx.wfo_consistency = wfo.get('consistency')

        ctx.sample_size_adequate = ctx.total_trades >= 100

        return ctx


@dataclass
class DecisionResult:
    """Structured output from the Decision Agent."""
    decision: Decision
    confidence: float = 0.5
    reasoning: str = ""

    # For DEPLOY decisions
    position_sizing: Optional[Dict[str, Any]] = None
    risk_limits: Optional[Dict[str, Any]] = None

    # For ITERATE decisions
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    priority_focus: Optional[str] = None

    # For ABANDON decisions
    fatal_flaws: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    rule_failures: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['decision'] = self.decision.value
        d['timestamp'] = self.timestamp.isoformat()
        return d

    def summary(self) -> str:
        """Return a concise summary of the decision."""
        lines = [
            f"Decision: {self.decision.value}",
            f"Confidence: {self.confidence:.0%}",
            f"Reasoning: {self.reasoning}",
        ]

        if self.decision == Decision.DEPLOY:
            if self.position_sizing:
                lines.append(f"Position Sizing: {self.position_sizing}")
        elif self.decision == Decision.ITERATE:
            if self.priority_focus:
                lines.append(f"Priority Focus: {self.priority_focus}")
            if self.suggestions:
                lines.append("Suggestions:")
                for s in self.suggestions[:3]:
                    lines.append(f"  - {s}")
        elif self.decision == Decision.ABANDON:
            if self.fatal_flaws:
                lines.append("Fatal Flaws:")
                for f in self.fatal_flaws:
                    lines.append(f"  - {f}")

        return "\n".join(lines)


class DecisionLogger:
    """Logs all decisions for audit trail."""

    def __init__(self, config: Dict[str, Any]):
        agent_config = config.get('decision_agent', {})
        self.enabled = agent_config.get('log_decisions', True)
        self.log_path = agent_config.get('decision_log_path', 'outputs/decision_log.json')

    def log(self, context: DecisionContext, result: DecisionResult):
        """Append decision to log file."""
        if not self.enabled:
            return

        entry = {
            'timestamp': result.timestamp.isoformat(),
            'strategy_name': context.strategy_name,
            'symbol': context.symbol,
            'decision': result.decision.value,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'model_used': result.model_used,
            'metrics': {
                'sharpe': context.sharpe_ratio,
                'total_return': context.total_return,
                'max_drawdown': context.max_drawdown,
                'total_trades': context.total_trades,
                'sharpe_pvalue': context.sharpe_pvalue,
                'risk_of_ruin': context.mc_risk_of_ruin,
                'profit_factor': context.profit_factor,
                'regimes_profitable': context.regimes_profitable,
                'param_stability': context.param_stability_score,
            },
            'rule_failures': result.rule_failures,
            'weaknesses': result.weaknesses,
            'suggestions': result.suggestions,
            'fatal_flaws': result.fatal_flaws,
            'position_sizing': result.position_sizing,
            'risk_limits': result.risk_limits,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)

        # Load existing log
        existing = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        existing.append(entry)

        # Write updated log
        with open(self.log_path, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Decision logged to {self.log_path}")


class ValidationDecisionAgent:
    """
    LLM-powered agent that makes deployment decisions for trading strategies.

    Uses Claude to reason about strategy quality and provide actionable guidance.
    Falls back to deterministic rules if API is unavailable.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_config = self.config.get('decision_agent', {})
        self.thresholds = self.agent_config.get('thresholds', {})
        self.llm_client = LLMClient.from_config(self.config)
        self.logger = DecisionLogger(self.config)

    def evaluate(self, context: DecisionContext) -> DecisionResult:
        """
        Main entry point: Evaluate a strategy and return a decision.

        Args:
            context: Aggregated validation data

        Returns:
            DecisionResult with decision, reasoning, and actionable guidance
        """
        # 1. Pre-check: Run deterministic rule checks first
        rule_failures = self._check_hard_rules(context)

        # 2. Build prompt with context and thresholds
        prompt = self._build_prompt(context, rule_failures)

        # 3. Call LLM for reasoning
        if self.llm_client.is_available():
            response = self.llm_client.call(
                prompt=prompt,
                system=self._get_system_prompt(),
                response_format="json",
            )

            if response.success and response.parsed_json:
                result = self._parse_response(response, context, rule_failures)
            else:
                logger.warning(f"LLM call failed: {response.error}")
                result = self._fallback_decision(context, rule_failures, response.error)
        else:
            logger.info("LLM not available, using deterministic fallback")
            result = self._fallback_decision(context, rule_failures, "LLM not configured")

        # 4. Log decision
        self.logger.log(context, result)

        return result

    def _check_hard_rules(self, ctx: DecisionContext) -> List[str]:
        """
        Deterministic rule checks that don't require LLM reasoning.
        Returns list of failed rules (empty = all pass).
        """
        failures = []
        t = self.thresholds

        # Sample size
        min_trades = t.get('min_trades', 100)
        if ctx.total_trades < min_trades:
            failures.append(
                f"INSUFFICIENT_SAMPLE: {ctx.total_trades} trades < {min_trades} minimum"
            )

        # Sharpe ratio
        min_sharpe = t.get('min_sharpe', 1.0)
        if ctx.sharpe_ratio < min_sharpe:
            failures.append(
                f"LOW_SHARPE: {ctx.sharpe_ratio:.2f} < {min_sharpe} threshold"
            )

        # Sharpe significance
        max_pvalue = t.get('max_sharpe_pvalue', 0.05)
        if ctx.sharpe_pvalue > max_pvalue:
            failures.append(
                f"SHARPE_NOT_SIGNIFICANT: p={ctx.sharpe_pvalue:.3f} > {max_pvalue}"
            )

        # Max drawdown (note: DD is negative)
        max_dd = t.get('max_drawdown', -0.25)
        if ctx.max_drawdown < max_dd:
            failures.append(
                f"EXCESSIVE_DRAWDOWN: {ctx.max_drawdown:.1%} worse than {max_dd:.1%} limit"
            )

        # Risk of ruin
        max_ruin = t.get('max_risk_of_ruin', 0.05)
        if ctx.mc_risk_of_ruin > max_ruin:
            failures.append(
                f"HIGH_RUIN_RISK: {ctx.mc_risk_of_ruin:.1%} > {max_ruin:.1%} threshold"
            )

        # Profit factor
        min_pf = t.get('min_profit_factor', 1.5)
        if ctx.profit_factor < min_pf:
            failures.append(
                f"LOW_PROFIT_FACTOR: {ctx.profit_factor:.2f} < {min_pf}"
            )

        # Regime robustness
        min_regimes = t.get('min_regimes_profitable', 2)
        if ctx.regimes_profitable < min_regimes:
            failures.append(
                f"REGIME_FRAGILITY: Only {ctx.regimes_profitable}/3 regimes profitable"
            )

        # Parameter stability
        min_stability = t.get('min_param_stability', 0.7)
        if ctx.param_stability_score < min_stability:
            failures.append(
                f"UNSTABLE_PARAMS: Stability {ctx.param_stability_score:.2f} < {min_stability}"
            )

        # WFO check if available
        if ctx.wfo_oos_return is not None:
            min_wfo = t.get('min_wfo_oos_return', 0.0)
            if ctx.wfo_oos_return < min_wfo:
                failures.append(
                    f"POOR_WFO_PERFORMANCE: OOS return {ctx.wfo_oos_return:.1%} < {min_wfo:.1%}"
                )

        return failures

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the LLM."""
        return """You are a quantitative trading strategy validation expert. Your role is to analyze backtest results and make deployment decisions.

You must respond with ONLY valid JSON - no markdown, no explanations outside the JSON.

Decision criteria:
- DEPLOY: All thresholds met, statistically significant, robust across regimes
- ITERATE: Some criteria met but fixable weaknesses identified
- ABANDON: Multiple critical failures, strategy not viable

Be specific and reference actual numbers. Provide actionable suggestions."""

    def _build_prompt(self, ctx: DecisionContext, rule_failures: List[str]) -> str:
        """Build the decision prompt for the LLM."""
        t = self.thresholds

        # Format regime stats
        regime_lines = []
        for regime, stats in ctx.regime_stats.items():
            if isinstance(stats, dict):
                ret = stats.get('return', stats.get('total_return', 0))
                regime_lines.append(f"  - {regime}: {ret:.1%} return")
        regime_str = "\n".join(regime_lines) if regime_lines else "  No regime data available"

        # Format rule failures
        if rule_failures:
            failures_str = "\n".join(f"  - {f}" for f in rule_failures)
        else:
            failures_str = "  All rules PASSED"

        prompt = f"""Analyze this trading strategy and make a deployment decision.

## Strategy
- Name: {ctx.strategy_name}
- Symbol: {ctx.symbol}
- Parameters: {json.dumps(ctx.current_params, indent=2) if ctx.current_params else "N/A"}

## Performance Metrics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Total Return | {ctx.total_return:.2%} | - | - |
| CAGR | {ctx.cagr:.2%} | - | - |
| Sharpe Ratio | {ctx.sharpe_ratio:.2f} | >= {t.get('min_sharpe', 1.0)} | {'PASS' if ctx.sharpe_ratio >= t.get('min_sharpe', 1.0) else 'FAIL'} |
| Sortino Ratio | {ctx.sortino_ratio:.2f} | - | - |
| Max Drawdown | {ctx.max_drawdown:.1%} | >= {t.get('max_drawdown', -0.25):.1%} | {'PASS' if ctx.max_drawdown >= t.get('max_drawdown', -0.25) else 'FAIL'} |
| Profit Factor | {ctx.profit_factor:.2f} | >= {t.get('min_profit_factor', 1.5)} | {'PASS' if ctx.profit_factor >= t.get('min_profit_factor', 1.5) else 'FAIL'} |
| Win Rate | {ctx.win_rate:.1%} | - | - |
| Total Trades | {ctx.total_trades} | >= {t.get('min_trades', 100)} | {'PASS' if ctx.total_trades >= t.get('min_trades', 100) else 'FAIL'} |

## Statistical Validation
- Sharpe p-value: {ctx.sharpe_pvalue:.4f} (threshold: < {t.get('max_sharpe_pvalue', 0.05)})
- Sharpe 95% CI: [{ctx.sharpe_ci_lower:.2f}, {ctx.sharpe_ci_upper:.2f}]
- Permutation p-value: {ctx.permutation_pvalue:.2f}

## Monte Carlo (simulations: {ctx.mc_simulations or 'N/A'})
- Risk of Ruin: {ctx.mc_risk_of_ruin:.1%} (threshold: < {t.get('max_risk_of_ruin', 0.05):.1%})
- 95% VaR Drawdown: {ctx.mc_var_95_drawdown:.1%}

## Regime Performance
{regime_str}
- Regimes Profitable: {ctx.regimes_profitable}/3 (threshold: >= {t.get('min_regimes_profitable', 2)})

## Parameter Stability
- Score: {ctx.param_stability_score:.2f} (threshold: >= {t.get('min_param_stability', 0.7)})
- Assessment: {'Robust' if not ctx.is_lucky_peak else 'WARNING: May be lucky peak'}

## Walk-Forward Results
{f'- OOS Return: {ctx.wfo_oos_return:.2%}' if ctx.wfo_oos_return is not None else '- Not performed'}
{f'- Degradation: {ctx.wfo_degradation:.1%}' if ctx.wfo_degradation is not None else ''}

## Rule Check Results
{failures_str}

---

Respond with ONLY this JSON format:
{{
    "decision": "DEPLOY" | "ITERATE" | "ABANDON",
    "confidence": 0.0-1.0,
    "reasoning": "2-3 sentence explanation",
    "position_sizing": {{"max_risk_pct": number, "kelly_fraction": number}} | null,
    "risk_limits": {{"daily_loss_limit_pct": number, "max_drawdown_halt_pct": number}} | null,
    "weaknesses": ["weakness1", ...] | [],
    "suggestions": ["actionable suggestion1", ...] | [],
    "priority_focus": "single most important thing to fix" | null,
    "fatal_flaws": ["flaw1", ...] | []
}}"""

        return prompt

    def _parse_response(self, response: LLMResponse, context: DecisionContext,
                        rule_failures: List[str]) -> DecisionResult:
        """Parse LLM response into DecisionResult."""
        data = response.parsed_json

        try:
            decision = Decision(data.get('decision', 'ITERATE'))
        except ValueError:
            decision = Decision.ITERATE

        return DecisionResult(
            decision=decision,
            confidence=float(data.get('confidence', 0.5)),
            reasoning=data.get('reasoning', ''),
            position_sizing=data.get('position_sizing'),
            risk_limits=data.get('risk_limits'),
            weaknesses=data.get('weaknesses', []),
            suggestions=data.get('suggestions', []),
            priority_focus=data.get('priority_focus'),
            fatal_flaws=data.get('fatal_flaws', []),
            timestamp=datetime.now(),
            model_used=response.model or self.config.get('llm', {}).get('model', 'unknown'),
            rule_failures=rule_failures,
            raw_response=response.content,
        )

    def _fallback_decision(self, context: DecisionContext, rule_failures: List[str],
                           error: str = None) -> DecisionResult:
        """Generate deterministic decision when LLM is unavailable."""
        num_failures = len(rule_failures)

        if num_failures == 0:
            decision = Decision.DEPLOY
            confidence = 0.6
            reasoning = "All deterministic rules passed. LLM unavailable for detailed analysis."
            # Generate default position sizing
            pos_config = self.agent_config.get('position_sizing', {})
            position_sizing = {
                'max_risk_pct': pos_config.get('max_risk_per_trade_pct', 2.0),
                'kelly_fraction': pos_config.get('kelly_fraction', 0.25),
            }
            risk_limits = {
                'daily_loss_limit_pct': 5.0,
                'max_drawdown_halt_pct': 15.0,
            }
        elif num_failures <= 2:
            decision = Decision.ITERATE
            confidence = 0.5
            reasoning = f"{num_failures} rule(s) failed. Strategy shows potential but needs improvement."
            position_sizing = None
            risk_limits = None
        else:
            decision = Decision.ABANDON
            confidence = 0.7
            reasoning = f"{num_failures} critical failures detected. Strategy unlikely to be viable."
            position_sizing = None
            risk_limits = None

        # Generate suggestions based on failures
        suggestions = []
        for failure in rule_failures:
            if 'SHARPE' in failure:
                suggestions.append("Improve risk-adjusted returns by tightening stops or filtering trades")
            elif 'DRAWDOWN' in failure:
                suggestions.append("Reduce position size or add drawdown-based position scaling")
            elif 'RUIN' in failure:
                suggestions.append("Reduce risk per trade and add portfolio-level risk limits")
            elif 'REGIME' in failure:
                suggestions.append("Add trend filter to avoid trading in unfavorable market conditions")
            elif 'PARAM' in failure:
                suggestions.append("Test neighboring parameter values to ensure robustness")
            elif 'SAMPLE' in failure:
                suggestions.append("Extend backtest period or use more granular data")
            elif 'PROFIT_FACTOR' in failure:
                suggestions.append("Improve trade selection or adjust take-profit levels")

        return DecisionResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            position_sizing=position_sizing,
            risk_limits=risk_limits,
            weaknesses=rule_failures,
            suggestions=suggestions[:3],
            priority_focus=suggestions[0] if suggestions else None,
            fatal_flaws=rule_failures if decision == Decision.ABANDON else [],
            timestamp=datetime.now(),
            model_used="FALLBACK_RULES",
            rule_failures=rule_failures,
            raw_response=error or "Fallback to deterministic rules",
        )


def evaluate_strategy(stats: Dict[str, Any], strategy_name: str = "",
                      symbol: str = "", params: Dict[str, Any] = None,
                      config: Dict[str, Any] = None) -> DecisionResult:
    """
    Convenience function to evaluate a strategy from stats dictionary.

    Args:
        stats: Backtest statistics dictionary
        strategy_name: Name of the strategy
        symbol: Trading symbol
        params: Strategy parameters
        config: Application configuration

    Returns:
        DecisionResult with deployment decision
    """
    context = DecisionContext.from_stats(stats, strategy_name, symbol, params)
    agent = ValidationDecisionAgent(config)
    return agent.evaluate(context)
