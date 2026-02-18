"""
Logic Agent
===========
LLM-powered agent that analyzes strategy weaknesses and proposes
new entry/exit conditions, filters, and indicator combinations.
Generates vectorized implementation code for proposed improvements.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import re

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeakness:
    """Identified weakness in strategy performance."""
    category: str  # REGIME, TIMING, FILTER, RISK_MANAGEMENT, etc.
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    affected_trades_pct: float = 0.0


@dataclass
class StrategyHypothesis:
    """Proposed improvement to strategy logic."""
    name: str
    description: str
    category: str  # FILTER, ENTRY, EXIT, RISK
    addresses_weakness: str
    expected_impact: str
    implementation_code: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    complexity: str = "MEDIUM"  # LOW, MEDIUM, HIGH


@dataclass
class LogicSuggestion:
    """Complete suggestion from Logic Agent."""
    weaknesses: List[StrategyWeakness] = field(default_factory=list)
    hypotheses: List[StrategyHypothesis] = field(default_factory=list)
    priority_hypothesis: Optional[StrategyHypothesis] = None
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = ""

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 50,
            "STRATEGY LOGIC ANALYSIS",
            "=" * 50,
            "",
            "WEAKNESSES IDENTIFIED:",
        ]

        for w in self.weaknesses:
            lines.append(f"  [{w.severity}] {w.category}: {w.description}")
            if w.affected_trades_pct > 0:
                lines.append(f"      Affects ~{w.affected_trades_pct:.0%} of trades")

        lines.extend(["", "PROPOSED IMPROVEMENTS:"])

        for i, h in enumerate(self.hypotheses, 1):
            lines.append(f"  {i}. {h.name} ({h.category})")
            lines.append(f"     {h.description}")
            lines.append(f"     Expected impact: {h.expected_impact}")
            lines.append(f"     Confidence: {h.confidence:.0%}")

        if self.priority_hypothesis:
            lines.extend([
                "",
                "RECOMMENDED FIRST: " + self.priority_hypothesis.name,
                f"  {self.priority_hypothesis.description}",
            ])

        return "\n".join(lines)


class LogicAgent:
    """
    Analyzes strategy weaknesses and proposes logic improvements.

    Capabilities:
    - Analyze regime-specific failures
    - Identify timing issues
    - Propose new filters (volume, volatility, trend)
    - Suggest indicator combinations
    - Generate vectorized implementation code
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_config = self.config.get('logic_agent', {})
        self.max_suggestions = self.agent_config.get('max_filter_suggestions', 3)
        self.generate_code = self.agent_config.get('generate_code', True)
        self.llm_client = LLMClient.from_config(self.config)

    def analyze_weaknesses(self, stats: Dict[str, Any],
                           trade_analysis: Optional[Dict] = None) -> List[StrategyWeakness]:
        """
        Analyze strategy statistics to identify weaknesses.

        Args:
            stats: Backtest statistics
            trade_analysis: Optional detailed trade-level analysis

        Returns:
            List of identified weaknesses
        """
        weaknesses = []

        # Analyze regime performance
        regime_stats = stats.get('regime_stats', stats.get('regime_performance', {}))
        if regime_stats:
            weaknesses.extend(self._analyze_regime_weakness(regime_stats))

        # Analyze drawdowns
        max_dd = stats.get('Max Drawdown', stats.get('max_drawdown', 0))
        if max_dd < -0.20:  # More than 20% drawdown
            weaknesses.append(StrategyWeakness(
                category="RISK_MANAGEMENT",
                description=f"Excessive drawdown of {max_dd:.1%} indicates poor risk control",
                evidence={'max_drawdown': max_dd},
                severity="HIGH",
            ))

        # Analyze win rate vs profit factor
        win_rate = stats.get('Win Rate', stats.get('win_rate', 0))
        profit_factor = stats.get('Profit Factor', stats.get('profit_factor', 0))

        if win_rate > 0.6 and profit_factor < 1.5:
            weaknesses.append(StrategyWeakness(
                category="EXIT",
                description="High win rate but low profit factor - winners may be cut too early",
                evidence={'win_rate': win_rate, 'profit_factor': profit_factor},
                severity="MEDIUM",
            ))
        elif win_rate < 0.4 and profit_factor < 2.0:
            weaknesses.append(StrategyWeakness(
                category="FILTER",
                description="Low win rate with modest profit factor - entry filter may be too loose",
                evidence={'win_rate': win_rate, 'profit_factor': profit_factor},
                severity="HIGH",
            ))

        # Analyze Sharpe
        sharpe = stats.get('Sharpe', stats.get('sharpe_ratio', 0))
        if sharpe < 0.5:
            weaknesses.append(StrategyWeakness(
                category="GENERAL",
                description="Low risk-adjusted returns indicate fundamental strategy issues",
                evidence={'sharpe': sharpe},
                severity="HIGH",
            ))

        # Analyze losing trades if available
        if trade_analysis:
            top_losers = trade_analysis.get('top_losers', [])
            if top_losers:
                weaknesses.extend(self._analyze_losing_trades(top_losers))

        return weaknesses

    def _analyze_regime_weakness(self, regime_stats: Dict) -> List[StrategyWeakness]:
        """Analyze regime-specific weaknesses."""
        weaknesses = []

        for regime, stats in regime_stats.items():
            if not isinstance(stats, dict):
                continue

            ret = stats.get('return', stats.get('total_return', 0))
            sharpe = stats.get('sharpe', stats.get('sharpe_ratio', 0))
            trade_count = stats.get('trades', stats.get('trade_count', 0))

            if ret < -0.05 and trade_count > 10:
                weaknesses.append(StrategyWeakness(
                    category="REGIME",
                    description=f"Strategy loses money in {regime} market conditions ({ret:.1%})",
                    evidence={'regime': regime, 'return': ret, 'trades': trade_count},
                    severity="HIGH",
                    affected_trades_pct=trade_count / 100 if trade_count else 0,
                ))
            elif sharpe < 0 and trade_count > 5:
                weaknesses.append(StrategyWeakness(
                    category="REGIME",
                    description=f"Negative risk-adjusted returns in {regime} conditions",
                    evidence={'regime': regime, 'sharpe': sharpe},
                    severity="MEDIUM",
                ))

        return weaknesses

    def _analyze_losing_trades(self, top_losers: List[Dict]) -> List[StrategyWeakness]:
        """Analyze patterns in losing trades."""
        weaknesses = []

        if not top_losers:
            return weaknesses

        # Check for common patterns
        regimes = [t.get('regime', 'unknown') for t in top_losers]
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # If >60% of losers in one regime
        total = len(top_losers)
        for regime, count in regime_counts.items():
            if count / total > 0.6:
                weaknesses.append(StrategyWeakness(
                    category="REGIME",
                    description=f"{count / total:.0%} of worst losses occurred in {regime} conditions",
                    evidence={'regime': regime, 'loss_concentration': count / total},
                    severity="HIGH",
                ))

        return weaknesses

    def propose_improvements(self, weaknesses: List[StrategyWeakness],
                             strategy_code: Optional[str] = None) -> List[StrategyHypothesis]:
        """
        Propose strategy improvements based on identified weaknesses.

        Args:
            weaknesses: List of identified weaknesses
            strategy_code: Optional current strategy source code

        Returns:
            List of proposed improvements
        """
        if not weaknesses:
            return []

        hypotheses = []

        if self.llm_client.is_available():
            hypotheses = self._llm_propose(weaknesses, strategy_code)

        # Fall back to heuristics if LLM unavailable or returned no results
        if not hypotheses:
            hypotheses = self._heuristic_propose(weaknesses)

        return hypotheses[:self.max_suggestions]

    def _llm_propose(self, weaknesses: List[StrategyWeakness],
                     strategy_code: Optional[str]) -> List[StrategyHypothesis]:
        """Use LLM to propose improvements."""
        weakness_text = "\n".join([
            f"- [{w.severity}] {w.category}: {w.description}"
            for w in weaknesses
        ])

        code_context = ""
        if strategy_code:
            # Truncate for token limits
            code_context = f"""
## Current Strategy Code (excerpt)
```python
{strategy_code[:4000]}
```
"""

        prompt = f"""Analyze these trading strategy weaknesses and propose specific improvements.

## Identified Weaknesses
{weakness_text}

{code_context}

## Requirements
1. Propose 2-3 specific, actionable improvements
2. Each should directly address one or more weaknesses
3. Include vectorized Python implementation code (using pandas/numpy)
4. Consider: filters, entry conditions, exit rules, position sizing

Respond with ONLY valid JSON:
{{
    "hypotheses": [
        {{
            "name": "Short descriptive name",
            "description": "What this improvement does",
            "category": "FILTER" | "ENTRY" | "EXIT" | "RISK",
            "addresses_weakness": "Which weakness this fixes",
            "expected_impact": "Expected improvement",
            "confidence": 0.0-1.0,
            "complexity": "LOW" | "MEDIUM" | "HIGH",
            "parameters": {{"param_name": default_value}},
            "implementation_code": "vectorized Python code snippet"
        }}
    ],
    "priority_index": 0,
    "reasoning": "Why these improvements were chosen"
}}"""

        response = self.llm_client.call(
            prompt=prompt,
            system="You are a quantitative strategy developer. Propose practical, testable improvements with working code.",
            response_format="json",
        )

        hypotheses = []
        priority_idx = 0

        if response.success and response.parsed_json:
            data = response.parsed_json
            priority_idx = data.get('priority_index', 0)

            for h_data in data.get('hypotheses', []):
                hypotheses.append(StrategyHypothesis(
                    name=h_data.get('name', 'Unnamed'),
                    description=h_data.get('description', ''),
                    category=h_data.get('category', 'FILTER'),
                    addresses_weakness=h_data.get('addresses_weakness', ''),
                    expected_impact=h_data.get('expected_impact', ''),
                    implementation_code=h_data.get('implementation_code') if self.generate_code else None,
                    parameters=h_data.get('parameters', {}),
                    confidence=h_data.get('confidence', 0.5),
                    complexity=h_data.get('complexity', 'MEDIUM'),
                ))

        # Mark priority
        if hypotheses and 0 <= priority_idx < len(hypotheses):
            hypotheses[priority_idx].confidence = min(1.0, hypotheses[priority_idx].confidence + 0.1)

        return hypotheses

    def _heuristic_propose(self, weaknesses: List[StrategyWeakness]) -> List[StrategyHypothesis]:
        """Heuristic-based proposals without LLM."""
        hypotheses = []

        for w in weaknesses:
            if w.category == "REGIME":
                # Propose trend filter
                hypotheses.append(StrategyHypothesis(
                    name="Trend Filter",
                    description="Add trend filter to avoid trading against major trend",
                    category="FILTER",
                    addresses_weakness=w.description,
                    expected_impact="Avoid losses in unfavorable regime conditions",
                    implementation_code=self._generate_trend_filter_code(),
                    parameters={'trend_period': 50, 'trend_threshold': 0},
                    confidence=0.7,
                    complexity="LOW",
                ))

            elif w.category == "RISK_MANAGEMENT":
                # Propose ATR-based position sizing
                hypotheses.append(StrategyHypothesis(
                    name="Volatility-Adjusted Position Size",
                    description="Scale position size inversely to volatility",
                    category="RISK",
                    addresses_weakness=w.description,
                    expected_impact="Reduce drawdowns during high volatility",
                    implementation_code=self._generate_vol_sizing_code(),
                    parameters={'atr_period': 14, 'risk_target': 0.02},
                    confidence=0.8,
                    complexity="MEDIUM",
                ))

            elif w.category == "FILTER":
                # Propose volume filter
                hypotheses.append(StrategyHypothesis(
                    name="Volume Confirmation Filter",
                    description="Only take trades when volume confirms the move",
                    category="FILTER",
                    addresses_weakness=w.description,
                    expected_impact="Improve win rate by filtering low-conviction setups",
                    implementation_code=self._generate_volume_filter_code(),
                    parameters={'volume_sma_period': 20, 'volume_threshold': 1.2},
                    confidence=0.6,
                    complexity="LOW",
                ))

            elif w.category == "EXIT":
                # Propose trailing stop
                hypotheses.append(StrategyHypothesis(
                    name="ATR Trailing Stop",
                    description="Use trailing stop to let winners run",
                    category="EXIT",
                    addresses_weakness=w.description,
                    expected_impact="Increase profit factor by capturing more of winning moves",
                    implementation_code=self._generate_trailing_stop_code(),
                    parameters={'atr_period': 14, 'atr_mult': 2.0},
                    confidence=0.7,
                    complexity="MEDIUM",
                ))

            elif w.category == "GENERAL":
                # General weakness - propose comprehensive filter
                hypotheses.append(StrategyHypothesis(
                    name="Multi-Factor Filter",
                    description="Add multiple confirmation filters to improve signal quality",
                    category="FILTER",
                    addresses_weakness=w.description,
                    expected_impact="Improve overall risk-adjusted returns",
                    implementation_code=self._generate_trend_filter_code(),
                    parameters={'trend_period': 50, 'trend_threshold': 0},
                    confidence=0.5,
                    complexity="MEDIUM",
                ))

        # Deduplicate by name
        seen_names = set()
        unique = []
        for h in hypotheses:
            if h.name not in seen_names:
                seen_names.add(h.name)
                unique.append(h)

        return unique

    def _generate_trend_filter_code(self) -> str:
        """Generate vectorized trend filter code."""
        return '''
# Trend Filter - Only trade in direction of major trend
def apply_trend_filter(df, trend_period=50, trend_threshold=0):
    """
    Filter signals to only trade with the trend.

    Args:
        df: DataFrame with 'Close' and 'signal' columns
        trend_period: SMA period for trend detection
        trend_threshold: Minimum distance above/below SMA

    Returns:
        DataFrame with filtered signals
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Calculate trend
    df['trend_sma'] = df['Close'].rolling(trend_period).mean()
    df['trend'] = np.where(df['Close'] > df['trend_sma'] + trend_threshold, 1,
                          np.where(df['Close'] < df['trend_sma'] - trend_threshold, -1, 0))

    # Filter signals: only long in uptrend, only short in downtrend
    df['filtered_signal'] = np.where(
        (df['signal'] == 1) & (df['trend'] >= 0), 1,
        np.where((df['signal'] == -1) & (df['trend'] <= 0), -1, 0)
    )

    return df
'''

    def _generate_vol_sizing_code(self) -> str:
        """Generate volatility-adjusted position sizing code."""
        return '''
# Volatility-Adjusted Position Sizing
def calculate_position_size(df, atr_period=14, risk_target=0.02, account_size=100000):
    """
    Calculate position size based on volatility.

    Args:
        df: DataFrame with OHLC data
        atr_period: ATR calculation period
        risk_target: Target risk per trade as fraction of account
        account_size: Trading account size

    Returns:
        Series with position sizes
    """
    import pandas as pd
    import numpy as np

    # Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # Position size = (Account * Risk Target) / (ATR * Multiplier)
    dollar_risk = account_size * risk_target
    position_size = dollar_risk / (atr * 2)  # 2 ATR stop

    # Cap at reasonable maximum
    max_size = account_size * 0.1 / df['Close']  # Max 10% of account
    position_size = np.minimum(position_size, max_size)

    return position_size
'''

    def _generate_volume_filter_code(self) -> str:
        """Generate volume confirmation filter code."""
        return '''
# Volume Confirmation Filter
def apply_volume_filter(df, volume_sma_period=20, volume_threshold=1.2):
    """
    Filter signals to only trade when volume confirms.

    Args:
        df: DataFrame with 'Volume' and 'signal' columns
        volume_sma_period: Period for volume moving average
        volume_threshold: Minimum relative volume (1.2 = 20% above average)

    Returns:
        DataFrame with filtered signals
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Calculate relative volume
    df['volume_sma'] = df['Volume'].rolling(volume_sma_period).mean()
    df['rvol'] = df['Volume'] / df['volume_sma']

    # Only trade when volume is above threshold
    df['volume_confirmed'] = df['rvol'] >= volume_threshold

    # Apply filter
    df['filtered_signal'] = np.where(df['volume_confirmed'], df['signal'], 0)

    return df
'''

    def _generate_trailing_stop_code(self) -> str:
        """Generate ATR trailing stop code."""
        return '''
# ATR Trailing Stop
def apply_trailing_stop(df, atr_period=14, atr_mult=2.0):
    """
    Apply ATR-based trailing stop to positions.

    Args:
        df: DataFrame with OHLC and 'position' columns
        atr_period: ATR calculation period
        atr_mult: ATR multiplier for stop distance

    Returns:
        DataFrame with adjusted positions (exited on stop)
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()

    # Track trailing stop level
    df['trail_stop'] = np.nan

    position = 0
    entry_price = 0
    highest_since_entry = 0
    lowest_since_entry = float('inf')

    for i in range(len(df)):
        if df['position'].iloc[i] != 0 and position == 0:
            # New position
            position = df['position'].iloc[i]
            entry_price = df['Close'].iloc[i]
            highest_since_entry = df['High'].iloc[i]
            lowest_since_entry = df['Low'].iloc[i]
        elif position != 0:
            # Update trailing levels
            highest_since_entry = max(highest_since_entry, df['High'].iloc[i])
            lowest_since_entry = min(lowest_since_entry, df['Low'].iloc[i])

            atr = df['atr'].iloc[i]

            if position > 0:  # Long position
                stop_level = highest_since_entry - atr_mult * atr
                if df['Low'].iloc[i] <= stop_level:
                    df.loc[df.index[i], 'position'] = 0  # Exit
                    position = 0
            else:  # Short position
                stop_level = lowest_since_entry + atr_mult * atr
                if df['High'].iloc[i] >= stop_level:
                    df.loc[df.index[i], 'position'] = 0  # Exit
                    position = 0

    return df
'''

    def generate_full_suggestion(self, stats: Dict[str, Any],
                                 trade_analysis: Optional[Dict] = None,
                                 strategy_code: Optional[str] = None) -> LogicSuggestion:
        """
        Full analysis and suggestion generation.

        Args:
            stats: Backtest statistics
            trade_analysis: Optional trade-level analysis
            strategy_code: Optional strategy source code

        Returns:
            Complete LogicSuggestion with weaknesses and hypotheses
        """
        # Analyze weaknesses
        weaknesses = self.analyze_weaknesses(stats, trade_analysis)

        # Propose improvements
        hypotheses = self.propose_improvements(weaknesses, strategy_code)

        # Determine priority
        priority = None
        if hypotheses:
            # Highest confidence hypothesis addressing highest severity weakness
            high_sev_weaknesses = [w.description for w in weaknesses if w.severity == "HIGH"]
            for h in hypotheses:
                if any(w in h.addresses_weakness for w in high_sev_weaknesses):
                    priority = h
                    break
            if not priority:
                priority = hypotheses[0]

        return LogicSuggestion(
            weaknesses=weaknesses,
            hypotheses=hypotheses,
            priority_hypothesis=priority,
            reasoning=f"Identified {len(weaknesses)} weaknesses, proposed {len(hypotheses)} improvements",
            model_used=self.config.get('llm', {}).get('model', 'heuristic'),
        )


def analyze_and_suggest(stats: Dict[str, Any],
                        trade_analysis: Optional[Dict] = None,
                        strategy_code: Optional[str] = None,
                        config: Dict[str, Any] = None) -> LogicSuggestion:
    """
    Convenience function for strategy logic analysis.

    Args:
        stats: Backtest statistics
        trade_analysis: Optional trade analysis data
        strategy_code: Optional strategy source code
        config: Application configuration

    Returns:
        LogicSuggestion with weaknesses and improvement hypotheses
    """
    agent = LogicAgent(config)
    return agent.generate_full_suggestion(stats, trade_analysis, strategy_code)
