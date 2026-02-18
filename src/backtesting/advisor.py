
from typing import Dict, Any, List

class StrategyAdvisor:
    """
    Analyzes strategy performance metrics and provides constructive, natural advice 
    to improve robustness and reduce overfitting.
    """
    def __init__(self, metrics: Dict[str, Any], params: Dict[str, Any]):
        self.metrics = metrics
        self.params = params
        self.advice = []
        self.diagnoses = []

    def analyze(self) -> List[str]:
        """Runs all checks and returns a list of advice strings."""
        self.advice = []
        self.diagnoses = []
        
        self._check_sample_size()
        self._check_regime_stability()
        self._check_parameter_stability()
        self._check_risk_reward()
        self._check_overfitting_signs()
        
        return self.advice

    def _check_sample_size(self):
        trades = self.metrics.get('Total Trades', 0)
        if trades < 30:
            self.advice.append("⚠️ **Small Sample Size (<30)**: Use a longer timeframe or wider parameters to generate more trades. Results are statistically insignificant.")
        elif trades > 1000:
            self.advice.append("ℹ️ **High Frequency**: High trade count (>1000). Ensure commission costs are modeled accurately, as they will dominate returns.")

    def _check_regime_stability(self):
        regime_stats = self.metrics.get('regime_stats', {})
        if not regime_stats: return
        
        bull = regime_stats.get('Bull', {}).get('Return', 0)
        bear = regime_stats.get('Bear', {}).get('Return', 0)
        
        # Parse percentage strings if needed
        if isinstance(bull, str): bull = float(bull.strip('%')) / 100
        if isinstance(bear, str): bear = float(bear.strip('%')) / 100
        
        if bull > 0 and bear < -0.05:
            self.advice.append("📉 **Bear Market Weakness**: Strategy fails significantly in Bear markets. Consider adding a Trend Filter (e.g. Close < 200 EMA -> No Longs).")
        elif bull < 0 and bear > 0:
            self.advice.append("📈 **Bull Market Weakness**: Strategy fails in Bull markets. Are you fighting the trend?")

    def _check_parameter_stability(self):
        score = self.metrics.get('param_stability_score')
        if score is not None and score < 0.7:
            self.advice.append("🎯 **Brittle Parameters**: Detailed analysis shows neighbors in parameter space perform poorly. Your current settings might be a 'lucky peak'. Widen stop losses or reduce parameter precision.")

    def _check_risk_reward(self):
        # Infer RR from params if possible
        sl = self.params.get('sl_atr_mult')
        tp = self.params.get('tp_atr_mult')
        
        if sl and tp:
            rr = tp / sl
            if rr < 1.0:
                self.advice.append(f"⚖️ **Inverted Risk/Reward (1:{rr:.2f})**: You are risking more than you make. This requires a very high win rate (> {1/(1+rr):.0%}) to sustain.")
            elif rr > 3.0:
                self.advice.append(f"⚖️ **High Reward Target (1:{rr:.2f})**: Aggressive targets may rarely be hit. Ensure you have patience for lower win rates.")

    def _check_overfitting_signs(self):
        # Heuristic: Difference between Vector (Ideal) and Event (Real)
        diff = self.metrics.get('Discrepancy', 0)
        if diff > 0.10: # 10% return difference
            self.advice.append("🚨 **Execution Gap**: Significant drop from Vector to Event results. Your strategy might rely on unrealistic fills or liquidity.")

        # Monte Carlo
        ruin = self.metrics.get('mc_ruin_prob', 0)
        if ruin > 0.05:
            self.advice.append(f"🎲 **High Risk of Ruin ({ruin:.1%})**: Monte Carlo suggests a non-trivial chance of blowing up. Decrease position size.")

if __name__ == "__main__":
    # Test
    metrics = {
        'Total Trades': 25,
        'regime_stats': {'Bull': {'Return': '10%'}, 'Bear': {'Return': '-6%'}},
        'param_stability_score': 0.6,
        'Discrepancy': 0.02
    }
    params = {'sl_atr_mult': 2.0, 'tp_atr_mult': 1.5}
    
    advisor = StrategyAdvisor(metrics, params)
    print("\n".join(advisor.analyze()))
