"""
Walk-Forward Optimization Analytics Module.

Provides rigorous analysis of WFO results to detect:
- Overfitting (IS vs OOS divergence)
- Parameter stability across windows
- Out-of-sample consistency
- Probability of backtest overfitting (PBO)

This helps answer: "Will my strategy work in live trading?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats as scipy_stats


@dataclass
class WFOWindow:
    """Single walk-forward window results."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    params: Dict
    is_return: float    # In-sample return
    oos_return: float   # Out-of-sample return
    is_sharpe: float = None
    oos_sharpe: float = None
    n_trades_is: int = 0
    n_trades_oos: int = 0


class WFOAnalytics:
    """
    Comprehensive Walk-Forward Optimization analysis.

    Analyzes WFO results to assess strategy robustness and detect overfitting.
    """

    def __init__(self, wfo_results: pd.DataFrame):
        """
        Args:
            wfo_results: DataFrame with columns:
                - train_start, train_end, test_end
                - params (dict or serialized)
                - train_return (in-sample)
                - test_return (out-of-sample)
        """
        self.wfo_df = wfo_results.copy()
        self.windows: List[WFOWindow] = []
        self._parse_windows()

    def _parse_windows(self):
        """Convert DataFrame rows to WFOWindow objects."""
        for i, row in self.wfo_df.iterrows():
            window = WFOWindow(
                window_id=i,
                train_start=pd.to_datetime(row.get('train_start')),
                train_end=pd.to_datetime(row.get('train_end')),
                test_end=pd.to_datetime(row.get('test_end')),
                params=row.get('params', {}),
                is_return=row.get('train_return', 0),
                oos_return=row.get('test_return', 0),
                is_sharpe=row.get('train_sharpe'),
                oos_sharpe=row.get('test_sharpe'),
                n_trades_is=row.get('train_trades', 0),
                n_trades_oos=row.get('test_trades', 0)
            )
            self.windows.append(window)

    def parameter_stability_score(self) -> Dict:
        """
        Measure how much optimal parameters change across WFO windows.

        High stability (close to 1.0) = Parameters are robust
        Low stability (close to 0.0) = Parameters are unstable (overfitting signal)

        Returns:
            Dictionary with stability metrics per parameter and overall score
        """
        if len(self.windows) < 2:
            return {'overall_stability': 1.0, 'message': 'Insufficient windows'}

        # Extract parameters from each window
        param_history = []
        for w in self.windows:
            if isinstance(w.params, dict):
                param_history.append(w.params)
            elif isinstance(w.params, str):
                # Try to parse if serialized
                try:
                    import ast
                    param_history.append(ast.literal_eval(w.params))
                except Exception:
                    continue

        if len(param_history) < 2:
            return {'overall_stability': 1.0, 'message': 'Could not parse parameters'}

        # Get all parameter names
        all_params = set()
        for p in param_history:
            all_params.update(p.keys())

        stability_scores = {}

        for param_name in all_params:
            values = []
            for p in param_history:
                if param_name in p:
                    val = p[param_name]
                    if isinstance(val, (int, float)):
                        values.append(val)

            if len(values) < 2:
                stability_scores[param_name] = 1.0
                continue

            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)

            if mean_val == 0:
                # If mean is zero, use absolute std
                stability = 1.0 if std_val < 0.01 else max(0, 1 - std_val)
            else:
                # Coefficient of variation: lower = more stable
                cv = std_val / abs(mean_val)
                stability = max(0, 1 - cv)

            stability_scores[param_name] = stability

        # Overall stability = weighted average (or min for conservative estimate)
        if stability_scores:
            overall = np.mean(list(stability_scores.values()))
        else:
            overall = 1.0

        return {
            'param_stability': stability_scores,
            'overall_stability': overall,
            'interpretation': self._interpret_stability(overall)
        }

    def _interpret_stability(self, score: float) -> str:
        """Interpret stability score."""
        if score >= 0.9:
            return "EXCELLENT - Parameters are highly stable across time"
        elif score >= 0.7:
            return "GOOD - Parameters show reasonable stability"
        elif score >= 0.5:
            return "MODERATE - Some parameter drift observed"
        elif score >= 0.3:
            return "WEAK - Significant parameter instability (overfitting risk)"
        else:
            return "POOR - Parameters are highly unstable (likely overfitting)"

    def is_oos_divergence(self) -> Dict:
        """
        Calculate divergence between in-sample and out-of-sample performance.

        Key metric for detecting overfitting.

        Returns:
            Dictionary with divergence metrics and verdict
        """
        if not self.windows:
            return {'verdict': 'NO DATA'}

        is_returns = [w.is_return for w in self.windows]
        oos_returns = [w.oos_return for w in self.windows]

        mean_is = np.mean(is_returns)
        mean_oos = np.mean(oos_returns)

        # Divergence ratio: How much does OOS underperform IS?
        if mean_is != 0:
            divergence_ratio = (mean_is - mean_oos) / abs(mean_is)
        else:
            divergence_ratio = 0

        # Degradation percentage
        degradation_pct = divergence_ratio * 100

        # Correlation between IS and OOS (should be positive for valid strategy)
        if len(is_returns) >= 3:
            is_oos_corr = np.corrcoef(is_returns, oos_returns)[0, 1]
        else:
            is_oos_corr = None

        # Verdict
        if degradation_pct < 20:
            verdict = "HEALTHY"
            explanation = "Strategy shows good IS/OOS alignment"
        elif degradation_pct < 40:
            verdict = "MODERATE"
            explanation = "Some overfitting detected but within acceptable range"
        elif degradation_pct < 60:
            verdict = "CONCERNING"
            explanation = "Significant overfitting detected - use caution"
        else:
            verdict = "OVERFIT"
            explanation = "Severe overfitting - strategy likely won't work live"

        return {
            'mean_is_return': mean_is,
            'mean_oos_return': mean_oos,
            'divergence_ratio': divergence_ratio,
            'degradation_pct': degradation_pct,
            'is_oos_correlation': is_oos_corr,
            'verdict': verdict,
            'explanation': explanation
        }

    def oos_consistency_score(self) -> Dict:
        """
        Calculate out-of-sample consistency: % of windows with positive OOS return.

        A robust strategy should be profitable in most OOS windows.

        Returns:
            Dictionary with consistency metrics
        """
        if not self.windows:
            return {'consistency': 0, 'message': 'No windows'}

        oos_returns = [w.oos_return for w in self.windows]
        n_positive = sum(1 for r in oos_returns if r > 0)
        n_total = len(oos_returns)

        consistency = n_positive / n_total

        # Consecutive wins/losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for r in oos_returns:
            if r <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        # Interpretation
        if consistency >= 0.7:
            interpretation = "STRONG - Strategy profitable in most test periods"
        elif consistency >= 0.5:
            interpretation = "MODERATE - Mixed OOS results"
        else:
            interpretation = "WEAK - Strategy often fails out-of-sample"

        return {
            'consistency': consistency,
            'n_profitable_windows': n_positive,
            'n_total_windows': n_total,
            'max_consecutive_losses': max_consecutive_losses,
            'interpretation': interpretation
        }

    def efficiency_ratio(self) -> Dict:
        """
        Calculate OOS/IS efficiency: How much of in-sample edge survives out-of-sample?

        Ratio close to 1.0 = Strategy is robust
        Ratio << 1.0 = Heavy overfitting

        Returns:
            Dictionary with efficiency metrics
        """
        if not self.windows:
            return {'efficiency': 0}

        is_returns = [w.is_return for w in self.windows]
        oos_returns = [w.oos_return for w in self.windows]

        sum_is = sum(is_returns)
        sum_oos = sum(oos_returns)

        if sum_is != 0:
            efficiency = sum_oos / sum_is
        else:
            efficiency = 1.0 if sum_oos >= 0 else 0.0

        # Per-window efficiency
        window_efficiencies = []
        for is_r, oos_r in zip(is_returns, oos_returns):
            if is_r != 0:
                window_efficiencies.append(oos_r / is_r)

        avg_window_efficiency = np.mean(window_efficiencies) if window_efficiencies else efficiency

        # Interpretation
        if efficiency >= 0.8:
            interpretation = "EXCELLENT - Strategy retains most of its edge"
        elif efficiency >= 0.5:
            interpretation = "GOOD - Reasonable edge retention"
        elif efficiency >= 0.2:
            interpretation = "FAIR - Significant edge decay"
        else:
            interpretation = "POOR - Most edge lost out-of-sample"

        return {
            'overall_efficiency': efficiency,
            'avg_window_efficiency': avg_window_efficiency,
            'interpretation': interpretation
        }

    def rolling_parameter_evolution(self) -> pd.DataFrame:
        """
        Create a DataFrame showing how parameters evolved across WFO windows.

        Returns:
            DataFrame with window_id as index and parameters as columns
        """
        if not self.windows:
            return pd.DataFrame()

        data = []
        for w in self.windows:
            row = {
                'window_id': w.window_id,
                'train_start': w.train_start,
                'is_return': w.is_return,
                'oos_return': w.oos_return
            }

            if isinstance(w.params, dict):
                row.update(w.params)
            elif isinstance(w.params, str):
                try:
                    import ast
                    row.update(ast.literal_eval(w.params))
                except Exception:
                    pass

            data.append(row)

        return pd.DataFrame(data)

    def generate_summary(self) -> Dict:
        """
        Generate comprehensive WFO analysis summary.

        Returns:
            Complete summary dictionary with all metrics
        """
        stability = self.parameter_stability_score()
        divergence = self.is_oos_divergence()
        consistency = self.oos_consistency_score()
        efficiency = self.efficiency_ratio()

        # Overall health score (0-100)
        health_components = [
            stability['overall_stability'] * 25,
            max(0, (1 - divergence['degradation_pct'] / 100)) * 25,
            consistency['consistency'] * 25,
            min(1, max(0, efficiency['overall_efficiency'])) * 25
        ]
        health_score = sum(health_components)

        # Overall verdict
        if health_score >= 80:
            overall_verdict = "ROBUST"
            overall_message = "Strategy shows strong robustness across all metrics"
        elif health_score >= 60:
            overall_verdict = "ACCEPTABLE"
            overall_message = "Strategy is reasonably robust but has some concerns"
        elif health_score >= 40:
            overall_verdict = "QUESTIONABLE"
            overall_message = "Multiple signs of overfitting - proceed with caution"
        else:
            overall_verdict = "UNRELIABLE"
            overall_message = "Strategy is likely overfit and may not work live"

        return {
            'n_windows': len(self.windows),
            'stability': stability,
            'divergence': divergence,
            'consistency': consistency,
            'efficiency': efficiency,
            'health_score': health_score,
            'overall_verdict': overall_verdict,
            'overall_message': overall_message
        }


class OverfitDetector:
    """
    Advanced overfitting detection methods.
    """

    @staticmethod
    def probability_of_backtest_overfitting(
        is_returns: np.ndarray,
        oos_returns: np.ndarray
    ) -> Dict:
        """
        Calculate Probability of Backtest Overfitting (PBO).

        Based on Bailey et al. methodology: What's the probability that the
        best in-sample performer is NOT the best out-of-sample?

        Args:
            is_returns: Array of in-sample returns for each strategy/window
            oos_returns: Array of out-of-sample returns for same strategies

        Returns:
            Dictionary with PBO and related metrics
        """
        n = len(is_returns)

        if n < 2:
            return {'pbo': 0.5, 'message': 'Insufficient data'}

        # Find best IS performer
        best_is_idx = np.argmax(is_returns)
        best_is_return = is_returns[best_is_idx]
        oos_of_best_is = oos_returns[best_is_idx]

        # Find best OOS performer
        best_oos_idx = np.argmax(oos_returns)
        best_oos_return = oos_returns[best_oos_idx]

        # PBO: Fraction of strategies that beat the "best IS" strategy in OOS
        n_beat_oos = sum(1 for r in oos_returns if r > oos_of_best_is)
        pbo = n_beat_oos / n

        # Rank correlation between IS and OOS
        if n >= 3:
            rank_corr, _ = scipy_stats.spearmanr(is_returns, oos_returns)
        else:
            rank_corr = None

        # Interpretation
        if pbo < 0.25:
            interpretation = "LOW - Best IS strategy also performs well OOS"
        elif pbo < 0.5:
            interpretation = "MODERATE - Some inconsistency between IS and OOS rankings"
        else:
            interpretation = "HIGH - IS optimization does not predict OOS performance (overfitting)"

        return {
            'pbo': pbo,
            'best_is_idx': best_is_idx,
            'best_is_return': best_is_return,
            'oos_of_best_is': oos_of_best_is,
            'best_oos_return': best_oos_return,
            'rank_correlation': rank_corr,
            'interpretation': interpretation
        }

    @staticmethod
    def deflated_performance(
        raw_sharpe: float,
        n_trials: int,
        n_observations: int
    ) -> Dict:
        """
        Calculate deflated Sharpe ratio accounting for multiple testing.

        The more strategies you test, the more likely you find one that
        looks good by chance.

        Args:
            raw_sharpe: Observed Sharpe ratio
            n_trials: Number of strategies/parameters tested
            n_observations: Number of return observations

        Returns:
            Dictionary with deflated Sharpe and adjustment factor
        """
        if n_trials <= 1:
            return {
                'raw_sharpe': raw_sharpe,
                'deflated_sharpe': raw_sharpe,
                'adjustment_factor': 1.0
            }

        # Expected Sharpe under null (all strategies have true Sharpe = 0)
        # E[max(SR)] ~ sqrt(2 * ln(N)) for N strategies
        expected_max = np.sqrt(2 * np.log(n_trials))

        # Standard error of Sharpe
        se = np.sqrt((1 + 0.5 * raw_sharpe**2) / max(1, n_observations - 1))

        # Deflated Sharpe = observed - expected_max
        # Or: probability that observed exceeds expected under null
        deflated = raw_sharpe - expected_max * se

        # Adjustment factor
        if raw_sharpe != 0:
            adjustment = deflated / raw_sharpe
        else:
            adjustment = 1.0

        # Interpretation
        if deflated > 0.5:
            interpretation = "SIGNIFICANT - Performance likely genuine"
        elif deflated > 0:
            interpretation = "MARGINAL - Some edge but may be partially luck"
        else:
            interpretation = "NOT SIGNIFICANT - Performance may be entirely due to data mining"

        return {
            'raw_sharpe': raw_sharpe,
            'deflated_sharpe': deflated,
            'expected_max_under_null': expected_max,
            'adjustment_factor': adjustment,
            'n_trials': n_trials,
            'interpretation': interpretation
        }


class ParameterSensitivityMapper:
    """
    Analyze sensitivity of performance to parameter changes.
    """

    def __init__(self, optimization_results: pd.DataFrame, metric_col: str = 'Total Return'):
        """
        Args:
            optimization_results: DataFrame from grid search with parameter columns and returns
            metric_col: Column name for the performance metric
        """
        self.results = optimization_results.copy()
        self.metric_col = metric_col

        # Identify parameter columns (non-metric columns)
        self.metric_cols = {'Total Return', 'Final Equity', 'Sharpe Ratio', 'Max Drawdown', 'Error'}
        self.param_cols = [c for c in self.results.columns if c not in self.metric_cols]

    def sensitivity_surface(self, x_param: str, y_param: str) -> Dict:
        """
        Create 2D sensitivity surface for two parameters.

        Args:
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis

        Returns:
            Dictionary with surface data for plotting
        """
        if x_param not in self.param_cols or y_param not in self.param_cols:
            return {'error': 'Invalid parameters'}

        # Pivot to create surface
        try:
            pivot = self.results.pivot_table(
                values=self.metric_col,
                index=y_param,
                columns=x_param,
                aggfunc='mean'
            )

            return {
                'x_values': pivot.columns.tolist(),
                'y_values': pivot.index.tolist(),
                'z_values': pivot.values.tolist(),
                'x_param': x_param,
                'y_param': y_param
            }
        except Exception as e:
            return {'error': str(e)}

    def cliff_detection(self) -> List[Dict]:
        """
        Identify parameter regions where small changes cause large performance drops.

        "Cliffs" are dangerous - they indicate fragile parameter regions.

        Returns:
            List of detected cliff regions
        """
        cliffs = []

        if self.results.empty or len(self.results) < 3:
            return cliffs

        # For each parameter, look for large drops between adjacent values
        for param in self.param_cols:
            param_values = sorted(self.results[param].unique())

            if len(param_values) < 2:
                continue

            # Group by this parameter (averaging over others)
            grouped = self.results.groupby(param)[self.metric_col].mean()

            for i in range(len(param_values) - 1):
                val1 = param_values[i]
                val2 = param_values[i + 1]

                if val1 not in grouped.index or val2 not in grouped.index:
                    continue

                ret1 = grouped[val1]
                ret2 = grouped[val2]

                # Calculate percentage change
                if ret1 != 0:
                    pct_change = (ret2 - ret1) / abs(ret1)
                else:
                    pct_change = ret2 - ret1

                # Flag as cliff if > 50% drop
                if abs(pct_change) > 0.5:
                    cliffs.append({
                        'parameter': param,
                        'from_value': val1,
                        'to_value': val2,
                        'from_return': ret1,
                        'to_return': ret2,
                        'pct_change': pct_change,
                        'severity': 'HIGH' if abs(pct_change) > 1.0 else 'MEDIUM'
                    })

        return sorted(cliffs, key=lambda x: abs(x['pct_change']), reverse=True)

    def robustness_score(self) -> Dict:
        """
        Calculate overall parameter robustness score.

        Based on:
        - Smoothness of performance surface (fewer cliffs = better)
        - Variance of returns across parameter space
        - How wide is the "good" region vs total space

        Returns:
            Dictionary with robustness metrics
        """
        if self.results.empty:
            return {'robustness_score': 0}

        returns = self.results[self.metric_col].values

        # 1. Variance metric (lower = more stable)
        variance = np.var(returns)
        mean_ret = np.mean(returns)
        cv = np.sqrt(variance) / abs(mean_ret) if mean_ret != 0 else 1

        # 2. Cliff count (fewer = better)
        cliffs = self.cliff_detection()
        cliff_penalty = min(1, len(cliffs) * 0.1)

        # 3. Good region fraction (higher = better)
        best_return = np.max(returns)
        threshold = best_return * 0.8  # Within 80% of best
        good_region_frac = np.mean(returns >= threshold)

        # 4. Best-to-worst ratio
        worst_return = np.min(returns)
        if worst_return != 0:
            best_worst_ratio = best_return / abs(worst_return)
        else:
            best_worst_ratio = float('inf') if best_return > 0 else 0

        # Combined score (0-100)
        cv_score = max(0, (1 - min(cv, 2) / 2)) * 30  # Lower CV is better
        cliff_score = (1 - cliff_penalty) * 20
        region_score = good_region_frac * 30
        ratio_score = min(20, best_worst_ratio * 5) if best_worst_ratio != float('inf') else 20

        total_score = cv_score + cliff_score + region_score + ratio_score

        # Interpretation
        if total_score >= 70:
            interpretation = "ROBUST - Strategy performs well across parameter space"
        elif total_score >= 50:
            interpretation = "MODERATE - Some sensitivity to parameters"
        elif total_score >= 30:
            interpretation = "SENSITIVE - Performance varies significantly with parameters"
        else:
            interpretation = "FRAGILE - Strategy highly dependent on specific parameters"

        return {
            'robustness_score': total_score,
            'coefficient_of_variation': cv,
            'n_cliffs': len(cliffs),
            'good_region_fraction': good_region_frac,
            'best_return': best_return,
            'worst_return': worst_return,
            'interpretation': interpretation
        }


def analyze_wfo_results(
    wfo_results: pd.DataFrame,
    optimization_results: pd.DataFrame = None
) -> Dict:
    """
    Main entry point for WFO analysis.

    Args:
        wfo_results: DataFrame from WalkForwardOptimizer
        optimization_results: Optional full grid search results

    Returns:
        Complete WFO analysis
    """
    # WFO Analytics
    wfo = WFOAnalytics(wfo_results)
    summary = wfo.generate_summary()

    # Parameter evolution
    param_evolution = wfo.rolling_parameter_evolution()

    result = {
        'wfo_summary': summary,
        'parameter_evolution': param_evolution.to_dict() if not param_evolution.empty else {}
    }

    # Add sensitivity analysis if optimization results provided
    if optimization_results is not None and not optimization_results.empty:
        mapper = ParameterSensitivityMapper(optimization_results)
        result['sensitivity'] = {
            'robustness': mapper.robustness_score(),
            'cliffs': mapper.cliff_detection()
        }

    # Add overfitting detection
    if len(wfo.windows) >= 2:
        is_returns = np.array([w.is_return for w in wfo.windows])
        oos_returns = np.array([w.oos_return for w in wfo.windows])

        pbo = OverfitDetector.probability_of_backtest_overfitting(is_returns, oos_returns)
        result['overfitting'] = pbo

    return result
