"""
Insights Engine and Dashboard for Strategy Analysis.

Generates human-readable explanations of strategy performance:
- Executive summary in plain English
- Edge explanation (where does the alpha come from?)
- Risk assessment with context
- Actionable recommendations
- WFO health report

Outputs to HTML, Console, and Markdown formats.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class InsightsEngine:
    """
    Generates plain-language insights from strategy analysis.
    """

    def __init__(
        self,
        stats: Dict,
        attribution: Dict = None,
        wfo_analytics: Dict = None,
        stat_significance: Dict = None,
        regime_stats: Dict = None,
        ai_decision: Dict = None,
        iteration_history: Dict = None,
        code_review: Dict = None
    ):
        """
        Args:
            stats: Basic performance statistics from TearSheet
            attribution: Trade attribution analysis results
            wfo_analytics: Walk-forward optimization analysis
            stat_significance: Statistical significance analysis
            regime_stats: Performance breakdown by market regime
        """
        self.stats = stats or {}
        self.attribution = attribution or {}
        self.wfo = wfo_analytics or {}
        self.significance = stat_significance or {}
        self.regime_stats = regime_stats or {}
        self.ai_decision = ai_decision or {}
        self.iteration_history = iteration_history or {}
        self.code_review = code_review or {}

    def generate_executive_summary(self) -> str:
        """
        Generate a plain-English executive summary of strategy performance.
        """
        lines = []

        # Total return and timeframe
        total_return = self.stats.get('Total Return', 0)
        sharpe = self.stats.get('Sharpe Ratio', 0)
        max_dd = self.stats.get('Max Drawdown', 0)
        final_equity = self.stats.get('Ending Equity', 0)

        # Performance summary
        if total_return > 0:
            lines.append(f"Your strategy generated a {total_return:.1%} total return")
        else:
            lines.append(f"Your strategy lost {abs(total_return):.1%}")

        if final_equity:
            lines.append(f"with a final equity of ${final_equity:,.0f}.")
        else:
            lines[-1] += "."

        # Risk-adjusted performance
        if sharpe > 1.5:
            lines.append(f"The risk-adjusted performance is excellent (Sharpe ratio: {sharpe:.2f}).")
        elif sharpe > 1.0:
            lines.append(f"The risk-adjusted performance is good (Sharpe ratio: {sharpe:.2f}).")
        elif sharpe > 0.5:
            lines.append(f"The risk-adjusted performance is moderate (Sharpe ratio: {sharpe:.2f}).")
        elif sharpe > 0:
            lines.append(f"The risk-adjusted performance is weak (Sharpe ratio: {sharpe:.2f}).")
        else:
            lines.append(f"The risk-adjusted performance is negative (Sharpe ratio: {sharpe:.2f}).")

        # Drawdown context
        if max_dd < -0.30:
            lines.append(f"However, the maximum drawdown of {max_dd:.1%} is severe and may be psychologically difficult to tolerate.")
        elif max_dd < -0.20:
            lines.append(f"The maximum drawdown of {max_dd:.1%} is significant but manageable for most investors.")
        elif max_dd < -0.10:
            lines.append(f"The maximum drawdown of {max_dd:.1%} is relatively contained.")
        elif max_dd < 0:
            lines.append(f"The maximum drawdown of {max_dd:.1%} is impressively small.")

        # Statistical significance
        if self.significance:
            sharpe_pval = self.significance.get('sharpe_pvalue')
            if sharpe_pval is not None:
                if sharpe_pval < 0.05:
                    lines.append(f"This performance is statistically significant (p={sharpe_pval:.3f}).")
                else:
                    lines.append(f"Note: This performance is NOT statistically significant (p={sharpe_pval:.2f}). Results may be due to chance.")

            ci_lower = self.significance.get('sharpe_ci_lower')
            ci_upper = self.significance.get('sharpe_ci_upper')
            if ci_lower is not None and ci_upper is not None:
                if ci_lower < 0 < ci_upper:
                    lines.append(f"The 95% confidence interval for Sharpe ratio [{ci_lower:.2f}, {ci_upper:.2f}] includes zero, indicating uncertainty.")

        # WFO verdict
        if self.wfo and 'wfo_summary' in self.wfo:
            wfo_verdict = self.wfo['wfo_summary'].get('overall_verdict', '')
            wfo_msg = self.wfo['wfo_summary'].get('overall_message', '')
            if wfo_verdict:
                lines.append(f"Walk-forward analysis: {wfo_verdict}. {wfo_msg}")

        return " ".join(lines)

    def generate_edge_explanation(self) -> str:
        """
        Explain where the strategy's edge comes from.
        """
        if not self.attribution:
            return "Insufficient data for edge attribution analysis."

        lines = []
        edge = self.attribution.get('edge_breakdown', {})

        if not edge:
            return "No edge breakdown available."

        # Entry vs Exit contribution
        entry_contrib = edge.get('entry_contribution_pct', 0)
        exit_contrib = edge.get('exit_contribution_pct', 0)
        selection_contrib = edge.get('selection_contribution_pct', 0)

        # Determine primary edge source
        contribs = [
            ('entry timing', entry_contrib),
            ('exit timing', exit_contrib),
            ('trade selection/regime', selection_contrib)
        ]
        contribs.sort(key=lambda x: x[1], reverse=True)

        primary = contribs[0]
        lines.append(f"Your primary edge comes from {primary[0]} (contributing approximately {primary[1]:.0f}% of alpha).")

        # Secondary contributors
        if contribs[1][1] > 20:
            lines.append(f"{contribs[1][0].capitalize()} also contributes ({contribs[1][1]:.0f}%).")

        # Entry quality insight
        avg_entry = edge.get('avg_entry_quality', 0)
        if avg_entry > 0.7:
            lines.append("Entry execution is excellent - you're consistently getting good fills.")
        elif avg_entry < 0.4:
            lines.append("Entry execution could be improved - you're often buying near bar highs or selling near lows.")

        # Exit quality insight
        avg_exit = edge.get('avg_exit_quality', 0)
        if avg_exit > 0.7:
            lines.append("Exit timing is strong.")
        elif avg_exit < 0.4:
            lines.append("Exit timing is weak - you may be leaving profits on the table or exiting at poor prices.")

        # MFE capture
        mfe_capture = edge.get('mfe_capture_rate', 0)
        if mfe_capture > 0.7:
            lines.append(f"You're capturing {mfe_capture:.0%} of the maximum favorable excursion - excellent profit taking.")
        elif mfe_capture < 0.4:
            lines.append(f"You're only capturing {mfe_capture:.0%} of available profits - consider adjusting take-profit levels.")

        return " ".join(lines)

    def generate_risk_assessment(self) -> str:
        """
        Generate risk assessment with context.
        """
        lines = []

        max_dd = self.stats.get('Max Drawdown', 0)
        var_95 = self.stats.get('VaR (95%)', 0)

        lines.append(f"Maximum drawdown experienced was {max_dd:.1%}.")

        # Monte Carlo context
        mc_worst = self.stats.get('MC_Min', 0)
        if mc_worst:
            lines.append(f"Monte Carlo simulations suggest a 5% chance of final equity falling to ${mc_worst:,.0f}.")

        risk_of_ruin = self.stats.get('Risk_of_Ruin', 0)
        if risk_of_ruin:
            if risk_of_ruin > 0.1:
                lines.append(f"WARNING: Risk of ruin (losing 50%+) is {risk_of_ruin:.1%}. Consider reducing position sizes.")
            else:
                lines.append(f"Risk of catastrophic loss (50%+) is low at {risk_of_ruin:.1%}.")

        # Statistical risk
        if self.significance:
            dd_p95 = self.significance.get('dd_95_worst_case', 0)
            if dd_p95:
                lines.append(f"Under bootstrap resampling, 95th percentile worst-case drawdown is {dd_p95:.1%}.")

            ci_lower = self.significance.get('sharpe_ci_lower')
            ci_upper = self.significance.get('sharpe_ci_upper')
            if ci_lower is not None:
                lines.append(f"Sharpe ratio confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}].")

        # Calmar context
        calmar = self.stats.get('Calmar Ratio', 0)
        if calmar > 1:
            lines.append(f"The Calmar ratio of {calmar:.2f} indicates returns well compensate for the drawdown risk.")
        elif calmar > 0:
            lines.append(f"The Calmar ratio of {calmar:.2f} is modest.")

        return " ".join(lines)

    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on all analysis.
        """
        recommendations = []

        # From attribution
        if self.attribution:
            insights = self.attribution.get('insights', [])
            recommendations.extend(insights)

            # Optimal stop suggestion
            optimal_stop = self.attribution.get('optimal_stop', {})
            if optimal_stop and 'recommendation' in optimal_stop:
                recommendations.append(optimal_stop['recommendation'])

            # Optimal target suggestion
            optimal_target = self.attribution.get('optimal_target', {})
            if optimal_target and 'recommendation' in optimal_target:
                recommendations.append(optimal_target['recommendation'])

        # From regime analysis
        if self.regime_stats:
            for regime, stats in self.regime_stats.items():
                if isinstance(stats, dict):
                    cum_ret = stats.get('Return', '0%')
                    # Parse string percentage if needed
                    if isinstance(cum_ret, str) and '%' in cum_ret:
                        try:
                            cum_val = float(cum_ret.replace('%', '')) / 100
                            if cum_val < -0.2:
                                recommendations.append(
                                    f"Consider avoiding or reducing exposure during {regime} conditions "
                                    f"(historical return: {cum_ret})."
                                )
                        except Exception:
                            pass

        # From statistical significance
        if self.significance:
            min_track = self.significance.get('min_track_record_days')
            n_obs = self.significance.get('n_observations', 0)
            if min_track and n_obs < min_track:
                recommendations.append(
                    f"Need {min_track - n_obs} more observations to confirm Sharpe ratio is statistically significant."
                )

            if self.significance.get('sharpe_ci_crosses_zero'):
                recommendations.append(
                    "Sharpe ratio confidence interval crosses zero - consider longer test period or larger sample size."
                )

        # From WFO
        if self.wfo and 'wfo_summary' in self.wfo:
            wfo_summary = self.wfo['wfo_summary']

            divergence = wfo_summary.get('divergence', {})
            if divergence.get('degradation_pct', 0) > 40:
                recommendations.append(
                    "High IS/OOS degradation detected. Consider simplifying the strategy or using fewer parameters."
                )

            stability = wfo_summary.get('stability', {})
            if stability.get('overall_stability', 1) < 0.5:
                recommendations.append(
                    "Parameters are unstable across WFO windows. Consider using wider parameter ranges or more robust filters."
                )

        # Default recommendations if none generated
        if not recommendations:
            recommendations.append("Continue monitoring strategy performance and maintain proper position sizing.")

        return recommendations

    def generate_wfo_health_report(self) -> str:
        """
        Generate WFO-specific health report.
        """
        if not self.wfo or 'wfo_summary' not in self.wfo:
            return "No walk-forward optimization data available."

        lines = []
        summary = self.wfo['wfo_summary']

        n_windows = summary.get('n_windows', 0)
        lines.append(f"Walk-forward analysis was performed across {n_windows} rolling windows.")

        # Divergence
        divergence = summary.get('divergence', {})
        if divergence:
            deg_pct = divergence.get('degradation_pct', 0)
            mean_is = divergence.get('mean_is_return', 0)
            mean_oos = divergence.get('mean_oos_return', 0)

            lines.append(
                f"Average in-sample return: {mean_is:.1%}, average out-of-sample return: {mean_oos:.1%} "
                f"({deg_pct:.0f}% degradation)."
            )

            verdict = divergence.get('verdict', '')
            if verdict:
                lines.append(f"Verdict: {verdict}.")

        # Stability
        stability = summary.get('stability', {})
        if stability:
            score = stability.get('overall_stability', 0)
            interp = stability.get('interpretation', '')
            lines.append(f"Parameter stability score: {score:.2f}. {interp}")

        # Consistency
        consistency = summary.get('consistency', {})
        if consistency:
            cons_score = consistency.get('consistency', 0)
            lines.append(f"Out-of-sample consistency: {cons_score:.0%} of windows were profitable.")

        # Overall
        health = summary.get('health_score', 0)
        overall_verdict = summary.get('overall_verdict', '')
        lines.append(f"Overall WFO health score: {health:.0f}/100. Verdict: {overall_verdict}.")

        return " ".join(lines)

    def generate_ai_decision_summary(self) -> Dict[str, Any]:
        """
        Generate AI decision agent summary.
        """
        if not self.ai_decision:
            return {'available': False}

        decision = self.ai_decision.get('decision', 'UNKNOWN')
        confidence = self.ai_decision.get('confidence', 0)
        reasoning = self.ai_decision.get('reasoning', '')

        # Determine verdict class
        if decision == 'DEPLOY':
            verdict_class = 'good'
            verdict_text = 'Ready for Live Trading'
        elif decision == 'ITERATE':
            verdict_class = 'warning'
            verdict_text = 'Needs Improvement'
        else:
            verdict_class = 'bad'
            verdict_text = 'Not Viable'

        return {
            'available': True,
            'decision': decision,
            'confidence': confidence,
            'confidence_pct': f"{confidence:.0%}",
            'reasoning': reasoning,
            'verdict_class': verdict_class,
            'verdict_text': verdict_text,
            'weaknesses': self.ai_decision.get('weaknesses', []),
            'suggestions': self.ai_decision.get('suggestions', []),
            'priority_focus': self.ai_decision.get('priority_focus', ''),
            'position_sizing': self.ai_decision.get('position_sizing', {}),
            'risk_limits': self.ai_decision.get('risk_limits', {}),
            'fatal_flaws': self.ai_decision.get('fatal_flaws', []),
            'model_used': self.ai_decision.get('model_used', 'N/A'),
        }

    def generate_iteration_summary(self) -> Dict[str, Any]:
        """
        Generate iteration history summary.
        """
        if not self.iteration_history:
            return {'available': False}

        iterations = self.iteration_history.get('iterations', [])
        return {
            'available': True,
            'total_iterations': len(iterations),
            'final_decision': self.iteration_history.get('final_decision', 'N/A'),
            'total_improvement': self.iteration_history.get('total_improvement_pct', 0),
            'iterations': iterations,
            'started_at': self.iteration_history.get('started_at', ''),
            'completed_at': self.iteration_history.get('completed_at', ''),
        }

    def generate_code_review_summary(self) -> Dict[str, Any]:
        """
        Generate code review summary.
        """
        if not self.code_review:
            return {'available': False}

        issues = self.code_review.get('issues', [])
        critical = sum(1 for i in issues if i.get('severity') == 'CRITICAL')
        warnings = sum(1 for i in issues if i.get('severity') == 'WARNING')

        return {
            'available': True,
            'passed': self.code_review.get('passed', True),
            'total_issues': len(issues),
            'critical_count': critical,
            'warning_count': warnings,
            'issues': issues[:10],  # Limit to 10 for display
            'review_time_ms': self.code_review.get('review_time_ms', 0),
        }

    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate complete insights report.
        """
        return {
            'executive_summary': self.generate_executive_summary(),
            'edge_explanation': self.generate_edge_explanation(),
            'risk_assessment': self.generate_risk_assessment(),
            'recommendations': self.generate_recommendations(),
            'wfo_health': self.generate_wfo_health_report(),
            'ai_decision': self.generate_ai_decision_summary(),
            'iteration_history': self.generate_iteration_summary(),
            'code_review': self.generate_code_review_summary(),
            'generated_at': datetime.now().isoformat()
        }


class InsightsDashboard:
    """
    Generate formatted output in multiple formats.
    """

    def __init__(self, insights_engine: InsightsEngine, stats: Dict = None):
        self.engine = insights_engine
        self.stats = stats or insights_engine.stats
        self.report = self.engine.generate_full_report()

    def print_console_summary(self):
        """
        Print concise summary to console.
        """
        print("\n" + "=" * 60)
        print("STRATEGY INSIGHTS SUMMARY")
        print("=" * 60)

        # Executive Summary
        print("\n>>> EXECUTIVE SUMMARY")
        print("-" * 40)
        # Word wrap for console
        summary = self.report['executive_summary']
        words = summary.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            print(line)

        # Edge Explanation
        print("\n>>> WHERE YOUR EDGE COMES FROM")
        print("-" * 40)
        edge = self.report['edge_explanation']
        words = edge.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            print(line)

        # Recommendations
        print("\n>>> RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(self.report['recommendations'][:5], 1):
            # Truncate long recommendations
            if len(rec) > 100:
                rec = rec[:97] + "..."
            print(f"  {i}. {rec}")

        # Key Metrics
        print("\n>>> KEY METRICS")
        print("-" * 40)
        metrics = [
            ('Total Return', self.stats.get('Total Return', 0), '.1%'),
            ('Sharpe Ratio', self.stats.get('Sharpe Ratio', 0), '.2f'),
            ('Max Drawdown', self.stats.get('Max Drawdown', 0), '.1%'),
            ('Calmar Ratio', self.stats.get('Calmar Ratio', 0), '.2f'),
        ]
        for name, value, fmt in metrics:
            try:
                formatted = f"{value:{fmt}}"
            except Exception:
                formatted = str(value)
            print(f"  {name}: {formatted}")

        print("\n" + "=" * 60 + "\n")

    def generate_markdown(self) -> str:
        """
        Generate markdown report.
        """
        md = []

        # Header
        md.append("# Strategy Insights Report")
        md.append(f"*Generated: {self.report['generated_at']}*\n")

        # Executive Summary
        md.append("## Executive Summary")
        md.append(self.report['executive_summary'])
        md.append("")

        # Key Metrics Table
        md.append("## Key Metrics")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")

        metrics = [
            ('Total Return', self.stats.get('Total Return', 0), '.1%'),
            ('CAGR', self.stats.get('CAGR', 0), '.1%'),
            ('Sharpe Ratio', self.stats.get('Sharpe Ratio', 0), '.2f'),
            ('Sortino Ratio', self.stats.get('Sortino Ratio', 0), '.2f'),
            ('Max Drawdown', self.stats.get('Max Drawdown', 0), '.1%'),
            ('Calmar Ratio', self.stats.get('Calmar Ratio', 0), '.2f'),
            ('VaR (95%)', self.stats.get('VaR (95%)', 0), '.2%'),
            ('Ending Equity', self.stats.get('Ending Equity', 0), ',.0f'),
        ]

        for name, value, fmt in metrics:
            try:
                if fmt == ',.0f':
                    formatted = f"${value:{fmt}}"
                else:
                    formatted = f"{value:{fmt}}"
            except Exception:
                formatted = str(value)
            md.append(f"| {name} | {formatted} |")

        md.append("")

        # Edge Explanation
        md.append("## Edge Attribution")
        md.append(self.report['edge_explanation'])
        md.append("")

        # Risk Assessment
        md.append("## Risk Assessment")
        md.append(self.report['risk_assessment'])
        md.append("")

        # WFO Health
        md.append("## Walk-Forward Analysis")
        md.append(self.report['wfo_health'])
        md.append("")

        # Recommendations
        md.append("## Recommendations")
        for i, rec in enumerate(self.report['recommendations'], 1):
            md.append(f"{i}. {rec}")
        md.append("")

        return "\n".join(md)

    def generate_html(self) -> str:
        """
        Generate interactive HTML dashboard.
        """
        # Prepare data for charts
        recommendations_html = "\n".join([
            f'<li class="recommendation">{rec}</li>'
            for rec in self.report['recommendations']
        ])

        # Metrics for display
        total_return = self.stats.get('Total Return', 0)
        sharpe = self.stats.get('Sharpe Ratio', 0)
        max_dd = self.stats.get('Max Drawdown', 0)
        calmar = self.stats.get('Calmar Ratio', 0)

        # Color coding
        def get_color(value, thresholds):
            """Get color based on value and thresholds."""
            if value >= thresholds[0]:
                return '#27ae60'  # Green
            elif value >= thresholds[1]:
                return '#f39c12'  # Orange
            else:
                return '#e74c3c'  # Red

        return_color = get_color(total_return, (0.2, 0))
        sharpe_color = get_color(sharpe, (1.0, 0.5))
        dd_color = get_color(-max_dd, (0.1, 0.2))
        calmar_color = get_color(calmar, (1.0, 0.5))

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Insights Dashboard</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .card-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 15px;
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #aaa;
            font-size: 0.9rem;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .section-title {{
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #3a7bd5;
            border-bottom: 2px solid #3a7bd5;
            padding-bottom: 10px;
        }}
        .insight-text {{
            line-height: 1.8;
            color: #ddd;
        }}
        .recommendation {{
            padding: 10px 15px;
            margin: 10px 0;
            background: rgba(58, 123, 213, 0.1);
            border-left: 3px solid #3a7bd5;
            border-radius: 0 8px 8px 0;
        }}
        .recommendations-list {{
            list-style: none;
            padding: 0;
        }}
        .verdict {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .verdict-good {{ background: rgba(39, 174, 96, 0.3); color: #27ae60; }}
        .verdict-warning {{ background: rgba(243, 156, 18, 0.3); color: #f39c12; }}
        .verdict-bad {{ background: rgba(231, 76, 60, 0.3); color: #e74c3c; }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Strategy Insights Dashboard</h1>

        <!-- Key Metrics -->
        <div class="dashboard-grid">
            <div class="card">
                <div class="card-title">Total Return</div>
                <div class="metric-value" style="color: {return_color}">{total_return:.1%}</div>
                <div class="metric-label">Cumulative Performance</div>
            </div>
            <div class="card">
                <div class="card-title">Sharpe Ratio</div>
                <div class="metric-value" style="color: {sharpe_color}">{sharpe:.2f}</div>
                <div class="metric-label">Risk-Adjusted Return</div>
            </div>
            <div class="card">
                <div class="card-title">Max Drawdown</div>
                <div class="metric-value" style="color: {dd_color}">{max_dd:.1%}</div>
                <div class="metric-label">Worst Peak-to-Trough</div>
            </div>
            <div class="card">
                <div class="card-title">Calmar Ratio</div>
                <div class="metric-value" style="color: {calmar_color}">{calmar:.2f}</div>
                <div class="metric-label">Return / Max DD</div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="card full-width">
            <div class="section-title">Executive Summary</div>
            <p class="insight-text">{self.report['executive_summary']}</p>
        </div>

        <div class="dashboard-grid">
            <!-- Edge Explanation -->
            <div class="card">
                <div class="section-title">Edge Attribution</div>
                <p class="insight-text">{self.report['edge_explanation']}</p>
            </div>

            <!-- Risk Assessment -->
            <div class="card">
                <div class="section-title">Risk Assessment</div>
                <p class="insight-text">{self.report['risk_assessment']}</p>
            </div>
        </div>

        <!-- WFO Health -->
        <div class="card full-width">
            <div class="section-title">Walk-Forward Analysis</div>
            <p class="insight-text">{self.report['wfo_health']}</p>
        </div>

        <!-- Recommendations -->
        <div class="card full-width">
            <div class="section-title">Recommendations</div>
            <ul class="recommendations-list">
                {recommendations_html}
            </ul>
        </div>

        {self._generate_ai_decision_html()}

        {self._generate_iteration_history_html()}

        {self._generate_code_review_html()}

        <div class="timestamp">
            Generated: {self.report['generated_at']}
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_ai_decision_html(self) -> str:
        """Generate HTML for AI Decision panel."""
        ai = self.report.get('ai_decision', {})
        if not ai.get('available'):
            return ""

        decision = ai.get('decision', 'UNKNOWN')
        confidence = ai.get('confidence_pct', '0%')
        reasoning = ai.get('reasoning', '')
        verdict_class = ai.get('verdict_class', 'warning')
        verdict_text = ai.get('verdict_text', '')

        # Decision badge colors
        badge_colors = {
            'DEPLOY': '#27ae60',
            'ITERATE': '#f39c12',
            'ABANDON': '#e74c3c',
        }
        badge_color = badge_colors.get(decision, '#888')

        # Build suggestions list
        suggestions_html = ""
        suggestions = ai.get('suggestions', [])
        if suggestions:
            suggestions_html = "<div class='section-subtitle'>Suggestions:</div><ul class='recommendations-list'>"
            for s in suggestions[:5]:
                suggestions_html += f'<li class="recommendation">{s}</li>'
            suggestions_html += "</ul>"

        # Build weaknesses list
        weaknesses_html = ""
        weaknesses = ai.get('weaknesses', [])
        if weaknesses:
            weaknesses_html = "<div class='section-subtitle'>Weaknesses Identified:</div><ul style='color: #f39c12;'>"
            for w in weaknesses[:5]:
                weaknesses_html += f'<li>{w}</li>'
            weaknesses_html += "</ul>"

        # Position sizing for DEPLOY
        position_html = ""
        if decision == 'DEPLOY' and ai.get('position_sizing'):
            pos = ai['position_sizing']
            position_html = f"""
            <div class='section-subtitle'>Recommended Position Sizing:</div>
            <p style='color: #aaa;'>
                Max Risk Per Trade: {pos.get('max_risk_pct', 2)}% |
                Kelly Fraction: {pos.get('kelly_fraction', 0.25):.0%}
            </p>
            """

        return f"""
        <!-- AI Decision Panel -->
        <div class="card full-width" style="border: 2px solid {badge_color};">
            <div class="section-title" style="color: {badge_color};">
                AI Decision Agent
                <span class="verdict verdict-{verdict_class}" style="float: right; margin-top: -5px;">
                    {decision} - {confidence} Confidence
                </span>
            </div>
            <p class="insight-text"><strong>Verdict:</strong> {verdict_text}</p>
            <p class="insight-text"><strong>Reasoning:</strong> {reasoning}</p>
            {weaknesses_html}
            {suggestions_html}
            {position_html}
            <p style="color: #666; font-size: 0.8rem; margin-top: 15px;">Model: {ai.get('model_used', 'N/A')}</p>
        </div>
        """

    def _generate_iteration_history_html(self) -> str:
        """Generate HTML for iteration history."""
        hist = self.report.get('iteration_history', {})
        if not hist.get('available'):
            return ""

        iterations = hist.get('iterations', [])
        if not iterations:
            return ""

        total_improvement = hist.get('total_improvement', 0)
        improvement_color = '#27ae60' if total_improvement > 0 else '#e74c3c'

        # Build iteration timeline
        timeline_html = ""
        for it in iterations:
            decision = it.get('decision', 'ITERATE')
            improvement = it.get('improvement_pct', 0)
            imp_str = f"+{improvement:.1%}" if improvement > 0 else f"{improvement:.1%}"

            decision_colors = {'DEPLOY': '#27ae60', 'ITERATE': '#f39c12', 'ABANDON': '#e74c3c'}
            dec_color = decision_colors.get(decision, '#888')

            timeline_html += f"""
            <div style="display: flex; align-items: center; margin: 10px 0; padding: 10px;
                        background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: {dec_color};
                            display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    {it.get('number', '?')}
                </div>
                <div style="flex: 1;">
                    <div style="font-weight: bold;">{decision}</div>
                    <div style="color: #aaa; font-size: 0.9rem;">
                        Sharpe change: {imp_str} | Modifications: {it.get('modifications', 0)}
                    </div>
                </div>
            </div>
            """

        return f"""
        <!-- Iteration History -->
        <div class="card full-width">
            <div class="section-title">
                Autonomous Iteration History
                <span style="float: right; color: {improvement_color};">
                    Total Improvement: {total_improvement:+.1%}
                </span>
            </div>
            <p class="insight-text">
                Completed {hist.get('total_iterations', 0)} iteration(s).
                Final Decision: <strong>{hist.get('final_decision', 'N/A')}</strong>
            </p>
            <div style="margin-top: 15px;">
                {timeline_html}
            </div>
        </div>
        """

    def _generate_code_review_html(self) -> str:
        """Generate HTML for code review results."""
        review = self.report.get('code_review', {})
        if not review.get('available'):
            return ""

        passed = review.get('passed', True)
        status_color = '#27ae60' if passed else '#e74c3c'
        status_text = 'PASSED' if passed else 'FAILED'

        critical = review.get('critical_count', 0)
        warnings = review.get('warning_count', 0)

        # Build issues list
        issues_html = ""
        issues = review.get('issues', [])
        if issues:
            issues_html = "<div style='margin-top: 15px;'>"
            for issue in issues[:5]:
                severity = issue.get('severity', 'INFO')
                sev_colors = {'CRITICAL': '#e74c3c', 'WARNING': '#f39c12', 'INFO': '#3498db'}
                sev_color = sev_colors.get(severity, '#888')

                issues_html += f"""
                <div style="padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.05);
                            border-left: 3px solid {sev_color}; border-radius: 0 8px 8px 0;">
                    <span style="color: {sev_color}; font-weight: bold;">[{severity}]</span>
                    {issue.get('category', 'Unknown')}: {issue.get('message', '')}
                </div>
                """
            issues_html += "</div>"

        return f"""
        <!-- Code Review Results -->
        <div class="card full-width">
            <div class="section-title">
                Pre-Optimization Code Review
                <span class="verdict" style="float: right; background: rgba({status_color.replace('#', '')}, 0.3);
                                            color: {status_color};">
                    {status_text}
                </span>
            </div>
            <div class="dashboard-grid" style="grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: #e74c3c;">{critical}</div>
                    <div style="color: #aaa;">Critical Issues</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: #f39c12;">{warnings}</div>
                    <div style="color: #aaa;">Warnings</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: #888;">{review.get('review_time_ms', 0):.0f}ms</div>
                    <div style="color: #aaa;">Review Time</div>
                </div>
            </div>
            {issues_html}
        </div>
        """

    def save_all_formats(self, base_path: str = "strategy_insights"):
        """
        Save report in all formats.

        Args:
            base_path: Base filename (without extension)
        """
        import os

        # HTML
        html_path = f"{base_path}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_html())
        print(f"HTML report saved to: {html_path}")

        # Markdown
        md_path = f"{base_path}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_markdown())
        print(f"Markdown report saved to: {md_path}")

        # Console
        self.print_console_summary()

        return {
            'html': html_path,
            'markdown': md_path
        }


def generate_insights(
    stats: Dict,
    attribution: Dict = None,
    wfo_analytics: Dict = None,
    stat_significance: Dict = None,
    regime_stats: Dict = None,
    ai_decision: Dict = None,
    iteration_history: Dict = None,
    code_review: Dict = None,
    output_path: str = None,
    print_console: bool = True
) -> Dict:
    """
    Main entry point for generating strategy insights.

    Args:
        stats: Performance statistics
        attribution: Trade attribution results
        wfo_analytics: WFO analysis results
        stat_significance: Statistical significance analysis
        regime_stats: Regime-based performance breakdown
        ai_decision: AI Decision Agent results
        iteration_history: Iteration loop history
        code_review: Code review results
        output_path: Base path for saving reports (optional)
        print_console: Whether to print console summary

    Returns:
        Dictionary with all generated insights
    """
    # Create engine
    engine = InsightsEngine(
        stats=stats,
        attribution=attribution,
        wfo_analytics=wfo_analytics,
        stat_significance=stat_significance,
        regime_stats=regime_stats,
        ai_decision=ai_decision,
        iteration_history=iteration_history,
        code_review=code_review
    )

    # Create dashboard
    dashboard = InsightsDashboard(engine, stats)

    # Generate report
    report = engine.generate_full_report()

    # Output
    if print_console:
        dashboard.print_console_summary()

    if output_path:
        paths = dashboard.save_all_formats(output_path)
        report['output_files'] = paths

    return report
