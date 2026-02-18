
import os
import pandas as pd
from datetime import datetime

class ReportGenerator:
    """
    Generates a professional Markdown Report for the backtest run.
    """
    def __init__(self, run_name: str, output_dir: str = "reports"):
        self.run_name = run_name
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.content = []
        
        os.makedirs(output_dir, exist_ok=True)
        
    def add_header(self, strategy_name: str, params: dict):
        self.content.append(f"# Backtest Report: {strategy_name}")
        self.content.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.content.append(f"**Run ID:** {self.run_name}")
        self.content.append(f"**Parameters:** `{params}`")
        self.content.append("---")
        
    def add_performance_summary(self, vector_ret, event_ret, equity):
        diff = vector_ret - event_ret
        status = "✅ PASS" if abs(diff) < 0.05 else "❌ DISCREPANCY"
        
        self.content.append("## 1. Executive Summary")
        self.content.append("| Metric | Value | Reference |")
        self.content.append("| :--- | :--- | :--- |")
        self.content.append(f"| **Final Equity** | `${equity:,.2f}` | *Simulation* |")
        self.content.append(f"| **Event Return** | `{event_ret:.2%}` | *Reality* |")
        self.content.append(f"| **Vector Return** | `{vector_ret:.2%}` | *Theoretical* |")
        self.content.append(f"| **Robustness** | `{abs(diff):.2%}` | *{status}* |")
        self.content.append("")
        
    
    def add_forensics(self, mc_stats: dict, stability_score: float = None):
        self.content.append("## 2. Forensic Analysis")
        
        if mc_stats:
            self.content.append("### Monte Carlo Stress Test (1,000 Sims)")
            self.content.append(f"- **VaR (95%)**: `{mc_stats.get('worst_case_95', 0):.2%}` (Worst Case Drawdown)")
            self.content.append(f"- **Avg Drawdown**: `{mc_stats.get('avg_dd', 0):.2%}`")
            self.content.append("")
        
        if stability_score is not None:
            self.content.append("### Parameter Stability")
            stab_icon = "🟢" if stability_score > 0.8 else "🔴"
            self.content.append(f"- **Score**: `{stability_score:.2f}` {stab_icon}")
            self.content.append(f"- *Target > 0.8 indicates the strategy is not overfit to specific numbers.*")
            self.content.append("")

    def add_regime_analysis(self, regime_stats: dict):
        if not regime_stats: return
        self.content.append("### Regime Performance")
        self.content.append("| Regime | Return | Sharpe | Days |")
        self.content.append("| :--- | :--- | :--- | :--- |")
        
        for name, stats in regime_stats.items():
            ret = stats.get('Return', '0%')
            sharpe = stats.get('Sharpe', '0.0')
            days = stats.get('Days', 0)
            self.content.append(f"| {name} | {ret} | {sharpe} | {days} |")
        self.content.append("")

    def add_strategy_advice(self, advice_list: list):
        if not advice_list: return
        self.content.append("## 3. 🤖 AI Strategy Advisor")
        for item in advice_list:
            self.content.append(f"- {item}")
        self.content.append("")

    def add_pine_script(self, pine_path: str):
        if not pine_path: return
        self.content.append("## 4. Visual Parity")
        self.content.append(f"Generated Pine Script for TradingView: `{pine_path}`")
        self.content.append("```pine")
        try:
            with open(pine_path, 'r') as f:
                self.content.append(f.read())
        except Exception:
            self.content.append("// Error reading Pine Script file")
        self.content.append("```")

    def save(self):
        filename = f"{self.run_name}_{self.timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("\n".join(self.content))
            
        print(f"\n📄 Report Generated: {filepath}")
        return filepath
