import pandas as pd
from typing import Dict, Any, List

class AIOptimizationPrompter:
    """
    Generates AI-ready prompts to help optimize trading strategies.
    Analyzes backtest results and identifies weaknesses.
    """
    def __init__(self, stats: Dict[str, Any], top_losers: List[Dict[str, Any]] = None):
        self.stats = stats
        self.top_losers = top_losers if top_losers else []

    def generate_prompt(self, strategy_name: str, symbol: str) -> str:
        """
        Creates a prompt for an LLM to debug the strategy.
        """
        # Using a list and join to avoid escaping hell
        lines = [
            f"# Strategy Optimization Request",
            f"I need help optimizing my trading strategy '{strategy_name}' on {symbol}.",
            f"Current Backtest Performance:",
            f"- Sharpe Ratio: {self.stats.get('Sharpe Ratio', 0):.2f}",
            f"- Total Return: {self.stats.get('Total Return', 0):.2%}",
            f"- Max Drawdown: {self.stats.get('Max Drawdown', 0):.2%}",
            f"- Profit Factor: {self.stats.get('Profit Factor', 0):.2f}",
            "",
            "## Performance Issues:"
        ]
        
        # Identify specific issues
        if self.stats.get('Sharpe Ratio', 0) < 0.5:
            lines.append("- The risk-adjusted return (Sharpe) is critically low.")
        if self.stats.get('Max Drawdown', 0) < -0.15:
            lines.append("- The drawdown is too deep, indicating a lack of effective stop-losses or poor market regime alignment.")
        if self.stats.get('Profit Factor', 0) < 1.1:
            lines.append("- The strategy has low expectancy (Profit Factor near 1.0).")

        if self.top_losers:
            lines.append("\n## Top 5 Losing Trade Contexts:")
            for i, loser in enumerate(self.top_losers):
                lines.append(f"{i+1}. Time: {loser['datetime']}, PnL: ${loser['realized_pnl']:.2f}, Regime: {loser.get('regime', 'UNKNOWN')}")

        lines.extend([
            "",
            "## Task:",
            "Please suggest filtering criteria or indicator modifications to improve this strategy.",
            "Focus specifically on how to avoid the losing trades identified above in their respective market regimes.",
            "Should I add a Volatility filter, change the Lookback period, or implement a dynamic Exit?"
        ])
        
        return "\n".join(lines)

    def save_prompt(self, strategy_name: str, symbol: str, filepath: str = "ai_optimization_prompt.txt"):
        prompt = self.generate_prompt(strategy_name, symbol)
        with open(filepath, "w") as f:
            f.write(prompt)
        print(f"[AI FEEDBACK] Optimization prompt saved to: {filepath}")
        return filepath
