
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

class DashboardGenerator:
    """
    Generates a rich, interactive HTML Dashboard for strategy results.
    """
    def __init__(self, 
                 strategy_name: str, 
                 params: Dict, 
                 stats: Dict, 
                 equity_curve: pd.DataFrame,
                 skeptic_results: Dict = None):
        self.strategy_name = strategy_name
        self.params = params
        self.stats = stats
        self.equity_curve = equity_curve # Should have date index and 'equity' col
        self.skeptic_results = skeptic_results

    def generate(self, filename: str = "report.html"):
        print(f"    📊 Generating Interactive Dashboard: {filename}...")
        
        # Create Subplots
        # Row 1: Equity Curve
        # Row 2: Drawdown
        # Row 3: Skeptic Distribution (if avail) or Monthly Returns
        
        rows = 3
        fig = make_subplots(
            rows=rows, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Equity Curve', 'Drawdown Underwater', 'Statistical Denial (Permutation Test)')
        )
        
        # 1. Equity Curve
        if not self.equity_curve.empty:
            fig.add_trace(
                go.Scatter(
                    x=self.equity_curve.index, 
                    y=self.equity_curve['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='#00ff00', width=2)
                ),
                row=1, col=1
            )
            
            # 2. Drawdown
            # Assume 'drawdown' col exists or calc it
            if 'drawdown' not in self.equity_curve.columns:
                 peak = self.equity_curve['equity'].cummax()
                 safe_peak = peak.replace(0, np.nan)
                 dd = ((self.equity_curve['equity'] - peak) / safe_peak).fillna(0)
                 self.equity_curve['drawdown'] = dd
            
            fig.add_trace(
                go.Scatter(
                    x=self.equity_curve.index,
                    y=self.equity_curve['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='#ff0000', width=1)
                ),
                row=2, col=1
            )
            
        # 3. Skeptic Results (if available) -> Histogram
        if self.skeptic_results and 'p_value' in self.skeptic_results:
             # We assume we might have the list of returns? 
             # Skeptic class currently only returns p-value.
             # Visualization of the p-value: Bar chart?
             # For now, just Text Annotation or simple indicator
             pass

        # Styling
        fig.update_layout(
            title=f"Strategy Report: {self.strategy_name}",
            template="plotly_dark",
            height=1000,
            showlegend=True
        )
        
        # Add Text Stats Annotation
        stats_text = f"Total Return: {self.stats.get('Total Return', 0):.2%}<br>"
        stats_text += f"Sharpe: {self.stats.get('Sharpe', 0):.2f}<br>"
        if self.skeptic_results:
             stats_text += f"Skeptic P-Value: {self.skeptic_results.get('p_value', 1.0):.3f} ({self.skeptic_results.get('verdict')})"
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            showarrow=False,
            bgcolor="#333",
            bordercolor="#fff",
            font=dict(color="#fff")
        )

        # Save
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.write_html(path)
        print(f"      ✅ Dashboard Saved: {path}")
