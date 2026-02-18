import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import io
import base64
import webbrowser
import os
from .regime import MarketRegimeDetector
from .statistics import StatisticalSignificance, BootstrapInference, analyze_strategy_significance

def create_sharpe_ratio(returns, periods=252):
    std = np.std(returns)
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(periods) * (np.mean(returns)) / std

def create_sortino_ratio(returns, periods=252):
    neg_returns = returns[returns < 0]
    if len(neg_returns) < 1:
        return 0.0
    neg_std = np.std(neg_returns)
    if neg_std == 0 or np.isnan(neg_std):
        return 0.0
    return np.sqrt(periods) * (np.mean(returns)) / neg_std

def create_drawdowns(equity_curve: pd.DataFrame):
    hwm = equity_curve['equity'].cummax()
    # Guard against zero hwm (degenerate case)
    safe_hwm = hwm.replace(0, np.nan)
    drawdown = ((equity_curve['equity'] - hwm) / safe_hwm).fillna(0)
    max_drawdown = drawdown.min()
    max_duration = (drawdown < 0).astype(int).groupby(drawdown.eq(0).cumsum()).cumsum().max()
    return drawdown, max_drawdown, max_duration

def calculate_var(returns, confidence_level=0.95):
    """Calculate Value at Risk (Historical Simulation)."""
    if len(returns) < 5:
        return 0.0
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_tail_ratio(returns):
    """95th percentile / 5th percentile return."""
    p95 = np.percentile(returns, 95)
    p5 = np.abs(np.percentile(returns, 5))
    return p95 / p5 if p5 != 0 else 0.0

class TearSheet:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.equity_curve = pd.DataFrame(self.portfolio.equity_curve)
        
    def analyze(self, benchmark_series=None):
        if self.equity_curve.empty:
            print("No trades or history to analyze.")
            return {}

        self.equity_curve['datetime'] = pd.to_datetime(self.equity_curve['datetime'])
        self.equity_curve.set_index('datetime', inplace=True)
        
        self.equity_curve['returns'] = self.equity_curve['equity'].pct_change()
        
        # Metrics
        total_return = (self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0]) - 1.0
        returns = self.equity_curve['returns'].dropna()
        
        if len(returns) > 1:
            sharpe = create_sharpe_ratio(returns)
            sortino = create_sortino_ratio(returns)
            volatility = returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
            sortino = 0.0
            volatility = 0.0
            
        drawdown_series, max_dd, max_dd_duration = create_drawdowns(self.equity_curve)
        self.equity_curve['drawdown'] = drawdown_series
        
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        if days > 0:
            final_equity = self.equity_curve['equity'].iloc[-1]
            initial_equity = self.equity_curve['equity'].iloc[0]
            if final_equity < 0:
                cagr = -1.0 # Total loss
            else:
                cagr = (final_equity / initial_equity) ** (1/years) - 1.0
        else:
            cagr = 0.0

        stats = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Annual Volatility': volatility,
            'Ending Equity': self.equity_curve['equity'].iloc[-1],
            'Calmar Ratio': cagr / abs(max_dd) if max_dd != 0 else 0.0,
            'VaR (95%)': calculate_var(returns, 0.95),
            'VaR (99%)': calculate_var(returns, 0.99),
            'Tail Ratio': calculate_tail_ratio(returns)
        }

        # Statistical Significance Analysis
        if len(returns) > 5:
            try:
                sig_analysis = analyze_strategy_significance(returns.values)
                stats['Sharpe SE'] = sig_analysis.get('sharpe_se', 0)
                stats['Sharpe CI Lower'] = sig_analysis.get('sharpe_ci_lower', 0)
                stats['Sharpe CI Upper'] = sig_analysis.get('sharpe_ci_upper', 0)
                stats['Sharpe P-Value'] = sig_analysis.get('sharpe_pvalue', 1.0)
                stats['Sharpe Significant'] = sig_analysis.get('sharpe_significant', False)
                stats['Min Track Record Days'] = sig_analysis.get('min_track_record_days', 0)
                stats['_significance_analysis'] = sig_analysis  # Store full analysis for insights
            except Exception as e:
                # Graceful fallback if statistics module has issues
                pass
        
        # Trade level metrics
        if self.portfolio.trade_log:
            trades = pd.DataFrame(self.portfolio.trade_log)
            # We assume 'pnl' or similar is not there, we calculate it from equity if needed
            # But better to use trade log if it has returns.
            # For now, keep it simple as the user might not have pnl in trades yet.
            # Let's check trade log structure from viewed_file
            # trade log has price, qty, etc. 
            # We can calculate Profit Factor if we had exit price.
            # Actually, the user asked for enhanced metrics. 
            # Let's see if we can derive Profit Factor from returns.
            pos_rets = returns[returns > 0]
            neg_rets = returns[returns < 0]
            profit_factor = pos_rets.sum() / abs(neg_rets.sum()) if len(neg_rets) > 0 and neg_rets.sum() != 0 else 0.0
            stats['Profit Factor'] = profit_factor
            
            # --- Trade Breakdown (Top Losers) ---
            trade_df = pd.DataFrame(self.portfolio.trade_log)
            if 'realized_pnl' in trade_df.columns:
                exits = trade_df[trade_df['realized_pnl'] != 0].copy()
                if not exits.empty:
                    exits = exits.sort_values(by='realized_pnl', ascending=True) # Losers first
                    top_losers = exits.head(5).to_dict('records')
                    
                    # Add Regime Info
                    detector = MarketRegimeDetector()
                    for loser in top_losers:
                        symbol = loser['symbol']
                        df = self.portfolio.bars.symbol_data[symbol]
                        # Slice data up to trade time to detect regime at that moment
                        trade_time = pd.to_datetime(loser['datetime'])
                        history = df[df.index <= trade_time]
                        if not history.empty:
                            loser['regime'] = detector.detect(history).value
                        else:
                            loser['regime'] = "UNKNOWN"
                    
                    stats['Top_Losers'] = top_losers
        
        self._print_stats(stats)
        self.create_html_report(self.equity_curve, stats, benchmark_series)
        return stats

    def _print_stats(self, stats):
        print("-" * 50)
        print("PERFORMANCE TEAR SHEET")
        print("-" * 50)
        print(f"Total Return:      {stats['Total Return']:.2%}")
        print(f"CAGR:              {stats['CAGR']:.2%}")
        print(f"Sharpe Ratio:      {stats['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio:     {stats['Sortino Ratio']:.2f}")
        print(f"Calmar Ratio:      {stats.get('Calmar Ratio', 0):.2f}")
        print(f"Profit Factor:     {stats.get('Profit Factor', 0):.2f}")
        print(f"Max Drawdown:      {stats['Max Drawdown']:.2%}")
        print(f"Annual Volatility: {stats['Annual Volatility']:.2%}")
        print(f"VaR (95%):         {stats.get('VaR (95%)', 0):.2%}")
        print(f"Tail Ratio:        {stats.get('Tail Ratio', 0):.2f}")
        print("-" * 50)
        print(f"Ending Equity:     ${stats['Ending Equity']:,.2f}")
        print("-" * 50)

        # Statistical Significance
        if 'Sharpe P-Value' in stats:
            print("STATISTICAL SIGNIFICANCE")
            print("-" * 50)
            ci_lower = stats.get('Sharpe CI Lower', 0)
            ci_upper = stats.get('Sharpe CI Upper', 0)
            p_val = stats.get('Sharpe P-Value', 1.0)
            sig = "YES" if stats.get('Sharpe Significant', False) else "NO"
            print(f"Sharpe 95% CI:     [{ci_lower:.2f}, {ci_upper:.2f}]")
            print(f"P-Value:           {p_val:.4f}")
            print(f"Significant:       {sig}")
            if ci_lower < 0 < ci_upper:
                print("  [NOTE] CI crosses zero - results may be noise")
            print("-" * 50)

    def run_monte_carlo(self, n_sims=1000):
        """
        Runs Monte Carlo simulation on the trade list.
        Returns simulation stats and plotting data.
        """
        if not self.portfolio.trade_log:
            return {}, None

        # Attempt GPU acceleration
        from .accelerate import GPU_AVAILABLE, gpu_monte_carlo
        if GPU_AVAILABLE:
            returns = self.equity_curve['returns'].fillna(0)
            if len(returns) >= 10:
                initial_equity = self.equity_curve['equity'].iloc[0]
                mc_stats = gpu_monte_carlo(returns, n_sims=n_sims, initial_equity=initial_equity)
                # We still need curves for plotting (keep limited)
                sim_curves = []
                for _ in range(min(50, n_sims)):
                    sim_rets = returns.sample(n=len(returns), replace=True).values
                    sim_curves.append(initial_equity * (1 + sim_rets).cumprod())
                
                # Align mc_stats keys
                mc_stats['MC_Max'] = mc_stats['p95']
                mc_stats['MC_Median'] = mc_stats['p50']
                mc_stats['MC_Min'] = mc_stats['p5']
                mc_stats['Risk_of_Ruin'] = np.mean(np.array(mc_stats['final_values']) < initial_equity * 0.5)
                return mc_stats, sim_curves

        # Fallback to CPU logic
        returns = self.equity_curve['returns'].fillna(0)
        if len(returns) < 10:
            return {}, None
            
        final_values = []
        sim_curves = []
        
        initial_equity = self.equity_curve['equity'].iloc[0]
        
        for _ in range(n_sims):
            sim_rets = returns.sample(n=len(returns), replace=True).values
            sim_curve = initial_equity * (1 + sim_rets).cumprod()
            final_values.append(sim_curve[-1])
            if len(sim_curves) < 50:
                sim_curves.append(sim_curve)
                
        final_values = np.array(final_values)
        
        mc_stats = {
            'MC_Max': np.percentile(final_values, 95),
            'MC_Median': np.median(final_values),
            'MC_Min': np.percentile(final_values, 5),
            'Risk_of_Ruin': np.mean(final_values < initial_equity * 0.5)
        }
        
        return mc_stats, sim_curves

    def create_html_report(self, equity_df, stats, benchmark_series=None):
        """Generates a standalone HTML report with embedded charts."""
        
        # Run Monte Carlo
        mc_stats, sim_curves = self.run_monte_carlo()
        if mc_stats:
            stats.update(mc_stats)
        
        # 1. Generate Plots
        img_str = self._generate_plots(equity_df, sim_curves, benchmark_series)
        
        # Filter keys for the summary table (exclude lists/dicts)
        table_keys = [k for k, v in stats.items() if not isinstance(v, (list, dict))]
        
        html = f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f4f4f9; color: #333; }}
                h1 {{ color: #2c3e50; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 100%; }}
                .half {{ width: 48%; min-width: 300px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border-bottom: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; color: #495057; }}
                tr:hover {{ background-color: #f1f1f1; }}
                img {{ max-width: 100%; height: auto; }}
                .negative {{ color: #e74c3c; }}
                .positive {{ color: #27ae60; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            
            <div class="container">
                <div class="card half">
                    <h3>Key Metrics</h3>
                    <table>
                        {''.join([f"<tr><td>{k}</td><td class='{self._get_color_class(stats[k])}'>{self._fmt(k, stats[k])}</td></tr>" for k in table_keys])}
                    </table>
                </div>
                
                <div class="card half">
                     <h3>Equity, Drawdown & Monte Carlo</h3>
                     <img src="data:image/png;base64,{img_str}" />
                </div>
                
                <div class="card">
                    <h3>Trade Breakdown (Top 5 Losing Trades)</h3>
                    <p>Understanding why trades fail is the fastest way to improve a strategy. These trades represent your largest realized losses.</p>
                    <table>
                        <tr>
                            <th>DateTime</th><th>Symbol</th><th>Side</th><th>PnL</th><th>Market Regime</th>
                        </tr>
                        {''.join([f"<tr><td>{t['datetime']}</td><td>{t['symbol']}</td><td>{t['side']}</td><td class='negative'>${t['realized_pnl']:,.2f}</td><td>{t.get('regime', 'N/A')}</td></tr>" for t in stats.get('Top_Losers', [])])}
                    </table>
                </div>

                <div class="card">
                    <h3>Full Trade Log</h3>
                    <table>
                        <tr>
                            <th>DateTime</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Comm</th><th>Mult</th>
                        </tr>
        """
        
        for trade in self.portfolio.trade_log:
            html += f"""
                            <tr>
                                <td>{trade['datetime']}</td>
                                <td>{trade['symbol']}</td>
                                <td>{trade['side']}</td>
                                <td>{trade['quantity']}</td>
                                <td>{trade['price']:.2f}</td>
                                <td>{trade['commission']:.2f}</td>
                                <td>{trade.get('multiplier', 1.0)}</td>
                            </tr>
            """
            
        html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open("backtest_report.html", "w") as f:
            f.write(html)
        print("Report saved to backtest_report.html")
        
        # Auto-Open
        try:
            webbrowser.open('file://' + os.path.realpath("backtest_report.html"))
        except Exception:
            pass

    def _generate_plots(self, df, sim_curves=None, benchmark_series=None):
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Monte Carlo Cones
        if sim_curves:
            for sim in sim_curves:
                ax1.plot(df.index, sim, color='gray', alpha=0.1, linewidth=0.5)
        
        # Equity Curve
        ax1.plot(df.index, df['equity'], label='Strategy Equity', color='blue', linewidth=1.5)
        
        # Benchmark
        if benchmark_series is not None and not benchmark_series.empty:
            # Rebase benchmark to match initial equity
            initial = df['equity'].iloc[0]
            # Align benchmark to df index
            bench = benchmark_series.reindex(df.index, method='ffill').fillna(method='bfill')
            # Normalize
            bench_norm = (bench / bench.iloc[0]) * initial
            ax1.plot(bench_norm.index, bench_norm, label='Benchmark (SPX)', color='gray', alpha=0.6, linestyle='--')

        ax1.set_title('Equity Curve vs Benchmark')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Drawdown
        ax2.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(df.index, df['drawdown'], color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('% Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def _fmt(self, key, value):
        try:
            if 'Ratio' in key or 'Equity' in key or 'std' in key or 'mean' in key:
                 return f"{float(value):,.2f}"
            if 'p5' in key or 'p50' in key or 'p95' in key:
                 return f"{float(value):,.2f}"
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return str(value)

    def _get_color_class(self, value):
        try:
            val = float(value)
            return 'positive' if val >= 0 else 'negative'
        except (TypeError, ValueError):
            return ''
