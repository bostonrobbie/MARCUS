import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_standard_equity_curve(equity_series, strategy_name, filename="equity_curve.png"):
    """
    Generates a professional standardized equity curve PNG with metrics.
    
    Args:
        equity_series (pd.Series): Time series of equity values (DatetimeIndex).
        strategy_name (str): Name of the strategy for the title.
        filename (str): Output filename.
    """
    # 1. Calculate Metrics
    if len(equity_series) == 0:
        print("Warning: Empty equity series, skipping plot.")
        return

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    
    # CAGR approximation (assuming daily data is roughly continuous)
    days = (equity_series.index[-1] - equity_series.index[0]).days
    if days > 0:
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (365/days) - 1
    else:
        cagr = 0
        
    # Drawdown
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = dd.min()
    
    # Sharpe (Simple Daily approximation)
    daily_rets = equity_series.pct_change().dropna()
    if daily_rets.std() > 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sharpe = 0

    # 2. Setup Professional Plot
    plt.style.use('bmh') # Clean professional style
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Main Equity Line
    ax.plot(equity_series.index, equity_series.values, linewidth=1.5, color='#2c3e50', label='Equity')
    
    # Fill Under Curve
    ax.fill_between(equity_series.index, equity_series.values, equity_series.iloc[0], alpha=0.1, color='#2c3e50')
    
    # Watermark / Branding
    ax.text(0.99, 0.01, 'Quant Lab | AI-Generated Report', 
            transform=ax.transAxes, fontsize=10, color='gray', alpha=0.5,
            ha='right', va='bottom')

    # Title with Metrics
    title_text = f"{strategy_name}\n"
    subtitle_text = f"CAGR: {cagr:.1%} | MaxDD: {max_dd:.1%} | Sharpe: {sharpe:.2f} | Total Return: {total_return:.1%}"
    
    ax.set_title(title_text, fontsize=14, fontweight='bold', loc='left', pad=20)
    ax.text(0.0, 1.01, subtitle_text, transform=ax.transAxes, fontsize=11, color='#555555', va='bottom')

    # Formatting
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(''))
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Generated Standard Report: {filename}")

