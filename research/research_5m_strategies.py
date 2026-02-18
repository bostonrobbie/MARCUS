"""
Optimized 5-Minute Strategy Research Script
Uses vectorized backtesting for fast execution (seconds vs minutes)
Includes properly designed strategies with risk management
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'StrategyPipeline'))

import pandas as pd
import numpy as np
from datetime import time
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# Performance Metrics
# ==========================================
def calculate_sharpe(returns, periods_per_year=252*78):
    """Calculate annualized Sharpe ratio for 5m bars (~78 bars/day)"""
    if returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def calculate_metrics(equity_curve, returns, signals):
    """Calculate comprehensive performance metrics"""
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(equity_curve)

    # Trade analysis (count position changes)
    position_changes = signals.diff().abs()
    trade_count = (position_changes > 0).sum() // 2  # Entry + exit = 1 round trip

    # Win rate from returns where we had a position
    active_returns = returns[signals.shift(1).fillna(0) != 0]
    if len(active_returns) > 0:
        winning_bars = (active_returns > 0).sum()
        total_bars = len(active_returns)
        win_rate = winning_bars / max(total_bars, 1)
    else:
        win_rate = 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 0.0001)

    # CAGR
    years = len(equity_curve) / (252 * 78)  # Approximate trading bars per year
    if years > 0 and equity_curve.iloc[0] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
    else:
        cagr = 0

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Trade Count': trade_count
    }


# ==========================================
# ATR Calculation (used by multiple strategies)
# ==========================================
def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


# ==========================================
# Strategy 1: Opening Range Breakout (ORB)
# Classic intraday momentum strategy
# ==========================================
def orb_signals(df, orb_minutes=15, ema_filter=50, atr_period=14, atr_filter_mult=2.5):
    """
    Opening Range Breakout Strategy
    - Define opening range from first 15 minutes (9:30-9:45)
    - Trade breakouts in direction of EMA trend
    - Filter out days with excessive ATR (too volatile)
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # EMA for trend filter
    ema = close.ewm(span=ema_filter, adjust=False).mean()

    # ATR for volatility filter
    atr = calculate_atr(df, atr_period)

    # Time components
    times = df.index.time
    dates = df.index.date

    # Initialize
    signals = pd.Series(0.0, index=df.index)
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    # Group by date to calculate ORB
    for date in pd.unique(dates):
        day_mask = dates == date
        day_data = df[day_mask]

        if len(day_data) < 20:  # Need enough bars
            continue

        # Get ORB window (9:30 - 9:45)
        orb_mask = [(time(9, 30) <= t < time(9, 45)) for t in day_data.index.time]
        orb_data = day_data[orb_mask]

        if len(orb_data) == 0:
            continue

        orb_high = orb_data['High'].max()
        orb_low = orb_data['Low'].min()
        orb_range = orb_high - orb_low

        # Get trading window (9:45 - 15:30)
        trade_mask = [(time(9, 45) <= t < time(15, 30)) for t in day_data.index.time]
        trade_indices = day_data.index[trade_mask]

        for idx in trade_indices:
            current_close = close[idx]
            current_high = high[idx]
            current_low = low[idx]
            current_ema = ema[idx]
            current_atr = atr[idx]

            # ATR filter - skip very volatile days
            if pd.isna(current_atr) or orb_range > current_atr * atr_filter_mult:
                continue

            if position == 0:
                # Long entry: breakout above ORB high + trend filter
                if current_close > orb_high and current_close > current_ema:
                    position = 1
                    entry_price = current_close
                    stop_loss = entry_price - (current_atr * 2)
                    take_profit = entry_price + (current_atr * 3)

                # Short entry: breakdown below ORB low + trend filter
                elif current_close < orb_low and current_close < current_ema:
                    position = -1
                    entry_price = current_close
                    stop_loss = entry_price + (current_atr * 2)
                    take_profit = entry_price - (current_atr * 3)

            elif position == 1:
                # Long exit: stop or target
                if current_low <= stop_loss or current_high >= take_profit:
                    position = 0

            elif position == -1:
                # Short exit: stop or target
                if current_high >= stop_loss or current_low <= take_profit:
                    position = 0

            signals[idx] = position

        # End of day: flatten
        eod_mask = [t >= time(15, 30) for t in day_data.index.time]
        eod_indices = day_data.index[eod_mask]
        signals[eod_indices] = 0
        position = 0

    return signals


# ==========================================
# Strategy 2: Mean Reversion with Bollinger Bands
# Counter-trend strategy for ranging markets
# ==========================================
def bollinger_mean_reversion_signals(df, bb_period=20, bb_std=2.0, rsi_period=14,
                                      oversold=30, overbought=70, atr_period=14):
    """
    Mean Reversion using Bollinger Bands + RSI confirmation
    - Buy when price touches lower band AND RSI is oversold
    - Sell when price touches upper band AND RSI is overbought
    - Exit at middle band (mean)
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # Bollinger Bands
    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    # ATR for stops
    atr = calculate_atr(df, atr_period)

    # Time filter (avoid overnight)
    times = df.index.time
    in_session = pd.Series([(time(9, 30) <= t < time(15, 45)) for t in times], index=df.index)

    # Initialize
    signals = pd.Series(0.0, index=df.index)
    position = 0
    entry_price = 0
    stop_loss = 0

    for i in range(bb_period, len(df)):
        idx = df.index[i]

        if not in_session[idx]:
            if position != 0:
                position = 0  # Flatten at session end
            signals[idx] = 0
            continue

        current_close = close.iloc[i]
        current_low = low.iloc[i]
        current_high = high.iloc[i]
        current_lower = lower_band.iloc[i]
        current_upper = upper_band.iloc[i]
        current_mid = sma.iloc[i]
        current_rsi = rsi.iloc[i]
        current_atr = atr.iloc[i]

        if pd.isna(current_atr):
            signals[idx] = position
            continue

        if position == 0:
            # Long entry: price at lower band + RSI oversold
            if current_close <= current_lower and current_rsi < oversold:
                position = 1
                entry_price = current_close
                stop_loss = entry_price - (current_atr * 1.5)

            # Short entry: price at upper band + RSI overbought
            elif current_close >= current_upper and current_rsi > overbought:
                position = -1
                entry_price = current_close
                stop_loss = entry_price + (current_atr * 1.5)

        elif position == 1:
            # Long exit: hit middle band (target) or stop loss
            if current_close >= current_mid or current_low <= stop_loss:
                position = 0

        elif position == -1:
            # Short exit: hit middle band (target) or stop loss
            if current_close <= current_mid or current_high >= stop_loss:
                position = 0

        signals[idx] = position

    return signals


# ==========================================
# Strategy 3: Trend Following with ADX Filter
# Only trade when trend is strong
# ==========================================
def adx_trend_following_signals(df, ema_fast=20, ema_slow=50, adx_period=14,
                                 adx_threshold=25, atr_period=14):
    """
    Trend Following with ADX Filter
    - Use EMA crossover for direction
    - Only trade when ADX > threshold (strong trend)
    - ATR-based stops and targets
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # EMAs
    ema_fast_line = close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_line = close.ewm(span=ema_slow, adjust=False).mean()

    # ADX calculation
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = calculate_atr(df, adx_period)

    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=adx_period).mean()
    adx = adx.fillna(0)

    # ATR for risk management
    atr_risk = calculate_atr(df, atr_period)

    # Time filter
    times = df.index.time
    in_session = pd.Series([(time(9, 30) <= t < time(15, 45)) for t in times], index=df.index)

    # Initialize
    signals = pd.Series(0.0, index=df.index)
    position = 0
    entry_price = 0
    stop_loss = 0

    for i in range(ema_slow + adx_period, len(df)):
        idx = df.index[i]

        if not in_session[idx]:
            if position != 0:
                position = 0
            signals[idx] = 0
            continue

        current_close = close.iloc[i]
        current_low = low.iloc[i]
        current_high = high.iloc[i]
        current_ema_fast = ema_fast_line.iloc[i]
        current_ema_slow = ema_slow_line.iloc[i]
        current_adx = adx.iloc[i]
        current_atr = atr_risk.iloc[i]
        prev_ema_fast = ema_fast_line.iloc[i-1]
        prev_ema_slow = ema_slow_line.iloc[i-1]

        if pd.isna(current_atr) or pd.isna(current_adx):
            signals[idx] = position
            continue

        if position == 0:
            # Long: EMA crossover + ADX strong
            if (current_ema_fast > current_ema_slow and
                prev_ema_fast <= prev_ema_slow and
                current_adx > adx_threshold):
                position = 1
                entry_price = current_close
                stop_loss = entry_price - (current_atr * 2)

            # Short: EMA crossunder + ADX strong
            elif (current_ema_fast < current_ema_slow and
                  prev_ema_fast >= prev_ema_slow and
                  current_adx > adx_threshold):
                position = -1
                entry_price = current_close
                stop_loss = entry_price + (current_atr * 2)

        elif position == 1:
            # Trailing stop: update if price moves in our favor
            new_stop = current_close - (current_atr * 2)
            if new_stop > stop_loss:
                stop_loss = new_stop

            # Exit on stop or trend reversal
            if current_low <= stop_loss or current_ema_fast < current_ema_slow:
                position = 0

        elif position == -1:
            # Trailing stop for short
            new_stop = current_close + (current_atr * 2)
            if new_stop < stop_loss:
                stop_loss = new_stop

            # Exit on stop or trend reversal
            if current_high >= stop_loss or current_ema_fast > current_ema_slow:
                position = 0

        signals[idx] = position

    return signals


# ==========================================
# Vectorized Backtest Engine
# ==========================================
def run_vectorized_backtest(df, signals, initial_capital=100000, commission=2.05, slippage=1.0):
    """
    Fast vectorized backtest with position-aware returns
    """
    # Returns
    returns = df['Close'].pct_change().fillna(0)

    # Position held for NEXT bar after signal
    position = signals.shift(1).fillna(0)

    # Turnover for costs
    turnover = position.diff().abs().fillna(0)

    # Transaction costs as % of price
    cost_pct = (commission + slippage) / df['Close']
    transaction_costs = turnover * cost_pct

    # Strategy returns
    strategy_returns = position * returns - transaction_costs

    # Equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()

    return equity, strategy_returns


# ==========================================
# Data Loading
# ==========================================
def load_data(symbol='NQ', interval='5m', start_date='2020-01-01', end_date='2024-12-31'):
    """Load data efficiently with date filtering"""
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')

    # Map interval to filename
    int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
    suffix = int_map.get(interval, 'm5')

    filepath = os.path.join(csv_dir, f"A2API-{symbol.upper()}-{suffix}.csv")

    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}")
        return None

    print(f"Loading data from {filepath}...")

    # Read only needed columns for speed
    df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close', 'volume'])

    # Parse dates
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns]

    # Filter date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    print(f"Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    return df


# ==========================================
# Main Runner
# ==========================================
def run_multi_backtest(start_date='2022-01-01', end_date='2024-12-31'):
    """
    Run comparative backtest on multiple strategies
    Using 3-year period for robust yet fast testing
    """
    print(f"\n{'='*60}")
    print(f"Running Optimized 5m Strategy Research")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    # Load data once
    df = load_data('NQ', '5m', start_date, end_date)
    if df is None or df.empty:
        print("ERROR: Could not load data")
        return

    results = {}

    # --- Strategy 1: Opening Range Breakout ---
    print("\n>>> Testing Strategy 1: Opening Range Breakout (ORB)...")
    signals1 = orb_signals(df)
    equity1, returns1 = run_vectorized_backtest(df, signals1)
    metrics1 = calculate_metrics(equity1, returns1, signals1)
    results['ORB'] = metrics1

    print(f"   Total Return: {metrics1['Total Return']:.2%}")
    print(f"   CAGR: {metrics1['CAGR']:.2%}")
    print(f"   Sharpe Ratio: {metrics1['Sharpe Ratio']:.2f}")
    print(f"   Max Drawdown: {metrics1['Max Drawdown']:.2%}")
    print(f"   Win Rate: {metrics1['Win Rate']:.1%}")
    print(f"   Profit Factor: {metrics1['Profit Factor']:.2f}")
    print(f"   Trade Count: {metrics1['Trade Count']}")

    # --- Strategy 2: Bollinger Mean Reversion ---
    print("\n>>> Testing Strategy 2: Bollinger Mean Reversion...")
    signals2 = bollinger_mean_reversion_signals(df)
    equity2, returns2 = run_vectorized_backtest(df, signals2)
    metrics2 = calculate_metrics(equity2, returns2, signals2)
    results['Bollinger MR'] = metrics2

    print(f"   Total Return: {metrics2['Total Return']:.2%}")
    print(f"   CAGR: {metrics2['CAGR']:.2%}")
    print(f"   Sharpe Ratio: {metrics2['Sharpe Ratio']:.2f}")
    print(f"   Max Drawdown: {metrics2['Max Drawdown']:.2%}")
    print(f"   Win Rate: {metrics2['Win Rate']:.1%}")
    print(f"   Profit Factor: {metrics2['Profit Factor']:.2f}")
    print(f"   Trade Count: {metrics2['Trade Count']}")

    # --- Strategy 3: ADX Trend Following ---
    print("\n>>> Testing Strategy 3: ADX Trend Following...")
    signals3 = adx_trend_following_signals(df)
    equity3, returns3 = run_vectorized_backtest(df, signals3)
    metrics3 = calculate_metrics(equity3, returns3, signals3)
    results['ADX Trend'] = metrics3

    print(f"   Total Return: {metrics3['Total Return']:.2%}")
    print(f"   CAGR: {metrics3['CAGR']:.2%}")
    print(f"   Sharpe Ratio: {metrics3['Sharpe Ratio']:.2f}")
    print(f"   Max Drawdown: {metrics3['Max Drawdown']:.2%}")
    print(f"   Win Rate: {metrics3['Win Rate']:.1%}")
    print(f"   Profit Factor: {metrics3['Profit Factor']:.2f}")
    print(f"   Trade Count: {metrics3['Trade Count']}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Find best by Sharpe
    best_strat = max(results.items(), key=lambda x: x[1]['Sharpe Ratio'])
    print(f"\nBest by Sharpe: {best_strat[0]} (Sharpe: {best_strat[1]['Sharpe Ratio']:.2f})")

    # Find best by Return
    best_return = max(results.items(), key=lambda x: x[1]['Total Return'])
    print(f"Best by Return: {best_return[0]} (Return: {best_return[1]['Total Return']:.2%})")

    # Risk-adjusted ranking
    print("\n--- Risk-Adjusted Rankings ---")
    ranked = sorted(results.items(), key=lambda x: x[1]['Sharpe Ratio'], reverse=True)
    for i, (name, m) in enumerate(ranked, 1):
        status = "PASS" if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25 else "FAIL"
        print(f"  {i}. [{status}] {name}: Sharpe={m['Sharpe Ratio']:.2f}, Return={m['Total Return']:.2%}, MaxDD={m['Max Drawdown']:.2%}, PF={m['Profit Factor']:.2f}")

    # Risk Manager Criteria Check
    print(f"\n{'='*60}")
    print("RISK MANAGER CRITERIA")
    print(f"{'='*60}")
    print("Required: Sharpe > 1.0, MaxDD < 25%, ProfitFactor > 1.2, Trades > 30")
    print("")

    for name, m in results.items():
        passes = []
        fails = []

        if m['Sharpe Ratio'] > 1.0:
            passes.append(f"Sharpe={m['Sharpe Ratio']:.2f}")
        else:
            fails.append(f"Sharpe={m['Sharpe Ratio']:.2f} (<1.0)")

        if m['Max Drawdown'] > -0.25:
            passes.append(f"MaxDD={m['Max Drawdown']:.2%}")
        else:
            fails.append(f"MaxDD={m['Max Drawdown']:.2%} (>25%)")

        if m['Profit Factor'] > 1.2:
            passes.append(f"PF={m['Profit Factor']:.2f}")
        else:
            fails.append(f"PF={m['Profit Factor']:.2f} (<1.2)")

        if m['Trade Count'] > 30:
            passes.append(f"Trades={m['Trade Count']}")
        else:
            fails.append(f"Trades={m['Trade Count']} (<30)")

        status = "APPROVED" if len(fails) == 0 else "VETO"
        print(f"{name}: {status}")
        if passes:
            print(f"  [+] Passes: {', '.join(passes)}")
        if fails:
            print(f"  [-] Fails: {', '.join(fails)}")
        print("")

    # Recommendation
    print(f"{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    approved = [name for name, m in results.items()
                if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25
                and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30]

    if approved:
        print(f"DEPLOY: {', '.join(approved)} meet all Risk Manager criteria")
    elif best_strat[1]['Sharpe Ratio'] > 0.5:
        print(f"REFINE: {best_strat[0]} shows promise but needs optimization")
        print("  Suggestions:")
        print("  - Adjust entry/exit criteria")
        print("  - Optimize ATR multipliers for stops/targets")
        print("  - Add additional filters (volume, time-of-day)")
    else:
        print("RESEARCH: All strategies need fundamental redesign")
        print("  Suggestions:")
        print("  - Review market regime detection")
        print("  - Consider different timeframes")
        print("  - Explore alternative signal generation methods")

    return results


if __name__ == "__main__":
    # Use 3-year period for robust testing
    run_multi_backtest("2022-01-01", "2024-12-31")
