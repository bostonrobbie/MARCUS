import sys
import os
import pandas as pd
import numpy as np

# ==========================================
# CUSTOM VECTORIZED BACKTESTER (Dependency Free)
# ==========================================
class SimpleBacktester:
    def __init__(self, data, strategy_func, commission=2.05, contract_value=20):
        """
        data: DataFrame with 'Close' and 'Open' (at the very least)
              For Overnight: Entry at Close (Index i), Exit at Open (Index i+1)
        strategy_func: Function that takes (df) and returns a Series of Signals (1, -1, 0) aligned with Index
        """
        self.data = data.copy()
        self.strategy_func = strategy_func
        self.commission = commission
        self.contract_value = contract_value # NQ $20/point
        
    def run(self):
        # 1. Generate Signals
        # Signal at Index i applies to the overnight jump from Close[i] to Open[i+1]
        self.data['Signal'] = self.strategy_func(self.data)
        
        # 2. Calculate Returns
        # Return = (Open[i+1] - Close[i]) * Signal[i]
        # Shift Open backwards to align "Next Open" with "Current Close"
        self.data['Next_Open'] = self.data['Open'].shift(-1)
        
        # Point Change
        self.data['Points'] = (self.data['Next_Open'] - self.data['Close']) * self.data['Signal']
        
        # PnL Calculation
        # PnL = Points * Value - Commission (per trade)
        # Commission is paid per contract per side? Usually per round trip or per side.
        # Simple assumption: $4.10 RT ($2.05 * 2) if Signal != 0
        rt_comm = self.commission * 2
        
        self.data['PnL'] = np.where(self.data['Signal'] != 0, 
                                    (self.data['Points'] * self.contract_value) - rt_comm, 
                                    0.0)
        
        # 3. Statistics
        total_pnl = self.data['PnL'].sum()
        trades = self.data['Signal'].abs().sum()
        winners = self.data[self.data['PnL'] > 0]
        losers = self.data[self.data['PnL'] <= 0] # Includes Breakeven/Comm fees
        win_rate = len(winners) / trades if trades > 0 else 0
        
        avg_win = winners['PnL'].mean() if len(winners) > 0 else 0
        avg_loss = losers['PnL'].mean() if len(losers) > 0 else 0
        profit_factor = abs(avg_win * len(winners) / (avg_loss * len(losers))) if avg_loss != 0 else 0
        
        # Equity Curve
        self.data['Equity'] = self.data['PnL'].cumsum()
        self.data['Equity'] = self.data['Equity'].fillna(0)
        
        # Drawdown
        peak = self.data['Equity'].cummax()
        dd = self.data['Equity'] - peak
        max_dd = dd.min()
        
        return {
            'Total PnL': total_pnl,
            'Trades': trades,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Max Drawdown': max_dd
        }

# ==========================================
# INDICATORS
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# STRATEGY LOGIC
# ==========================================
def strategy_mean_reversion(df):
    """
    Long Overnight if Daily RSI(2) < 10
    """
    # Resample to Daily to get Daily RSI
    daily = df.resample('B').agg({'Close': 'last', 'Open': 'first'}).dropna()
    daily['RSI'] = calculate_rsi(daily['Close'], period=2)
    
    # Logic: If RSI < 10, Buy Close
    daily['Signal'] = np.where(daily['RSI'] < 10, 1, 0)
    
    # Reindex to original if needed? 
    # With SimpleBacktester, we can run strictly on DAILY data because entry is Close, Exit is Open.
    # We do NOT need intraday 5m data for this specific logic if we have daily Open/Close.
    # Efficient!
    return daily

def strategy_short_bias(df):
    """
    Short Overnight if Daily RSI(14) > 70
    """
    daily = df.resample('B').agg({'Close': 'last', 'Open': 'first'}).dropna()
    daily['RSI'] = calculate_rsi(daily['Close'], period=14)
    daily['Signal'] = np.where(daily['RSI'] > 70, -1, 0) # Short
    return daily

def strategy_seasonality_tom(df):
    """
    Long Overnight only during Turn of Month (Last 4 days, First 4 days)
    """
    daily = df.resample('B').agg({'Close': 'last', 'Open': 'first'}).dropna()
    
    # Identify TOM days
    # Day > 25 or Day < 5 (Approximate)
    days = daily.index.day
    daily['Signal'] = np.where((days >= 25) | (days <= 4), 1, 0)
    
    return daily

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    data_path = r"c:\Users\User\Documents\AI\Quant_Lab\data\Intra OHLC\A2API-NQ-m5.csv" 
    
    if os.path.exists(data_path):
        print("Loading Data...")
        # Load all columns to ensure we have Open/Close
        df = pd.read_csv(data_path, index_col=None) # Read without index first
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        # Slicing with string requires DatetimeIndex
        df = df.loc['2015-01-01':'2024-12-31']
        
        # For Overnight Strategy: Entry at Close, Exit at Open.
        # We can simulate this using DAILY data.
        # Daily Close = Close of the day. Daily Open = Open of the day.
        # But wait! "Overnight" means Close(Today) -> Open(Tomorrow).
        # Standard Daily Data row: Date=T, Open=T, Close=T.
        # Logic: Buy Close(T) -> Sell Open(T+1).
        
        print("\n--- Strategy A: Overnight Mean Reversion (Deep Dip Buy) ---")
        # Logic is encapsulated in the function, returning a Daily DF with 'Signal'
        daily_mr = strategy_mean_reversion(df)
        
        # Run Backtest
        # We need to pass the 'daily_mr' df which has Open/Close/Signal
        # But wait, SimpleBacktester expects 'Next_Open' logic logic.
        # If we use Daily data: Close[i] is today's close. Open[i+1] is tomorrow's open.
        # My SimpleBacktester handles the shift logic.
        
        bt_mr = SimpleBacktester(daily_mr, lambda x: x['Signal'])
        stats_mr = bt_mr.run()
        print(f"RSI(2) < 10 | PnL: ${stats_mr['Total PnL']:,.2f} | PF: {stats_mr['Profit Factor']:.2f} | DD: ${stats_mr['Max Drawdown']:,.2f} | Trades: {stats_mr['Trades']}")
        
        print("\n--- Strategy B: Seasonality (Turn of Month) ---")
        daily_tom = strategy_seasonality_tom(df)
        bt_tom = SimpleBacktester(daily_tom, lambda x: x['Signal'])
        stats_tom = bt_tom.run()
        print(f"TOM (Day >= 25 or <= 4) | PnL: ${stats_tom['Total PnL']:,.2f} | PF: {stats_tom['Profit Factor']:.2f} | DD: ${stats_tom['Max Drawdown']:,.2f} | Trades: {stats_tom['Trades']}")

        print("\n--- Strategy C: Overnight Short Bias (Fade Heat) ---")
        daily_short = strategy_short_bias(df)
        bt_short = SimpleBacktester(daily_short, lambda x: x['Signal'])
        stats_short = bt_short.run()
        print(f"RSI(14) > 70 | PnL: ${stats_short['Total PnL']:,.2f} | PF: {stats_short['Profit Factor']:.2f} | DD: ${stats_short['Max Drawdown']:,.2f} | Trades: {stats_short['Trades']}")

    else:
        print("Data file not found.")
