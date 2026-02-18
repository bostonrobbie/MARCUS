import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# CUSTOM DAY-LOOP BACKTESTER
# ==========================================
class OrbBacktester:
    def __init__(self, data_path, commission=2.05, contract_value=20):
        self.data_path = data_path
        self.commission = commission
        self.contract_value = contract_value
        self.df = self.load_data()
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        # Specific loading logic for the NQ file we identified
        df = pd.read_csv(self.data_path, index_col=None)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert to US/Eastern for correct time filtering (9:30 ET)
        df.index = df.index.tz_convert('US/Eastern')
        
        # Filter range
        df = df.loc['2015-01-01':'2024-12-31']
        return df

    def run_orb_volatility(self, orb_minutes=30, vol_mode='NONE'):
        """
        Tests ORB Strategy with Volatility Filters:
        - vol_mode: 'NONE'
        - 'SQUEEZE': Only trade if Range < Avg Range (Expect Expansion)
        - 'EXPANSION': Only trade if Range > Avg Range (Expect Momentum)
        """
        label = f"ORB-{orb_minutes}m [{vol_mode}]"
        print(f"Running {label}...")
        
        trades = []
        grouped = self.df.groupby(self.df.index.date)
        
        # Pre-calc 30m Ranges
        # This is a bit complex in a generator, but solvable by pre-scan.
        # Efficient: Calculate all ranges first.
        ranges = {}
        dates = []
        
        # Pass 1: Get Ranges
        for date, day_data in grouped:
            if len(day_data) < 10: continue
            
            # ORB Times
            open_time = day_data.index[0].replace(hour=9, minute=30)
            if open_time not in day_data.index: continue # robustness
            orb_end_time = open_time + pd.Timedelta(minutes=orb_minutes)
            
            r_data = day_data.loc[open_time:orb_end_time]
            if len(r_data) > 0:
                rng = r_data['High'].max() - r_data['Low'].min()
                ranges[date] = rng
                dates.append(date)
            else:
                ranges[date] = np.nan
        
        # Pass 2: Calculate Rolling Avg
        if not ranges: return []
        
        range_series = pd.Series(ranges).sort_index()
        # Avg of last 10 sessions (shifted by 1 to use PREVIOUS data only)
        avg_range_series = range_series.rolling(10).mean().shift(1)
        
        # Pass 3: Trade
        grouped = self.df.groupby(self.df.index.date) # Re-group or iterate dates? 
        # Re-group ensures we have data access.
        
        for date, day_data in grouped:
            if len(day_data) < 10: continue
            if date not in avg_range_series.index: continue
            
            threshold = avg_range_series[date]
            if pd.isna(threshold): continue
            
            # Vol Filter
            current_range = range_series.get(date, np.nan)
            if pd.isna(current_range): continue
            
            if vol_mode == 'SQUEEZE':
                if current_range > threshold: continue # Skip Large
            elif vol_mode == 'EXPANSION':
                if current_range < threshold: continue # Skip Small
                
            # Execution
            open_time = day_data.index[0].replace(hour=9, minute=30)
            orb_end_time = open_time + pd.Timedelta(minutes=orb_minutes)
            session_end = day_data.index[0].replace(hour=12, minute=0)
            
            trading_data = day_data.loc[orb_end_time:session_end]
            if len(trading_data) == 0: continue
            
            orb_high = day_data.loc[open_time:orb_end_time]['High'].max()
            orb_low = day_data.loc[open_time:orb_end_time]['Low'].min()
            
            position = 0 
            entry_price = 0
            
            for t, row in trading_data.iterrows():
                if position == 0:
                    if row['High'] > orb_high:
                        entry_price = max(orb_high, row['Open'])
                        position = 1
                        break
                    elif row['Low'] < orb_low:
                        entry_price = min(orb_low, row['Open'])
                        position = -1
                        break
            
    def run_orb_combined(self, orb_minutes=30, vol_mode='SQUEEZE', stop_mode='OPPOSITE'):
        """
        Combines Squeeze Filter with Stop Logic.
        """
        label = f"ORB-{orb_minutes}m [{vol_mode} + {stop_mode}]"
        print(f"Running {label}...")
        
        trades = []
        grouped = self.df.groupby(self.df.index.date)
        
        # Pre-calc Ranges (Copy logic from before or assume it's clean)
        ranges = {}
        for date, day_data in grouped:
            if len(day_data) < 10: continue
            open_time = day_data.index[0].replace(hour=9, minute=30)
            orb_end = open_time + pd.Timedelta(minutes=orb_minutes)
            r_data = day_data.loc[open_time:orb_end]
            if len(r_data) > 0:
                rng = r_data['High'].max() - r_data['Low'].min()
                ranges[date] = rng
                dates.append(date)
            else: 
                print(f"DEBUG: No data for {date} at {open_time}. Day start: {day_data.index[0]}")
                ranges[date] = np.nan
            
        range_series = pd.Series(ranges).sort_index()
        avg_range_series = range_series.rolling(10).mean().shift(1)
        
        print(f"DEBUG: Found {len(ranges)} daily ranges.")
        if len(ranges) > 0:
            print(f"DEBUG: Sample Range: {list(ranges.values())[:3]}")
            print(f"DEBUG: Sample Avg keys: {list(avg_range_series.index)[:3]}")
        
        grouped = self.df.groupby(self.df.index.date)
        for date, day_data in grouped:
            if len(day_data) < 10: continue
            if date not in avg_range_series.index or pd.isna(avg_range_series[date]): continue
            
            threshold = avg_range_series[date]
            current_range = ranges.get(date, np.nan)
            
            if vol_mode == 'SQUEEZE' and current_range > threshold: continue
            if vol_mode == 'EXPANSION' and current_range < threshold: continue
            
            # ORB Logic
            open_time = day_data.index[0].replace(hour=9, minute=30)
            orb_end_time = open_time + pd.Timedelta(minutes=orb_minutes)
            session_end = day_data.index[0].replace(hour=12, minute=0)
            
            trading_data = day_data.loc[orb_end_time:session_end]
            if len(trading_data) == 0: continue
            
            orb_high = day_data.loc[open_time:orb_end_time]['High'].max()
            orb_low = day_data.loc[open_time:orb_end_time]['Low'].min()
            # Trade Logic (Single Pass)
            orb_mid = (orb_high + orb_low) / 2
            position = 0 
            entry_price = 0
            stop_price = 0
            exit_price = trading_data.iloc[-1]['Close'] # Default EOD
            
            # Find Entry
            entry_idx = -1
            for i in range(len(trading_data)):
                row = trading_data.iloc[i]
                if row['High'] > orb_high:
                    entry_price = max(orb_high, row['Open'])
                    position = 1
                    if stop_mode == 'OPPOSITE': stop_price = orb_low
                    elif stop_mode == 'MIDPOINT': stop_price = orb_mid
                    entry_idx = i
                    break
                elif row['Low'] < orb_low:
                    entry_price = min(orb_low, row['Open'])
                    position = -1
                    if stop_mode == 'OPPOSITE': stop_price = orb_high
                    elif stop_mode == 'MIDPOINT': stop_price = orb_mid
                    entry_idx = i
                    break
            
            # If Entered, Manage Trade
            if position != 0:
                # Check stops from NEXT bar (or same bar? Backtesting usually checks same bar for H/L violations if candle is large)
                # For robustness, we check same bar for Stop violation? 
                # If we entered at High, and Low < Stop, we are stopped out in same bar.
                # Let's check from entry_idx (inclusive) for simplicity/conservatism.
                
                for i in range(entry_idx, len(trading_data)):
                    row = trading_data.iloc[i]
                    if position == 1:
                        if row['Low'] < stop_price:
                            exit_price = stop_price
                            break
                    elif position == -1:
                        if row['High'] > stop_price:
                            exit_price = stop_price
                            break
                            
                pnl = 0
                if position == 1:
                    pnl = (exit_price - entry_price) * self.contract_value - (self.commission * 2)
                else:
                    pnl = (entry_price - exit_price) * self.contract_value - (self.commission * 2)
                trades.append({'PnL': pnl, 'Side': 'Long' if position == 1 else 'Short'})

        return trades

def analyze_trades(trades_list):
    if not trades_list: return {'PnL': 0, 'PF': 0, 'WR': 0, 'Trades': 0}
    
    df = pd.DataFrame(trades_list)
    total_pnl = df['PnL'].sum()
    n_trades = len(df)
    win_rate = len(df[df['PnL'] > 0]) / n_trades
    
    winners = df[df['PnL'] > 0]['PnL']
    losers = df[df['PnL'] <= 0]['PnL']
    avg_win = winners.mean() if not winners.empty else 0
    avg_loss = losers.mean() if not losers.empty else 0
    pf = abs((avg_win * len(winners)) / (avg_loss * len(losers))) if not losers.empty else 0
    
    return {'PnL': total_pnl, 'PF': pf, 'WR': win_rate, 'Trades': n_trades}

if __name__ == "__main__":
    data_path = r"c:\Users\User\Documents\AI\Quant_Lab\data\Intra OHLC\A2API-NQ-m5.csv"
    
    try:
        tester = OrbBacktester(data_path)
        metrics = []
        
        # Test Combined
        # 30m Squeeze + Opposite
        trades_30 = tester.run_orb_combined(30, 'SQUEEZE', 'OPPOSITE')
        metrics.append({**analyze_trades(trades_30), 'Type': '30m Sqz+Opp'})
        
        # 15m Squeeze + Opposite
        trades_15 = tester.run_orb_combined(15, 'SQUEEZE', 'OPPOSITE')
        metrics.append({**analyze_trades(trades_15), 'Type': '15m Sqz+Opp'})
        
        # Control: 30m Squeeze No Stop
        trades_30_ns = tester.run_orb_combined(30, 'SQUEEZE', 'NONE')
        metrics.append({**analyze_trades(trades_30_ns), 'Type': '30m Sqz+None'})

        print("\n" + "="*80)
        print(f"{'Strategy Type':<30} | {'Total PnL':<12} | {'PF':<6} | {'Win Rate':<8} | {'Trades':<8}")
        print("="*80)
        
        for m in metrics:
             print(f"{m['Type']:<30} | ${m['PnL']:<11,.2f} | {m['PF']:<6.2f} | {m['WR']:<8.2%} | {m['Trades']:<8}")
             
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
