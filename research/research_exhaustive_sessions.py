import pandas as pd
import numpy as np
import os
import sys

class ExhaustiveSessionBacktester:
    def __init__(self, data_path, commission=2.05, contract_value=20):
        self.data_path = data_path
        self.commission = commission
        self.contract_value = contract_value
        self.df = self.load_data()
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path, index_col=None)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert to US/Eastern
        df.index = df.index.tz_convert('US/Eastern')
        df = df.loc['2015-01-01':'2024-12-31']
        return df

    def get_overnight_sessions(self):
        """
        Generator yielding (date, full_session_data)
        Full Session: 18:00 Prev Day to 09:30 Current Day
        """
        unique_dates = sorted(list(set(self.df.index.date)))
        
        for i in range(1, len(unique_dates)):
            curr_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            
            # 18:00 Prev to 09:30 Curr
            sess_start = pd.Timestamp(prev_date).tz_localize('US/Eastern').replace(hour=18, minute=0)
            sess_end = pd.Timestamp(curr_date).tz_localize('US/Eastern').replace(hour=9, minute=30)
            
            # Extract Data
            data = self.df.loc[sess_start:sess_end]
            if len(data) < 10: continue
            
            yield {
                'date': curr_date,
                'data': data,
                'start_time': sess_start
            }

    # ==========================================
    # 1. ASIAN SESSION VARIANTS
    # ==========================================
    def run_asian_fade(self):
        """
        Define Range 18:00-02:00. 
        If Price at 02:00-03:00 is at Range High -> Short.
        If Price at 02:00-03:00 is at Range Low -> Long.
        Exit at 09:30.
        """
        print("Running Asian Range Fade...")
        trades = []
        
        for sess in self.get_overnight_sessions():
            data = sess['data']
            
            # Define Range Period: 18:00 to 02:00 ET (Next Day usually, or same session)
            # 18:00 is start. 02:00 is start + 8 hours.
            range_end_time = sess['start_time'] + pd.Timedelta(hours=8) 
            
            range_data = data.loc[:range_end_time]
            if len(range_data) < 10: continue
            
            high = range_data['High'].max()
            low = range_data['Low'].min()
            rng = high - low
            if rng == 0: continue
            
            # Signal Window: 02:00 to 03:00
            signal_start = range_end_time
            signal_end = signal_start + pd.Timedelta(hours=1)
            
            signal_data = data.loc[signal_start:signal_end]
            if len(signal_data) == 0: continue
            
            # Entry Logic: Fade Extremes
            # Simple Trigger: If any bar High > High - 0.1*Rng -> Short
            # If any bar Low < Low + 0.1*Rng -> Long
            
            # Only take first signal
            position = 0
            entry_price = 0
            
            for t, row in signal_data.iterrows():
                # Check for fading high
                if row['Close'] > (high - 0.1 * rng):
                    position = -1
                    entry_price = row['Close']
                    break
                # Check for fading low
                elif row['Close'] < (low + 0.1 * rng):
                    position = 1
                    entry_price = row['Close']
                    break
            
            if position != 0:
                # Exit at 09:30
                exit_price = data.iloc[-1]['Close']
                pnl = (exit_price - entry_price) * position * self.contract_value - (self.commission * 2)
                trades.append(pnl)
                
        return trades

    def run_asian_squeeze(self):
        """
        If Asian Range (18:00-03:00) < Threshold, Buy Breakout at 03:00+.
        """
        print("Running Asian Squeeze Breakout...")
        trades = []
        
        # Calculate ATR for relative squeeze
        daily_atr = self.df.resample('D').agg({'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        daily_atr['TR'] = np.maximum(daily_atr['High'] - daily_atr['Low'], 
                                     np.maximum(abs(daily_atr['High'] - daily_atr['Close'].shift(1)), 
                                                abs(daily_atr['Low'] - daily_atr['Close'].shift(1))))
        daily_atr['ATR'] = daily_atr['TR'].rolling(14).mean()
        
        for sess in self.get_overnight_sessions():
            # Get Prev Day ATR
            day_str = str(sess['date'])
            # We need ATR of previous day
            prev_day_idx = sess['date'] - pd.Timedelta(days=1)
            # Find closest available date in ATR
            # Simplification: use global lookup
            try:
                # Need to look up strictly before current date
                current_timestamp = pd.Timestamp(sess['date']).tz_localize(None) 
                # This is messy with timezones. 
                # Alternative: Just calculate absolute range points.
                # Let's use a fixed "Squeeze" definition for simplicity or just Range Points.
                # < 40 points range? 
                pass
            except:
                continue

            data = sess['data']
            
            # Asian Range 18:00 - 03:00
            range_end = sess['start_time'] + pd.Timedelta(hours=9)
            asia_data = data.loc[:range_end]
            if len(asia_data) < 10: continue
            
            h = asia_data['High'].max()
            l = asia_data['Low'].min()
            r = h - l
            
            # Squeeze Condition: Range < 50 points (Approximation for 'tight')
            if r > 50: continue 
            
            # Trade Breakout 03:00 - 09:30
            trade_data = data.loc[range_end:]
            position = 0
            entry_price = 0
            
            for t, row in trade_data.iterrows():
                if row['High'] > h:
                    position = 1
                    entry_price = max(h, row['Open'])
                    break
                elif row['Low'] < l:
                    position = -1
                    entry_price = min(l, row['Open'])
                    break
            
            if position != 0:
                exit_price = data.iloc[-1]['Close']
                pnl = (exit_price - entry_price) * position * self.contract_value - (self.commission * 2)
                trades.append(pnl)
                
        return trades

    # ==========================================
    # 2. LONDON SESSION VARIANTS
    # ==========================================
    def run_london_reversal(self):
        """
        Fade the London Breakout. 
        If Price breaks Asian Range (18:00-03:00) during 03:00-04:00, 
        and then CLOSES back inside -> Trade Reverse.
        """
        print("Running London Reversal (Fakeout)...")
        trades = []
        
        for sess in self.get_overnight_sessions():
            data = sess['data']
            
            # Asian Range 18:00 - 03:00
            asia_end = sess['start_time'] + pd.Timedelta(hours=9)
            asia_data = data.loc[:asia_end]
            if len(asia_data) < 10: continue
            
            h = asia_data['High'].max()
            l = asia_data['Low'].min()
            
            # Watch Window: 03:00 - 04:30
            watch_end = asia_end + pd.Timedelta(minutes=90)
            watch_data = data.loc[asia_end:watch_end]
            
            # State Machine
            # 0: Waiting
            # 1: Breakout Occurred
            # 2: Reversal Triggered (Entry)
            state = 0
            breakout_side = 0 # 1 High, -1 Low
            entry_price = 0
            
            for t, row in watch_data.iterrows():
                if state == 0:
                    if row['High'] > h:
                        state = 1
                        breakout_side = 1
                    elif row['Low'] < l:
                        state = 1
                        breakout_side = -1
                
                elif state == 1:
                    # Look for close back inside
                    if breakout_side == 1: # Broke High
                        if row['Close'] < h: # False Breakout High -> Short
                            entry_price = row['Close']
                            state = 2 # Entered Short
                            break
                    elif breakout_side == -1: # Broke Low
                        if row['Close'] > l: # False Breakout Low -> Long
                            entry_price = row['Close']
                            state = 2 # Entered Long
                            break
            
            if state == 2:
                # Exit at 09:30
                exit_price = data.iloc[-1]['Close']
                # If we entered Short (False High Break), position is -1
                position = -1 if breakout_side == 1 else 1
                
                pnl = (exit_price - entry_price) * position * self.contract_value - (self.commission * 2)
                trades.append(pnl)
                
        return trades

    def run_london_trend(self):
        """
        Simple Time Hold: Enter 03:00 Open, Exit 09:00.
        Pure London Drift.
        """
        print("Running London Trend (03:00-09:00)...")
        trades = []
        
        for sess in self.get_overnight_sessions():
            data = sess['data']
            
            # Start 03:00, End 09:00
            start_time = sess['start_time'] + pd.Timedelta(hours=9)
            end_time = sess['start_time'] + pd.Timedelta(hours=15)
            
            seg_data = data.loc[start_time:end_time]
            if len(seg_data) < 2: continue
            
            entry_price = seg_data.iloc[0]['Open']
            exit_price = seg_data.iloc[-1]['Close']
            
            # Long Only
            pnl = (exit_price - entry_price) * self.contract_value - (self.commission * 2)
            trades.append(pnl)
            
        return trades
        
    # ==========================================
    # 3. HOURLY SEGMENTATION
    # ==========================================
    def run_hourly_drift(self, start_hour_offset, end_hour_offset, name):
        """
        Generic hourly hold. Offset from 18:00 session start.
        00:00 is +6 hours.
        06:00 is +12 hours.
        09:30 is +15.5 hours.
        """
        print(f"Running {name}...")
        trades = []
        
        for sess in self.get_overnight_sessions():
            data = sess['data']
            
            t_start = sess['start_time'] + pd.Timedelta(hours=start_hour_offset)
            t_end = sess['start_time'] + pd.Timedelta(hours=end_hour_offset)
            
            seg = data.loc[t_start:t_end]
            if len(seg) < 2: continue
            
            entry = seg.iloc[0]['Open']
            # If t_end matches specific bar, use that close. 
            # slicing includes end, so take last bar close.
            exit_p = seg.iloc[-1]['Close']
            
            pnl = (exit_p - entry) * self.contract_value - (self.commission * 2)
            trades.append(pnl)
            
        return trades

def analyze_results(trades, name):
    if not trades:
        print(f"{name:<30} | NO TRADES")
        return
        
    trades_np = np.array(trades)
    total_pnl = np.sum(trades_np)
    n_trades = len(trades_np)
    win_rate = np.mean(trades_np > 0)
    
    winners = trades_np[trades_np > 0]
    losers = trades_np[trades_np <= 0]
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(losers) if len(losers) > 0 else 0
    pf = abs((avg_win * len(winners)) / (avg_loss * len(losers))) if len(losers) > 0 else 0
    
    # Drawdown
    equity = np.cumsum(trades_np)
    peak = np.maximum.accumulate(equity)
    max_dd = np.min(equity - peak)
    
    print(f"{name:<30} | PnL: ${total_pnl:<11,.2f} | PF: {pf:<5.2f} | WR: {win_rate:<6.1%} | Trades: {n_trades:<5} | DD: ${max_dd:,.2f}")

if __name__ == "__main__":
    data_path = r"c:\Users\User\Documents\AI\Quant_Lab\data\Intra OHLC\A2API-NQ-m5.csv"
    
    try:
        tester = ExhaustiveSessionBacktester(data_path)
        
        print("\n" + "="*100)
        print(f"{'Strategy Variant':<30} | {'PnL':<12} | {'PF':<5} | {'WR':<6} | {'Trades':<5} | {'Max DD':<10}")
        print("="*100)
        
        # Asian
        analyze_results(tester.run_asian_fade(), "Asian Range Fade (Reversion)")
        analyze_results(tester.run_asian_squeeze(), "Asian Squeeze Breakout")
        
        # London
        analyze_results(tester.run_london_reversal(), "London Reversal (Fakeout)")
        analyze_results(tester.run_london_trend(), "London Trend (03:00-09:00)")
        
        # Hourly
        analyze_results(tester.run_hourly_drift(6, 12, "Deep Night Drift (00:00-06:00)"), "Deep Night (00-06)")
        analyze_results(tester.run_hourly_drift(12, 15.5, "Pre-Market Drift (06:00-09:30)"), "Pre-Market (06-09:30)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
