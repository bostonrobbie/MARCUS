import pandas as pd
import numpy as np
import os
import sys

class AdvancedOvernightBacktester:
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
        # Assuming format matches previous success
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert to US/Eastern
        df.index = df.index.tz_convert('US/Eastern')
        df = df.loc['2015-01-01':'2024-12-31']
        return df

    def get_overnight_sessions(self):
        """
        Generator that yields (date, session_data_df, gap_info)
        Session: 18:00 ET (Prev Day) to 09:30 ET (Current Day)
        Gap Info: Close of 17:00 (Prev Day) vs Open of 18:00 (Prev Day)
        """
        # Get unique dates
        unique_dates = self.df.index.date
        unique_dates = sorted(list(set(unique_dates)))
        
        for i in range(1, len(unique_dates)):
            curr_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            
            # Timestamps
            # Session Start: 18:00 Prev Day
            sess_start = pd.Timestamp(prev_date).tz_localize('US/Eastern').replace(hour=18, minute=0)
            
            # Session End: 09:30 Current Day
            sess_end = pd.Timestamp(curr_date).tz_localize('US/Eastern').replace(hour=9, minute=30)
            
            # Gap Check (17:00 Prev Day Close)
            # Find last bar before 17:00 prev day? Or simply the bar AT 16:55/17:00?
            gap_ref_time = pd.Timestamp(prev_date).tz_localize('US/Eastern').replace(hour=17, minute=0)
            
            # Get Data Slices
            # 1. Previous Close (for Gap)
            # Look for data between 16:00 and 17:00 prev day
            prev_close_slice = self.df.loc[
                (self.df.index >= gap_ref_time - pd.Timedelta(minutes=60)) & 
                (self.df.index <= gap_ref_time + pd.Timedelta(minutes=15))
            ]
            
            prev_close_price = None
            if len(prev_close_slice) > 0:
                # Use the last available close in that window
                prev_close_price = prev_close_slice.iloc[-1]['Close']
                
            # 2. Session Data
            session_data = self.df.loc[sess_start:sess_end]
            
            if len(session_data) < 10:
                continue
                
            session_open_price = session_data.iloc[0]['Open']
            
            yield {
                'date': curr_date,
                'data': session_data,
                'prev_close': prev_close_price,
                'open_price': session_open_price,
                'sess_start': sess_start
            }

    def run_london_breakout(self):
        print("Running London Breakout (Asia Range 18:00-03:00)...")
        trades = []
        
        for sess in self.get_overnight_sessions():
            data = sess['data']
            
            # Asia Range: 18:00 to 03:00
            asia_end = sess['sess_start'].replace(hour=3, minute=0)
            # Handle date rollover if sess_start is 18:00 Prev, 03:00 is Current
            if asia_end < sess['sess_start']: 
                 # This shouldn't happen if we constructed timestamps correctly.
                 # sess_start is 18:00 Prev. Asia end is 03:00 Next Day.
                 # Timestamp.replace keeps the date? No.
                 # We need to act carefully.
                 pass
            
            # Re-construct Asia End correctly
            # It is 03:00 on the 'date' (Current Day)
            asia_end = pd.Timestamp(sess['date']).tz_localize('US/Eastern').replace(hour=3, minute=0)
            
            asia_data = data.loc[:asia_end]
            if len(asia_data) < 10: continue
            
            asia_high = asia_data['High'].max()
            asia_low = asia_data['Low'].min()
            
            # Trade Window: 03:00 to 09:30
            trade_data = data.loc[asia_end:]
            if len(trade_data) < 2: continue
            
            position = 0
            entry_price = 0
            
            for t, row in trade_data.iterrows():
                if position == 0:
                    # Breakout Long
                    if row['High'] > asia_high:
                        entry_price = max(asia_high, row['Open'])
                        position = 1
                        break
                    # Breakout Short
                    elif row['Low'] < asia_low:
                        entry_price = min(asia_low, row['Open'])
                        position = -1
                        break
            
            if position != 0:
                # Exit at 09:30 (Last bar of session)
                exit_price = trade_data.iloc[-1]['Close']
                pnl = 0
                if position == 1:
                    pnl = (exit_price - entry_price) * self.contract_value - (self.commission * 2)
                else:
                    pnl = (entry_price - exit_price) * self.contract_value - (self.commission * 2)
                trades.append(pnl)
                
        return trades

    def run_gap_continuation(self, threshold_pct=0.0025):
        print(f"Running Gap Continuation (Thresh: {threshold_pct*100}%)...")
        trades = []
        
        for sess in self.get_overnight_sessions():
            if sess['prev_close'] is None: continue
            
            prev_close = sess['prev_close']
            curr_open = sess['open_price']
            
            gap_pct = (curr_open - prev_close) / prev_close
            
            position = 0
            
            # Logic: Continuation
            if gap_pct > threshold_pct:
                position = 1
            elif gap_pct < -threshold_pct:
                position = -1
                
            if position != 0:
                # Entry: 18:00 Open (curr_open)
                entry_price = curr_open
                
                # Exit: 09:30 Open/Close (Last bar of session)
                exit_price = sess['data'].iloc[-1]['Close']
                
                pnl = 0
                if position == 1:
                    pnl = (exit_price - entry_price) * self.contract_value - (self.commission * 2)
                else:
                    pnl = (entry_price - exit_price) * self.contract_value - (self.commission * 2)
                trades.append(pnl)
                
        return trades

def analyze_results(trades, name):
    if not trades:
        print(f"{name}: No Trades")
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
    
    print(f"{name:<25} | PnL: ${total_pnl:<10,.2f} | PF: {pf:<5.2f} | WR: {win_rate:<6.1%} | Trades: {n_trades:<5} | DD: ${max_dd:,.2f}")

if __name__ == "__main__":
    data_path = r"c:\Users\User\Documents\AI\Quant_Lab\data\Intra OHLC\A2API-NQ-m5.csv"
    
    try:
        tester = AdvancedOvernightBacktester(data_path)
        
        print("\n" + "="*90)
        print(f"{'Strategy':<25} | {'PnL':<11} | {'PF':<5} | {'WR':<6} | {'Trades':<5} | {'Max DD':<10}")
        print("="*90)
        
        # 1. London Breakout
        trades_london = tester.run_london_breakout()
        analyze_results(trades_london, "London Breakout")
        
        # 2. Gap Continuation (0.25% and 0.5%)
        trades_gap_small = tester.run_gap_continuation(0.0025)
        analyze_results(trades_gap_small, "Gap Cont (>0.25%)")
        
        trades_gap_large = tester.run_gap_continuation(0.0050)
        analyze_results(trades_gap_large, "Gap Cont (>0.50%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
