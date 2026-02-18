import pandas as pd
import numpy as np
import os

class LondonSqueezeBacktester:
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
        Yields (date, full_session_data)
        Full Session: 18:00 Prev Day to 09:30 Current Day
        """
        unique_dates = sorted(list(set(self.df.index.date)))
        
        for i in range(1, len(unique_dates)):
            curr_date = unique_dates[i]
            prev_date = unique_dates[i-1]
            
            # 18:00 Prev to 09:30 Curr
            sess_start = pd.Timestamp(prev_date).tz_localize('US/Eastern').replace(hour=18, minute=0)
            sess_end = pd.Timestamp(curr_date).tz_localize('US/Eastern').replace(hour=9, minute=0) # End at 09:00 for London Close
            
            data = self.df.loc[sess_start:sess_end]
            if len(data) < 10: continue
            
            yield {
                'date': curr_date,
                'data': data,
                'start_time': sess_start
            }

    def run_london_squeeze(self, vol_mode='SQUEEZE', stop_mode='OPPOSITE'):
        """
        Asian Range: 18:00 - 03:00
        Trigger: Breakout > 03:00
        Filter: Range < Avg Range (Squeeze)
        """
        label = f"London [{vol_mode} + {stop_mode}]"
        print(f"Running {label}...")
        
        # 1. Calculate All Asian Ranges first
        asian_ranges = {}
        
        # Iterate once to get ranges
        # Using a separate lightweight loop to avoid memory overhead if possible, 
        # but generator is fine.
        
        temp_tester = self.get_overnight_sessions()
        for sess in temp_tester:
            data = sess['data']
            # Asian Range 18:00 - 03:00
            # 18:00 is start. 03:00 is start + 9 hours.
            asian_end = sess['start_time'] + pd.Timedelta(hours=9)
            asia_data = data.loc[:asian_end]
            
            if len(asia_data) > 0:
                rng = asia_data['High'].max() - asia_data['Low'].min()
                asian_ranges[sess['date']] = rng
            else:
                asian_ranges[sess['date']] = np.nan
                
        # 2. Rolling Average
        range_series = pd.Series(asian_ranges).sort_index()
        avg_range_series = range_series.rolling(10).mean().shift(1)
        
        # 3. Trade
        trades = []
        
        for sess in self.get_overnight_sessions():
            date = sess['date']
            if date not in avg_range_series.index or pd.isna(avg_range_series[date]): continue
            
            threshold = avg_range_series[date]
            current_range = asian_ranges.get(date, np.nan)
            
            if pd.isna(current_range): continue
            
            # Vol Filter
            if vol_mode == 'SQUEEZE' and current_range > threshold: continue
            if vol_mode == 'EXPANSION' and current_range < threshold: continue
            
            # define execution
            data = sess['data']
            asian_end = sess['start_time'] + pd.Timedelta(hours=9) # 03:00 ET
            
            # Range Levels
            asia_data = data.loc[:asian_end]
            if len(asia_data) == 0: continue
            
            h = asia_data['High'].max()
            l = asia_data['Low'].min()
            
            # Trade Window: 03:00 - 09:00
            trade_data = data.loc[asian_end:]
            if len(trade_data) == 0: continue
            
            position = 0
            entry_price = 0
            stop_price = 0
            
            # Find Entry
            entry_idx = -1
            for i in range(len(trade_data)):
                row = trade_data.iloc[i]
                if row['High'] > h:
                    entry_price = max(h, row['Open'])
                    position = 1
                    if stop_mode == 'OPPOSITE': stop_price = l
                    entry_idx = i
                    break
                elif row['Low'] < l:
                    entry_price = min(l, row['Open'])
                    position = -1
                    if stop_mode == 'OPPOSITE': stop_price = h
                    entry_idx = i
                    break
            
            if position != 0:
                exit_price = trade_data.iloc[-1]['Close'] # 09:00 Close
                pnl = 0
                
                # Check Stop (Only if Mode is not NONE)
                if stop_mode != 'NONE':
                    for i in range(entry_idx, len(trade_data)):
                        row = trade_data.iloc[i]
                        if position == 1:
                            if row['Low'] < stop_price:
                                exit_price = stop_price
                                break
                        elif position == -1:
                            if row['High'] > stop_price:
                                exit_price = stop_price
                                break
                
                if position == 1:
                    pnl = (exit_price - entry_price) * self.contract_value - (self.commission * 2)
                else:
                    pnl = (entry_price - exit_price) * self.contract_value - (self.commission * 2)
                    
                trades.append(pnl)
                
                if len(trades) <= 3:
                     print(f"DEBUG: Date={date}, Pos={position}, Entry={entry_price}, Exit={exit_price}, Stop={stop_price}, PnL={pnl}, H={h}, L={l}")

                
        return trades

def analyze_results(trades, name):
    if not trades:
        print(f"{name:<30} | NO TRADES")
        return { 'PnL': 0, 'PF': 0, 'WR': 0, 'Trades': 0}
        
    trades_np = np.array(trades)
    total_pnl = np.sum(trades_np)
    n_trades = len(trades_np)
    win_rate = np.mean(trades_np > 0)
    
    winners = trades_np[trades_np > 0]
    losers = trades_np[trades_np <= 0]
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(losers) if len(losers) > 0 else 0
    pf = abs((avg_win * len(winners)) / (avg_loss * len(losers))) if len(losers) > 0 else 0
    
    print(f"{name:<30} | PnL: ${total_pnl:<11,.2f} | PF: {pf:<5.2f} | WR: {win_rate:<6.1%} | Trades: {n_trades:<5}")
    return { 'PnL': total_pnl, 'PF': pf, 'WR': win_rate, 'Trades': n_trades}

if __name__ == "__main__":
    data_path = r"c:\Users\User\Documents\AI\Quant_Lab\data\Intra OHLC\A2API-NQ-m5.csv"
    
    try:
        tester = LondonSqueezeBacktester(data_path)
        print("\n" + "="*80)
        
        # Baselines
        analyze_results(tester.run_london_squeeze('NONE', 'NONE'), "London Raw")
        
        # Squeeze Tests
        analyze_results(tester.run_london_squeeze('SQUEEZE', 'NONE'), "London Squeeze NoStop")
        analyze_results(tester.run_london_squeeze('SQUEEZE', 'OPPOSITE'), "London Squeeze + OppStop")
        
        print("="*80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
