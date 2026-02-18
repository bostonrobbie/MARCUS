import pandas as pd
import numpy as np
import os

class DriftOptimizer:
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
            sess_end = pd.Timestamp(curr_date).tz_localize('US/Eastern').replace(hour=9, minute=30)
            
            data = self.df.loc[sess_start:sess_end]
            if len(data) < 10: continue
            
            yield {
                'date': curr_date,
                'data': data,
                'start_time': sess_start
            }

    def run_drift_flexible(self, start_offset_hours=6.0, end_offset_hours=15.5, use_sma=False, sma_period=200, use_day_filter=None):
        """
        Flexible Drift Session. 
        Start Offset: 0.0 = 18:00 Previous Day. 6.0 = 00:00 Current Day.
        End Offset: 15.5 = 09:30 Current Day.
        """
        label = f"Drift [{start_offset_hours}h-{end_offset_hours}h] [SMA={use_sma} Skip={use_day_filter}]"
        print(f"Running {label}...")
        
        if use_sma and 'SMA' not in self.df.columns:
            self.df['SMA'] = self.df['Close'].rolling(sma_period).mean()
        
        trades = []
        
        for sess in self.get_overnight_sessions():
            # Day Filter
            if use_day_filter:
                wd = sess['date'].weekday()
                if wd in use_day_filter: continue
                
            start_drift = sess['start_time'] + pd.Timedelta(hours=start_offset_hours)
            end_drift = sess['start_time'] + pd.Timedelta(hours=end_offset_hours)
            
            seg = sess['data'].loc[start_drift:end_drift]
            if len(seg) < 2: continue
            
            entry_price = seg.iloc[0]['Open']
            
            if use_sma:
                entry_time = seg.index[0]
                if entry_time not in self.df.index: continue
                sma_val = self.df.at[entry_time, 'SMA']
                if pd.isna(sma_val): continue
                if entry_price < sma_val: continue
            
            exit_price = seg.iloc[-1]['Close']
            pnl = (exit_price - entry_price) * self.contract_value - (self.commission * 2)
            trades.append(pnl)
            
        return trades

def analyze_results(trades, name):
    output_line = ""
    if not trades:
        output_line = f"{name:<40} | NO TRADES\n"
    else:
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
        
        output_line = f"{name:<40} | PnL: ${total_pnl:<11,.2f} | PF: {pf:<5.2f} | WR: {win_rate:<6.1%} | Trades: {n_trades:<5} | DD: ${max_dd:,.2f}\n"

    print(output_line.strip())
    with open("drift_results.txt", "a") as f:
        f.write(output_line)

if __name__ == "__main__":
    data_path = r"c:\Users\User\Documents\AI\Quant_Lab\data\Intra OHLC\A2API-NQ-m5.csv"
    
    with open("drift_results.txt", "a") as f:
        f.write("\n" + "="*110 + "\n")
        f.write(" OPTIMIZATION: Existing Strat (18:00-09:30) vs Deep Night (00:00-09:30) \n")
        f.write("="*110 + "\n")
    
    try:
        opt = DriftOptimizer(data_path)
        
        # 1. Full Existing (18:00 - 09:30)
        # Offset 0.0 to 15.5
        analyze_results(opt.run_drift_flexible(0.0, 15.5, False), "Existing (18:00-09:30) Raw")
        analyze_results(opt.run_drift_flexible(0.0, 15.5, True, use_day_filter=[3, 4]), "Existing + SMA + Skip Thu/Fri")
        
        # 2. Deep Night (00:00 - 06:00) Best Candidate
        analyze_results(opt.run_drift_flexible(6.0, 12.0, False), "Deep Night (00-06) Raw")
        analyze_results(opt.run_drift_flexible(6.0, 12.0, True, use_day_filter=[3, 4]), "Deep Night + SMA + Skip Thu/Fri")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
