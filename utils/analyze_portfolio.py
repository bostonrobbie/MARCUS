import pandas as pd
import numpy as np
from datetime import datetime

def analyze_portfolio(csv_path):
    try:
        # Load CSV (skip first line usually containing header metadata in TV exports if needed, 
        # but pandas often handles it. If 'Trade #' is in first row, it's fine)
        df = pd.read_csv(csv_path)
        
        # Standardize columns
        # TV Export cols: "Trade #", "Type", "Signal", "Date/Time", "Price", "Contracts", "Profit USD", "Profit %", "Cum. Profit USD", "Cum. Profit %", "Run-up USD", "Run-up %", "Drawdown USD", "Drawdown %"
        # We need "Date/Time" and "Profit USD" mainly.
        
        # Column Mapping based on inspection
        date_col = 'Date and time'
        if date_col not in df.columns:
             # Try fallback
             date_col = next((c for c in df.columns if 'Date' in c), None)

        if not date_col:
             raise ValueError("Could not find Date column")
             
        df['Date'] = pd.to_datetime(df[date_col])
        df['Day'] = df['Date'].dt.date
        
        # Filter for CLOSED trades only (Exit rows contain the PnL)
        # 'Type' column contains "Exit long" or "Exit short"
        if 'Type' in df.columns:
            df_trades = df[df['Type'].astype(str).str.contains('Exit', case=False)].copy()
        else:
            # Fallback for other formats
            df_trades = df.copy()

        # Profit Column
        # Looking at output, we have 'Price USD', 'Position size...', 'Cumulative P&L USD'
        # We likely have 'Profit USD' or 'Profit' but it was truncated in print.
        # Let's search specifically for Profit or P&L *per trade*.
        # If not, we can use delta of Cumulative P&L.
        
        profit_col = next((c for c in df.columns if 'Profit' in c and ('USD' in c or '$' in c) and 'Cum' not in c), None)
        if not profit_col:
             # Try P&L
             profit_col = next((c for c in df.columns if 'P&L' in c and ('USD' in c or '$' in c) and 'Cum' not in c), None)
        
        if not profit_col:
             print("Warning: Could not find explicit Per-Trade Profit column. Calculating from Cumulative P&L...")
             if 'Cumulative P&L USD' in df.columns:
                 # Sort by Trade # first? 
                 # Actually, if we use Exit rows, Cumulative P&L - Prev Cumulative P&L = Trade P&L
                 df_trades = df_trades.sort_values('Date')
                 df_trades['Profit USD'] = df_trades['Cumulative P&L USD'].diff().fillna(df_trades['Cumulative P&L USD'].iloc[0])
                 profit_col = 'Profit USD'
             else:
                 raise ValueError("Could not find Profit or Cumulative P&L column")
        
        # Metrics
        total_pnl = df_trades[profit_col].sum()
        total_trades = len(df_trades)
        wins = df_trades[df_trades[profit_col] > 0]
        losses = df_trades[df_trades[profit_col] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_win = wins[profit_col].mean() if not wins.empty else 0
        avg_loss = abs(losses[profit_col].mean()) if not losses.empty else 0
        pf = (avg_win * len(wins)) / (avg_loss * len(losses)) if len(losses) > 0 else 0
        
        # Max Drawdown (from Equity curve)
        df_trades = df_trades.sort_values('Date')
        # Use Cumulative P&L column directly if available for accurate equity curve
        if 'Cumulative P&L USD' in df_trades.columns:
            equity = df_trades['Cumulative P&L USD']
        else:
            equity = df_trades[profit_col].cumsum()
            
        peak = equity.cummax()
        drawdown = equity - peak
        max_dd = drawdown.min()
        
        # Consistency / Gaps
        all_trading_days = pd.date_range(start=df['Day'].min(), end=df['Day'].max(), freq='B') # Business Days
        traded_days = set(df['Day'].unique())
        
        # Holidays (approximate, or just list gaps > 3 days)
        # We will list gaps > 2 business days to avoid noise
        
        gaps = []
        streak = 0
        last_day = None
        
        for day in all_trading_days:
            d = day.date()
            if d not in traded_days:
                streak += 1
            else:
                if streak >= 3: # 3+ Business days without a trade
                   gaps.append((last_day, d, streak))
                streak = 0
                last_day = d
                
        # Grading
        grade = "F"
        if pf > 2.0: grade = "A+"
        elif pf > 1.5: grade = "A"
        elif pf > 1.3: grade = "B"
        elif pf > 1.1: grade = "C"
        else: grade = "D"
        
        if max_dd < -50000: grade += " (Risk Warning: High DD)"
        
        print("\n" + "="*50)
        print(f"PORTFOLIO REPORT CARD")
        print("="*50)
        print(f"Overall Grade:      {grade}")
        print(f"Total PnL:          ${total_pnl:,.2f}")
        print(f"Profit Factor:      {pf:.2f}")
        print(f"Win Rate:           {win_rate:.1%}")
        print(f"Total Trades:       {total_trades}")
        print(f"Max Drawdown:       ${max_dd:,.2f}")
        print(f"Avg Trade:          ${total_pnl/total_trades:,.2f}")
        print("-" * 50)
        
        print(f"\n[Consistency Analysis]")
        print(f"Total Business Days: {len(all_trading_days)}")
        print(f"Days Traded:         {len(traded_days)}")
        print(f"Utilization:         {len(traded_days)/len(all_trading_days):.1%}")
        
        if gaps:
            print(f"\n[Significant Gaps (>3 Business Days without Trade)]")
            count = 0 
            for start, end, days in gaps:
                print(f"- {start} to {end}: {days} days gap")
                count += 1
                if count > 10:
                    print(f"... and {len(gaps)-10} more.")
                    break
        else:
            print("\nExcellent Consistency! No trade gaps > 3 business days found.")
            
    except Exception as e:
        print(f"Error analyzing CSV: {e}")

if __name__ == "__main__":
    analyze_portfolio(r"c:\Users\User\Documents\AI\Quant_Lab\portfolio_trades.csv")
