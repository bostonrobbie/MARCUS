import pandas as pd
try:
    df = pd.read_csv(r"c:\Users\User\Documents\AI\Quant_Lab\portfolio_trades.csv")
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error reading CSV: {e}")
