
import pandas as pd
import argparse
import os

def analyze_wfo(file_path, output_path):
    print(f"Analyzing {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Basic stats
    total_segments = len(df)
    profitable_segments = len(df[df['test_return'] > 0])
    win_rate = (profitable_segments / total_segments) * 100 if total_segments > 0 else 0
    avg_return = df['test_return'].mean()
    cum_return = (1 + df['test_return']).prod() - 1
    
    # Analyze parameter stability
    # Simple count of unique params used
    unique_params = df['params'].nunique()
    most_common_params = df['params'].mode()[0] if not df['params'].empty else "N/A"

    report = f"""# WFO Analysis Report

## Summary
- **Total Segments**: {total_segments}
- **Win Rate (Segments)**: {win_rate:.2f}%
- **Average Segment Return**: {avg_return:.4%}
- **Cumulative Stitched Return**: {cum_return:.4%}

## Stability
- **Unique Parameter Sets**: {unique_params}
- **Most Common Parameters**:
  `{most_common_params}`

## Segment Details (First 5)
```
{df.head(5).to_string()}
```

## Segment Details (Last 5)
```
{df.tail(5).to_string()}
```
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    analyze_wfo(args.file, args.output)
