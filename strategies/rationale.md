# Strategy Rationale: 15-Minute Opening Range Breakout (ORB)

## 1. Market Mechanics (The "Why")
The Opening Range Breakout (ORB) exploits the information imbalance resolution that occurs at the US Equity Open (09:30 ET).
-   **Overnight Accumulation**: Orders accumulate overnight from global markets.
-   **Price Discovery**: The first 15-30 minutes represent the market "agreeing" on a valuation for the day.
-   **Volatility Expansion**: Once the initial range is broken, it signals that one side (Buyers or Sellers) has exhausted the liquidity of the other, leading to a directional move.

## 2. Quantitative Evidence (The "What")
Historical analysis of NQ/ES suggests:
1.  **Trend Alignment**: Breakouts that align with the daily trend (defined by EMA) have higher win rates (>55%) vs mean-reverting breakouts.
2.  **Volatility Filter**: Days with compressed opening ranges (Low Volatility) often lead to "False Breakouts". We use Minimum ATR or Range Size limits.
3.  **Momentum Filter**: ADX (Average Directional Index) > 20 filters out "Choppy" days where the market is seeking equilibrium rather than trending.

## 3. Implementation Logic (The "How")
-   **Timeframe**: 5-Minute Bars (Granular execution).
-   **Range Definition**: High/Low of 09:30 - 09:45 (First 3 bars of 5m).
-   **Entry Trigger**: Candle Close > Range High (Long) or Candle Close < Range Low (Short).
-   **Risk Management**:
    -   **Stop Loss**: 1.0 - 2.5x ATR (Adapts to daily volatility).
    -   **Take Profit**: 2.0 - 4.0x ATR (Targets large excursions).
    -   **EOD Exit**: 15:45 ET (Avoid overnight gap risk).

## 4. Why 5m Execution?
We define the range using 15 minutes of time (3 bars), but we execute on the 5-minute timeframe.
-   **Advantage**: If a breakout occurs at 09:50, we don't wait for 10:00 (15m close) to enter. We enter at 09:50 (5m close). This reduces slippage and gets us into the move 10 minutes earlier, improving R:R.
