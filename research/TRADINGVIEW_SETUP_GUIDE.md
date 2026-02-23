# TradingView Backtesting Setup Guide for Marcus Strategies

## Symbol & Chart Setup
1. **Symbol:** `CME_MINI:NQ1!` (or `NASDAQ:NQ` if unavailable)
2. **Timeframe:** 5-minute (CRITICAL - do not use any other)
3. **Chart timezone:** Exchange (or America/New_York)
4. **Extended hours:** OFF (we handle session logic internally)

## Strategy Properties (Settings gear icon)
After adding the strategy to your chart, click the gear icon:

### Pass 1 - Standard Test
| Setting | Value |
|---------|-------|
| Initial Capital | 100,000 |
| Order Size | 1 contract (Fixed) |
| Commission | $2.06 per order (=$4.12 round trip) |
| Slippage | 1 tick |
| Recalculate After Order Fills | OFF |
| Recalculate On Every Tick | OFF |

### Pass 2 - Gauntlet Stress Test
| Setting | Value |
|---------|-------|
| Initial Capital | 100,000 |
| Order Size | 1 contract (Fixed) |
| Commission | $4.12 per order (=$8.24 round trip) |
| Slippage | 2 ticks |
| Recalculate After Order Fills | OFF |
| Recalculate On Every Tick | OFF |

## Date Range Testing

### Full Backtest
- Start: 2011-01-01
- End: Current date
- This gives ~15 years of data

### Regime Split Tests (Pass 3)
Run each separately and record results:
1. **Period A:** 2011-01-01 to 2015-12-31 (Post-GFC recovery)
2. **Period B:** 2016-01-01 to 2020-12-31 (Mixed + COVID)
3. **Period C:** 2021-01-01 to Current (Rate hikes + AI)

## How to Run Parameter Sensitivity (Pass 4)
For each strategy, test these variant dimensions by changing ONE parameter at a time:

### Family A (Power Hour Momentum) Variants:
| Variant | Change from Baseline |
|---------|---------------------|
| A-v2 | Range window: 1100-1400 (wider) |
| A-v3 | ATR stop: 2.0x (tighter risk) |
| A-v4 | ADX threshold: 25 (stricter trend) |
| A-v5 | Trailing stop only (remove fixed TP) |

### Family B (First Hour Fade) Variants:
| Variant | Change from Baseline |
|---------|---------------------|
| B-v2 | Extension: 2.0x ATR (stricter trigger) |
| B-v3 | RSI thresholds: 75/25 (more extreme) |
| B-v4 | VWAP target: 0.5 (halfway reversion) |
| B-v5 | No morning range stop, use pure ATR stop |

### Family C (Lunch Fade) Variants:
| Variant | Change from Baseline |
|---------|---------------------|
| C-v2 | RSI thresholds: 70/30 (tighter) |
| C-v3 | BB Width: 0.015 (stricter squeeze) |
| C-v4 | Window: 1200-1400 (shifted later) |
| C-v5 | TP: 1.0 (full move to VWAP) |

## What to Record

For EACH test, fill in:

```
Strategy: _______________
Variant: ________________
Date Range: _____________
Commission: Standard / Gauntlet

Net Profit ($):      _______
Profit Factor:       _______
Max Drawdown ($):    _______
Max Drawdown (%):    _______
Win Rate (%):        _______
Total Trades:        _______
Avg Trade ($):       _______
Avg Bars in Trade:   _______
Sharpe Ratio:        _______
Largest Win ($):     _______
Largest Loss ($):    _______
```

## Pass/Fail Criteria

### Standard (Pass 1):
- Net Profit > $0
- Trades > 200
- No obvious equity curve anomalies

### Gauntlet (Pass 2):
- Sharpe >= 1.0
- Max DD <= 25%
- Win Rate >= 35% OR PF >= 1.5
- Trades >= 500 (over 15yr)
- PF >= 1.3

### Regime (Pass 3):
- Net profitable in ALL 3 periods

### Sensitivity (Pass 4):
- Profitable under ALL variants
- No variant shows >50% profit drop

### Complementarity (Pass 5):
- Time-of-day distribution doesn't overlap NQmain peak hours
- Daily return correlation with NQmain < 0.3

## Integration Rules

When a strategy passes all 5 gates:
1. Run it alongside NQmain in paper mode for 2 weeks
2. Track combined portfolio Sharpe
3. Monitor for signal overlap days
4. If combined Sharpe > individual = DEPLOY
