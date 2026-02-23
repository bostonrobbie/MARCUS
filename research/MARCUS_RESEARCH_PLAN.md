# MARCUS RESEARCH PLAN: NQ Complementary Strategy Development
## Version 1.0 | Generated 2025-02-16

---

## 1. INPUTS & ASSUMPTIONS

### Known from NQmain code review:
- **Symbol:** NQ futures (CME_MINI:NQ1! or equivalent)
- **Timeframe:** 5-minute bars
- **Session:** 9:30-16:00 ET (Regular Trading Hours)
- **Existing strategy:** 15-minute Opening Range Breakout (9:30-9:45 ET)
- **Entry logic:** Breakout of 9:30-9:45 high/low with EMA(50) filter
- **Filters:** ATR volatility (14-period, 2.5x max), HTF daily SMA(100) trend, RVOL(20) > 1.5
- **Risk:** ATR-based stops (2x SL, 4x TP), trailing stop (2x ATR)
- **Sizing:** 1-3 contracts (dynamic based on trend alignment + RVOL)
- **Hard exit:** 15:45 ET
- **Max trades:** 1 per day
- **Cost assumptions:** $5 slippage + $4.12 commission per round trip

### Assumptions I am making:
- NQmain is PRIMARILY active 9:30-10:30 (ORB window + breakout continuation)
- NQmain performs best in TRENDING days with volume
- NQmain is WEAKEST during low-vol chop and lunch hours
- Overnight/pre-market is NOT covered by NQmain
- Post-2pm session is minimally covered (only trailing stops on morning entries)

---

## 2. NQMAIN COVERAGE SUMMARY & GAPS

### Current Coverage (NQmain):
| Dimension | Coverage | Notes |
|-----------|----------|-------|
| **Time: 9:30-9:45** | STRONG | Core ORB capture window |
| **Time: 9:45-10:30** | MODERATE | Breakout continuation via trailing stop |
| **Time: 10:30-11:30** | WEAK | Only holding existing positions |
| **Time: 11:30-14:00** | NONE | Lunch dead zone |
| **Time: 14:00-15:45** | WEAK | Only exit management |
| **Regime: Trending** | STRONG | ORB excels in trending days |
| **Regime: Mean Reversion** | NONE | ORB gets stopped out on reversions |
| **Regime: Chop/Range** | NONE | ORB gets chopped up |

### Identified Gaps (Where Marcus adds value):
1. **GAP 1 - Post-Lunch Momentum (14:00-15:30):** Late-day directional moves when institutions position for the close
2. **GAP 2 - First Hour Exhaustion Fade (10:15-11:30):** Mean reversion after morning momentum exhausts
3. **GAP 3 - Lunch Range Fade (11:30-13:30):** Mean reversion during low-volume consolidation

---

## 3. PROPOSED STRATEGY FAMILIES

### Family A: TRENDING - "Power Hour Momentum" (14:00-15:30)
**Market behavior hypothesis:** Institutional order flow accelerates in the final 90 minutes. Large funds that need to execute daily allocations concentrate activity here. This creates reliable directional momentum that is distinct from the morning ORB.

- **Targeted window:** 14:00-15:30 ET
- **Entry trigger:** Price breaks above/below the 12:00-14:00 consolidation range with volume confirmation and ADX trend filter
- **Exit logic:** Trailing stop (1.5x ATR) or hard exit at 15:40
- **Risk control:** Fixed 2x ATR stop loss, max 1 trade/day
- **Why it complements NQmain:** Different time window, different catalyst (institutional flow vs opening imbalance)

### Family B: MEAN REVERSION - "First Hour Fade" (10:15-11:30)
**Market behavior hypothesis:** After the opening 45 minutes of volatility, overextended moves frequently revert. The opening move creates an exhaustion point that is statistically mean-reverting. This is the OPPOSITE of what NQmain does.

- **Targeted window:** 10:15-11:30 ET
- **Entry trigger:** Price is extended beyond 1.5x ATR from VWAP AND RSI shows divergence (overbought/oversold)
- **Exit logic:** Reversion to VWAP or time-based exit at 11:30
- **Risk control:** Fixed stop beyond session extreme, max 1 trade/day
- **Why it complements NQmain:** Counter-trend when NQmain is trend-following. Works on days NQmain gets stopped out.

### Family C: CHOP/RANGE - "Lunch Fade" (11:30-13:30)
**Market behavior hypothesis:** Lunch hours have statistically lower volume and tighter ranges. Price tends to oscillate around VWAP during this window. Range-bound strategies that fade moves to session extremes capture this behavior reliably.

- **Targeted window:** 11:30-13:30 ET
- **Entry trigger:** Price touches the upper/lower boundary of the morning range AND RSI reaches 70/30 AND volume is declining
- **Exit logic:** Reversion toward session VWAP or time exit at 13:30
- **Risk control:** Stop beyond the morning high/low, max 1 trade/day
- **Why it complements NQmain:** Profits from the exact conditions that damage ORB strategies.

---

## 4. GAUNTLET VERIFICATION PROTOCOL

### Pass 1 - Initial Backtest (Standard):
| Setting | Value |
|---------|-------|
| Commission | $4.12/round trip |
| Slippage | 1 tick (0.25 pts on NQ) |
| Initial Capital | $100,000 |
| Position Size | 1 contract |
| Date Range | 2011-01-01 to 2025-12-31 |
| Timeframe | 5-minute |

**Pass 1 Minimum Thresholds:**
- Net Profit > $0
- Total Trades > 200
- No single day accounts for >10% of total profit

### Pass 2 - Stressed Backtest (Gauntlet):
| Setting | Value | Why |
|---------|-------|-----|
| Commission | $8.24/round trip (2x) | Accounts for hidden execution costs |
| Slippage | 2 ticks (0.50 pts) | Worse fills in fast markets |
| Initial Capital | $100,000 | Same |
| Position Size | 1 contract | Same |

**Pass 2 Minimum Thresholds (STRICT):**
- Sharpe Ratio >= 1.0 (relaxed from 1.5 to survive stress)
- Max Drawdown <= 25%
- Win Rate >= 35% OR Profit Factor >= 1.5
- Total Trades >= 500 (over 15 years)
- Profit Factor >= 1.3

### Pass 3 - Regime Split Test:
Split data into 3 periods and each must be profitable:
1. **2011-2015** (Post-GFC recovery, low vol trending)
2. **2016-2020** (Mixed, includes COVID crash)
3. **2021-2025** (High vol, rate hikes, AI boom)

**Pass 3 Rule:** Strategy must be net profitable in ALL 3 periods. No cherry-picking.

### Pass 4 - Parameter Sensitivity:
Vary each key parameter +/- 20%. Strategy must remain profitable under ALL variations.
If performance collapses with small changes = OVERFIT = FAIL.

### Pass 5 - Complementarity Check:
- Correlation with NQmain daily returns < 0.3
- Combined portfolio Sharpe > individual Sharpe of either alone
- Different peak trading hours (verify via time-of-day distribution)

---

## 5. SAFETY & BIAS CHECKLIST

- [ ] No lookahead bias: All indicators use confirmed bars only (barstate.isconfirmed)
- [ ] No repainting: No request.security without proper lookahead handling
- [ ] 5-minute timeframe only: No multi-timeframe calculations that leak future data
- [ ] 1 trade/day target: tradedToday flag enforced
- [ ] All positions closed by 15:40 ET: Hard exit enforced
- [ ] Realistic costs: Commission + slippage modeled
- [ ] No overnight holds: Intraday only
- [ ] Parameter count minimal: <8 tunable parameters per strategy
- [ ] No curve-fitting: Variants are logical, not optimized to data

---

## 6. RESULTS TEMPLATE

For each strategy baseline and variant, record:

| Metric | Baseline | V2 | V3 | V4 | V5 |
|--------|----------|----|----|----|----|
| Net Profit ($) | | | | | |
| Profit Factor | | | | | |
| Max Drawdown ($) | | | | | |
| Max Drawdown (%) | | | | | |
| Win Rate (%) | | | | | |
| Total Trades | | | | | |
| Avg Trade ($) | | | | | |
| Avg Bars in Trade | | | | | |
| Sharpe Ratio | | | | | |
| Trades/Day Avg | | | | | |
| Best Month ($) | | | | | |
| Worst Month ($) | | | | | |

**How to interpret:**
- Sharpe > 1.5 = Strong candidate
- PF > 1.5 with WR > 35% = Robust
- Max DD < 15% = Excellent risk control
- Avg ~1 trade/day = On target
- Consistent across all 3 regime periods = Not overfit

---

## 7. FINAL CANDIDATE SELECTION

A strategy graduates to "Final Candidate" when:
1. Passes ALL 5 gauntlet tests
2. Has at least 2 variants that also pass (not just 1 lucky config)
3. Adds coverage to a TIME WINDOW not covered by NQmain
4. Adds coverage to a REGIME not covered by NQmain
5. Combined portfolio Sharpe improves

### Integration Guidance:
Each final candidate will include:
- Enable/disable rules (e.g., "only activate when VIX > 15" or "only during specific session")
- Signal overlap avoidance rules (e.g., "if NQmain has an open position, skip this entry")
- Suggested allocation (e.g., "1 contract, same as NQmain, independent P&L tracking")
