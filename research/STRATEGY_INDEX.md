# MARCUS STRATEGY INDEX
## Complete Pine Script Library for NQ Complementary Strategy Research

---

## Quick Reference

| ID | Strategy | File | Regime | Time Window | Key Change |
|----|----------|------|--------|-------------|------------|
| **A-v1** | Power Hour Momentum | `A_PowerHour_Momentum.pine` | TRENDING | 14:00-15:30 | **BASELINE** |
| A-v2 | Power Hour (Wider Range) | `A_PowerHour_v2_WiderRange.pine` | TRENDING | 14:00-15:30 | Range: 11:00-14:00 |
| A-v3 | Power Hour (Tighter Stop) | `A_PowerHour_v3_TighterStop.pine` | TRENDING | 14:00-15:30 | SL: 1.0x, TP: 2.0x, TS: 1.0x |
| A-v4 | Power Hour (Stricter ADX) | `A_PowerHour_v4_StricterADX.pine` | TRENDING | 14:00-15:30 | ADX threshold: 25 |
| A-v5 | Power Hour (Trailing Only) | `A_PowerHour_v5_TrailingOnly.pine` | TRENDING | 14:00-15:30 | No fixed TP |
| **B-v1** | First Hour Fade | `B_FirstHour_Fade.pine` | MEAN REV | 10:15-11:30 | **BASELINE** |
| B-v2 | First Hour (Stricter Ext) | `B_FirstHour_v2_StricterExtension.pine` | MEAN REV | 10:15-11:30 | Extension: 2.0x ATR |
| B-v3 | First Hour (Extreme RSI) | `B_FirstHour_v3_ExtremeRSI.pine` | MEAN REV | 10:15-11:30 | RSI: 75/25 |
| B-v4 | First Hour (Halfway VWAP) | `B_FirstHour_v4_HalfwayVWAP.pine` | MEAN REV | 10:15-11:30 | TP: 0.5 of VWAP dist |
| B-v5 | First Hour (Pure ATR Stop) | `B_FirstHour_v5_PureATRStop.pine` | MEAN REV | 10:15-11:30 | No morning range stop |
| **C-v1** | Lunch Range Fade | `C_Lunch_Fade.pine` | CHOP | 11:30-13:30 | **BASELINE** |
| C-v2 | Lunch Fade (Tighter RSI) | `C_Lunch_v2_TighterRSI.pine` | CHOP | 11:30-13:30 | RSI: 70/30 |
| C-v3 | Lunch Fade (Strict Squeeze) | `C_Lunch_v3_StricterSqueeze.pine` | CHOP | 11:30-13:30 | BB Width: 0.015 |
| C-v4 | Lunch Fade (Shifted Later) | `C_Lunch_v4_ShiftedLater.pine` | CHOP | 12:00-14:00 | Window shifted 30min |
| C-v5 | Lunch Fade (Full VWAP) | `C_Lunch_v5_FullVWAP.pine` | CHOP | 11:30-13:30 | TP: 1.0 (full VWAP) |

---

## Time Coverage Map (vs NQmain)

```
Hour    | 9:30  10:00  10:30  11:00  11:30  12:00  12:30  13:00  13:30  14:00  14:30  15:00  15:30  15:45
--------|-----------------------------------------------------------------------------------------
NQmain  | ████████████░░░░░░░░                                                              ░░░░
Fam B   |        ░░░░░░████████████
Fam C   |                          ████████████████████████
Fam A   |                                                                  ██████████████████
--------|-----------------------------------------------------------------------------------------
Legend  | ████ = Active entry window    ░░░░ = Position management only
```

**Zero overlap between entry windows. Maximum complementarity.**

---

## Testing Protocol (Order of Operations)

### Step 1: Baseline Validation (Pass 1)
Test each baseline (v1) with Standard settings:
1. `A_PowerHour_Momentum.pine` - Standard commission/slippage
2. `B_FirstHour_Fade.pine` - Standard commission/slippage
3. `C_Lunch_Fade.pine` - Standard commission/slippage

**Must pass:** Net Profit > $0, Trades > 200

### Step 2: Gauntlet Stress (Pass 2)
Re-test each passing baseline with Gauntlet settings (2x commission, 2x slippage):

**Must pass:** Sharpe >= 1.0, Max DD <= 25%, WR >= 35% OR PF >= 1.5, Trades >= 500, PF >= 1.3

### Step 3: Regime Split (Pass 3)
For each gauntlet survivor, run 3 separate backtests:
- 2011-01-01 to 2015-12-31
- 2016-01-01 to 2020-12-31
- 2021-01-01 to Current

**Must pass:** Net profitable in ALL 3 periods

### Step 4: Parameter Sensitivity (Pass 4)
Run all 4 variants for each surviving family:

**Must pass:** ALL variants profitable, no variant shows >50% profit drop from baseline

### Step 5: Complementarity (Pass 5)
For final candidates:
- Verify time-of-day distribution doesn't overlap NQmain
- Check daily return correlation with NQmain < 0.3
- Combined portfolio Sharpe > individual

---

## Safety Checklist (Every Script Verified)

- [x] All entries use `barstate.isconfirmed` - no lookahead
- [x] No `request.security()` calls - no multi-TF data leaks
- [x] 5-minute timeframe only - single timeframe throughout
- [x] `tradedToday` flag enforced - max 1 trade/day
- [x] Hard exit time enforced - no overnight holds
- [x] Session-based time windows - timezone-aware (America/New_York)
- [x] `calc_on_every_tick=false` - no tick-level repainting
- [x] `calc_on_order_fills=false` - no fill-triggered recalc
- [x] All indicators use standard lookback - no future data
- [x] Parameter count < 8 tunable per strategy

---

## Variant Logic Summary

### Family A Variants (What Each Tests)
| Variant | Hypothesis Being Tested |
|---------|------------------------|
| A-v2 | Does a wider consolidation range (3hrs vs 2hrs) produce better S/R? |
| A-v3 | Does tighter risk (1x ATR) improve Sharpe or get stopped out too much? |
| A-v4 | Is trend selectivity (ADX 25) the key, or does frequency (ADX 20) matter more? |
| A-v5 | Do winners need to run (trail only) or should we take fixed profits? |

### Family B Variants (What Each Tests)
| Variant | Hypothesis Being Tested |
|---------|------------------------|
| B-v2 | Does requiring 2x ATR extension (vs 1.5x) improve reversion probability? |
| B-v3 | Does extreme RSI (75/25) select better fade setups? |
| B-v4 | Should we target 50% reversion to VWAP or is 30% the sweet spot? |
| B-v5 | Is the morning range stop better than a generic ATR stop for fades? |

### Family C Variants (What Each Tests)
| Variant | Hypothesis Being Tested |
|---------|------------------------|
| C-v2 | Are relaxed RSI thresholds (65/35) necessary for chop, or does 70/30 work? |
| C-v3 | Is the BB squeeze filter the secret ingredient? Stricter = more selective. |
| C-v4 | Does the true lunch chop happen 11:30-13:30 or 12:00-14:00? |
| C-v5 | Should we target full VWAP reversion or take 75% and run? |

---

## File Locations

```
C:\Users\User\Desktop\Zero_Human_HQ\Quant_Lab\Marcus_Research\
  |-- MARCUS_RESEARCH_PLAN.md          (Full research methodology)
  |-- TRADINGVIEW_SETUP_GUIDE.md       (TV settings for each pass)
  |-- STRATEGY_INDEX.md                (This file)
  |-- strategies/
      |-- A_PowerHour_Momentum.pine    (Family A baseline)
      |-- A_PowerHour_v2_WiderRange.pine
      |-- A_PowerHour_v3_TighterStop.pine
      |-- A_PowerHour_v4_StricterADX.pine
      |-- A_PowerHour_v5_TrailingOnly.pine
      |-- B_FirstHour_Fade.pine        (Family B baseline)
      |-- B_FirstHour_v2_StricterExtension.pine
      |-- B_FirstHour_v3_ExtremeRSI.pine
      |-- B_FirstHour_v4_HalfwayVWAP.pine
      |-- B_FirstHour_v5_PureATRStop.pine
      |-- C_Lunch_Fade.pine            (Family C baseline)
      |-- C_Lunch_v2_TighterRSI.pine
      |-- C_Lunch_v3_StricterSqueeze.pine
      |-- C_Lunch_v4_ShiftedLater.pine
      |-- C_Lunch_v5_FullVWAP.pine
```

**Total: 15 Pine Scripts (3 baselines + 12 variants)**
**Total test runs needed: 15 strategies x 2 passes (Standard + Gauntlet) = 30 minimum**
**Plus regime splits: up to 9 more runs per surviving baseline = up to 27 additional**
