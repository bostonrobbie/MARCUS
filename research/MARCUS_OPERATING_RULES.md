# MARCUS OPERATING RULES v2.0
## Autonomous Research Agent - Output Quality Standards

---

## 1. CYCLE OUTPUT REQUIREMENTS

Every research cycle MUST produce the following outputs:

### 1.1 Cycle Summary Log
Each cycle log entry must include:
- Cycle number
- Ideas generated (count + archetype breakdown)
- Ideas tested vs skipped (with skip reasons)
- Stage results: S1 passed, S2 passed, S3/S4/S5 passed
- Best Sharpe in this cycle (strategy name + value)
- Duration in seconds
- Error count with details if > 0

### 1.2 Rejection Reason Codes
Every rejected strategy MUST log a structured reason using these codes:

| Code | Meaning | Stage |
|------|---------|-------|
| `S1_PROFIT` | Net profit <= $0 | Stage 1 |
| `S1_TRADES` | Total trades < min_trades | Stage 1 |
| `S2_SHARPE` | Sharpe ratio < threshold | Stage 2 |
| `S2_DRAWDOWN` | Max drawdown > 25% | Stage 2 |
| `S2_PF` | Profit factor < 1.3 | Stage 2 |
| `S2_WINRATE` | Win rate < 35% AND PF < 1.5 | Stage 2 |
| `S2_TRADES` | Trades < 500 over test period | Stage 2 |
| `S3_REGIME` | Not profitable in all 3 periods | Stage 3 |
| `S4_SENSITIVITY` | >50% profit drop on param variation | Stage 4 |
| `S5_CORRELATION` | Correlation with NQmain > 0.3 | Stage 5 |
| `GRAVEYARD` | Hash already tested and killed | Pre-filter |
| `EXHAUSTED` | Archetype has >200 tests, 0 S2 passes | Pre-filter |
| `TIMEOUT` | Backtest exceeded 120s timeout | Any |
| `ERROR` | Unexpected exception | Any |

### 1.3 Winner Artifacts
When a strategy passes S5, save:
- Full metrics JSON (all stages)
- Equity curve CSV (date, equity, drawdown)
- Parameter snapshot
- Pine Script source (if applicable)
- Complementarity correlation matrix

---

## 2. ARCHETYPE DIVERSITY RULES

### 2.1 Mandatory Diversity
- NO archetype may exceed 60% of any single batch
- Round-robin balancing across all non-exhausted archetypes
- Log archetype distribution every cycle

### 2.2 Exhaustion Protocol
- An archetype is EXHAUSTED when: tested > 200 AND S2 passes = 0
- Exhausted archetypes are SKIPPED in fallback generation
- Log CRITICAL if ALL archetypes are exhausted
- Review thresholds before adding more parameter combinations

### 2.3 Required Archetypes
The engine must always include these 8 archetype families:

| Archetype | Time Window | Regime | Strategy Type |
|-----------|-------------|--------|---------------|
| `orb_breakout` | 09:30-10:30 | Trending | Range breakout |
| `ma_crossover` | All day | Trending | Trend following |
| `eod_momentum` | 13:30-15:30 | Trending | Momentum |
| `lunch_hour_breakout` | 11:00-13:30 | Range | Range breakout |
| `gap_fill_fade` | 09:30-11:00 | Mean Rev | Counter-trend |
| `power_hour_momentum` | 14:00-15:30 | Trending | Institutional flow |
| `first_hour_fade` | 10:15-11:30 | Mean Rev | Exhaustion fade |
| `lunch_range_fade` | 11:30-13:30 | Chop | Range mean reversion |

---

## 3. REPORTING RULES

### 3.1 Session Summaries
- Auto-generate every 50 cycles
- Auto-generate on daemon shutdown
- Include: pipeline funnel, near misses table, archetype performance, recommendations
- Save to `Marcus_Research/reports/session_YYYYMMDD_HHMMSS.md`

### 3.2 Dashboard
- Auto-rebuild every 15 minutes (daemon schedule)
- Auto-refresh every 30 seconds (browser meta tag)
- Must display: pipeline funnel, near misses, archetype stats, LLM status, session timer

### 3.3 Near Miss Tracking
- A "near miss" = strategy that passed S1 but failed at S2
- Track top 10 near misses by S1 Sharpe ratio
- Display in dashboard and session summaries
- Near misses inform which parameter regions to explore further

---

## 4. LLM INTEGRATION RULES

### 4.1 LLM Status Management
- Check LLM health on dashboard refresh (ping /api/version)
- Display LLM status (Online/Offline) in dashboard header
- After 3 consecutive LLM failures, auto-disable LLM for session
- Log CRITICAL when LLM auto-disabled
- Reset failure counter on any successful generation

### 4.2 Fallback Behavior
When LLM is offline or disabled:
- Use deterministic fallback grid (8 archetypes, ~5000+ combos)
- Apply round-robin balancing across archetypes
- Apply exhaustion filtering
- Log that fallback mode is active

---

## 5. SAFETY & INTEGRITY

### 5.1 No Lookahead
- All entries use `barstate.isconfirmed` (Pine) or confirmed bars only (Python)
- No `request.security()` calls that leak future data
- Single timeframe only (5-minute)
- All indicators use standard lookback

### 5.2 Cost Modeling
- Stage 1: Standard costs ($4.12 commission + 1 tick slippage)
- Stage 2: Gauntlet costs (2x commission + 2x slippage)
- All costs modeled as percentage of futures notional value

### 5.3 Position Limits
- Max 1 trade per day per strategy
- Hard exit by 15:45 ET (no overnight holds)
- Position size: 1 contract (fixed for testing)

### 5.4 Instance Safety
- PID file guard prevents duplicate daemon instances
- Check if existing PID is still alive before claiming lock
- Clean up PID file on graceful shutdown

---

## 6. QUALITY GATES

### Stage 1 (Basic Backtest)
- Net Profit > $0
- Total Trades > 200
- No single day > 10% of total profit

### Stage 2 (Gauntlet Stress)
- Sharpe Ratio >= 1.0
- Max Drawdown <= 25%
- Win Rate >= 35% OR Profit Factor >= 1.5
- Total Trades >= 500
- Profit Factor >= 1.3

### Stage 3 (Regime Split)
- Net profitable in ALL 3 periods:
  - 2011-2015
  - 2016-2020
  - 2021-Present

### Stage 4 (Parameter Sensitivity)
- ALL +/-20% parameter variations remain profitable
- No variant shows >50% profit drop from baseline

### Stage 5 (Complementarity)
- Daily return correlation with NQmain < 0.3
- Combined portfolio Sharpe > individual Sharpe
- Different peak trading hours from NQmain

---

## 7. LOGGING STANDARDS

### Log Levels
- **CRITICAL**: All archetypes exhausted, LLM auto-disabled, data corruption
- **ERROR**: Backtest failure, import error, DB error
- **WARNING**: Single archetype exhausted, low diversity, stale heartbeat
- **INFO**: Cycle summary, stage results, data load, dashboard rebuild
- **DEBUG**: Individual strategy skip reasons, graveyard hits

### Log Suppression
- Data load message: log once, suppress on subsequent calls (shared handler)
- Heartbeat: every 5 minutes only
- Dashboard rebuild: log path only, not full stats

---

## 8. FILE STORAGE

### Directory Structure
```
Marcus_Research/
  |-- marcus_registry.db          (SQLite: all strategy data)
  |-- dashboard/
  |   |-- marcus_live.html        (Auto-refreshing dashboard)
  |-- logs/
  |   |-- marcus_daemon.log       (Rotating, 10MB x 5 files)
  |   |-- marcus.pid              (Instance guard)
  |-- reports/
  |   |-- session_*.md            (Session summaries)
  |-- strategies/
  |   |-- *.pine                  (Pine Script strategies)
  |-- winners/
  |   |-- *.json                  (S5 winner artifacts)
```

### Database Tables
- `backtest_runs`: All backtest results with metrics
- `strategy_lifecycle`: State machine tracking (CANDIDATE -> DEPLOYED)
- `strategy_graveyard`: Hash-based deduplication of killed strategies
- `winning_strategies`: S5 graduates with full metrics
- `cycle_results`: Per-cycle summary statistics

---

*MARCUS Operating Rules v2.0 - Updated 2026-02-17*
*These rules are enforced by the research engine, daemon, and dashboard code.*
