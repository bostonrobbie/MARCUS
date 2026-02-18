#!/usr/bin/env python3
"""Marcus SQLite query helper for the OpenClaw dashboard serve-web.js.

Called via: python marcus-query.py <db_path> [command] [args...]
Outputs JSON to stdout.

Commands:
  (default)       Full dashboard data (cycles, pipeline, leaderboard, etc.)
  winner <id>     Single winner detail with equity curve
  config          Current MarcusConfig values
  history <n>     Last N cycle_log entries
  lifecycle       All non-deleted lifecycle entries
"""

import json
import sqlite3
import sys
import os


def safe_query(cur, sql, params=None, default=None):
    """Run a query and return results, ignoring missing tables."""
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return cur.fetchall()
    except sqlite3.OperationalError:
        return default if default is not None else []


def get_dashboard_data(db_path):
    """Full dashboard payload."""
    result = {
        "cycles": [],
        "pipeline": {},
        "pipeline_cumulative": {},
        "leaderboard": [],
        "winners": [],
        "graveyard": [],
        "stats": {},
        "events": [],
        "active_lifecycle": [],
        "in_progress": [],
        "archived": [],
    }

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # --- Cycle History (last 30) ---
        rows = safe_query(cur, """
            SELECT cycle_num, started_at, finished_at, duration_seconds,
                   ideas_generated, backtests_run,
                   stage1_passed, stage2_passed, stage3_passed,
                   stage4_passed, stage5_passed,
                   rejected, errors, best_sharpe, best_strategy_name, gpu_used
            FROM cycle_log ORDER BY id DESC LIMIT 30
        """)
        result["cycles"] = [dict(r) for r in rows]

        # --- Pipeline Counts (current lifecycle snapshot) ---
        pipeline_rows = safe_query(cur, """
            SELECT current_stage, COUNT(*) as cnt
            FROM strategy_lifecycle
            GROUP BY current_stage
        """)
        result["pipeline"] = {r["current_stage"]: r["cnt"] for r in pipeline_rows}

        # --- Pipeline Cumulative (from cycle_log sums - shows total flow) ---
        cum_rows = safe_query(cur, """
            SELECT
                COALESCE(SUM(ideas_generated), 0) as CANDIDATE,
                COALESCE(SUM(backtests_run), 0) as TESTING,
                COALESCE(SUM(stage1_passed), 0) as STAGE1_PASS,
                COALESCE(SUM(stage2_passed), 0) as STAGE2_PASS,
                COALESCE(SUM(stage3_passed), 0) as STAGE3_PASS,
                COALESCE(SUM(stage4_passed), 0) as STAGE4_PASS,
                COALESCE(SUM(stage5_passed), 0) as STAGE5_PASS
            FROM cycle_log
        """)
        if cum_rows:
            result["pipeline_cumulative"] = dict(cum_rows[0])
            # Add deployed count from winning_strategies
            deployed_rows = safe_query(cur, "SELECT COUNT(*) as cnt FROM winning_strategies WHERE is_active = 1")
            if deployed_rows:
                result["pipeline_cumulative"]["DEPLOYED"] = dict(deployed_rows[0])["cnt"]

        # --- Winners (full list from winning_strategies) ---
        winner_rows = safe_query(cur, """
            SELECT w.id, w.strategy_name, w.archetype, w.symbol, w.interval,
                   w.sharpe_ratio, w.total_return, w.net_profit,
                   w.max_drawdown, w.max_drawdown_pct, w.win_rate, w.profit_factor,
                   w.total_trades, w.win_trades, w.loss_trades, w.avg_trade_pnl,
                   w.params_json, w.source_code, w.quality_score,
                   w.quality_notes, w.monte_carlo_var95, w.permutation_pvalue,
                   w.regime_analysis_json, w.data_range_start, w.data_range_end,
                   w.notes, w.tags, w.is_active, w.priority, w.timestamp
            FROM winning_strategies w
            ORDER BY w.sharpe_ratio DESC
        """)
        winners = []
        for r in winner_rows:
            w = dict(r)
            # Check for equity curve
            eq_rows = safe_query(cur, """
                SELECT data_json FROM equity_curves
                WHERE strategy_id = ?
                ORDER BY id DESC LIMIT 1
            """, (w["id"],))
            w["has_equity_curve"] = len(eq_rows) > 0
            winners.append(w)
        result["winners"] = winners

        # --- Leaderboard (top 15 - winners first, then best backtest_runs) ---
        # Use winning_strategies with correct column name (timestamp, not created_at)
        lb_rows = safe_query(cur, """
            SELECT strategy_name, sharpe_ratio, profit_factor, max_drawdown_pct,
                   total_trades, net_profit, win_rate, timestamp as created_at,
                   'winner' as source
            FROM winning_strategies
            WHERE is_active = 1
            ORDER BY sharpe_ratio DESC
            LIMIT 15
        """)
        leaderboard = [dict(r) for r in lb_rows]

        # If fewer than 15 winners, pad with top backtest_runs
        if len(leaderboard) < 15:
            winner_names = {r["strategy_name"] for r in leaderboard}
            remaining = 15 - len(leaderboard)
            bt_rows = safe_query(cur, f"""
                SELECT strategy_name, sharpe_ratio, profit_factor, max_drawdown_pct,
                       total_trades, net_profit, win_rate, timestamp as created_at,
                       'backtest' as source
                FROM backtest_runs
                WHERE sharpe_ratio > 0 AND total_trades >= 200 AND net_profit > 0
                ORDER BY sharpe_ratio DESC
                LIMIT {remaining + 20}
            """)
            for r in bt_rows:
                rd = dict(r)
                if rd["strategy_name"] not in winner_names:
                    leaderboard.append(rd)
                    winner_names.add(rd["strategy_name"])
                    if len(leaderboard) >= 15:
                        break

        result["leaderboard"] = leaderboard

        # --- Graveyard / Archived (last 30) ---
        gv_rows = safe_query(cur, """
            SELECT strategy_name, killed_at_stage, reason, best_sharpe,
                   total_trades, created_at
            FROM strategy_graveyard
            ORDER BY id DESC LIMIT 30
        """)
        result["graveyard"] = [dict(r) for r in gv_rows]

        # --- In Progress (currently testing strategies) ---
        prog_rows = safe_query(cur, """
            SELECT strategy_name, current_stage, degradation_strikes,
                   s1_passed_at, s2_passed_at, s3_passed_at, s4_passed_at, s5_passed_at,
                   created_at, updated_at
            FROM strategy_lifecycle
            WHERE current_stage NOT IN ('DELETED', 'ARCHIVED', 'REJECTED')
            ORDER BY updated_at DESC
            LIMIT 30
        """)
        result["in_progress"] = [dict(r) for r in prog_rows]

        # --- Archived/Rejected ---
        arch_rows = safe_query(cur, """
            SELECT strategy_name, current_stage, degradation_strikes,
                   created_at, updated_at, notes
            FROM strategy_lifecycle
            WHERE current_stage IN ('ARCHIVED', 'REJECTED')
            ORDER BY updated_at DESC
            LIMIT 30
        """)
        result["archived"] = [dict(r) for r in arch_rows]

        # --- Aggregate Stats ---
        stat_rows = safe_query(cur, """
            SELECT
                COUNT(*) as total_cycles,
                COALESCE(SUM(ideas_generated), 0) as total_ideas,
                COALESCE(SUM(backtests_run), 0) as total_backtests,
                COALESCE(SUM(stage1_passed), 0) as total_s1,
                COALESCE(SUM(stage2_passed), 0) as total_s2,
                COALESCE(SUM(stage3_passed), 0) as total_s3,
                COALESCE(SUM(stage4_passed), 0) as total_s4,
                COALESCE(SUM(stage5_passed), 0) as total_s5,
                COALESCE(SUM(rejected), 0) as total_rejected,
                COALESCE(SUM(errors), 0) as total_errors,
                COALESCE(MAX(best_sharpe), 0) as best_sharpe_ever,
                COALESCE(AVG(duration_seconds), 0) as avg_cycle_sec,
                COALESCE(MAX(cycle_num), 0) as latest_cycle
            FROM cycle_log
        """)
        if stat_rows:
            s = dict(stat_rows[0])
            total = s.get("total_ideas", 0) or 1
            s["kill_rate_pct"] = round(((total - (s.get("total_s5", 0) or 0)) / total) * 100, 1)
            # S1 pass rate
            s["s1_pass_rate"] = round(((s.get("total_s1", 0) or 0) / total) * 100, 1) if total > 0 else 0
            # S2 pass rate (of S1 passers)
            s1 = s.get("total_s1", 0) or 1
            s["s2_pass_rate"] = round(((s.get("total_s2", 0) or 0) / s1) * 100, 1) if s1 > 0 else 0
            result["stats"] = s

        # --- Winner count from DB ---
        wc_rows = safe_query(cur, "SELECT COUNT(*) as cnt FROM winning_strategies WHERE is_active = 1")
        if wc_rows:
            result["stats"]["total_winners"] = dict(wc_rows[0])["cnt"]

        # --- Backtest run count ---
        bt_rows = safe_query(cur, "SELECT COUNT(*) as cnt FROM backtest_runs")
        if bt_rows:
            result["stats"]["total_backtest_runs"] = dict(bt_rows[0]).get("cnt", 0)

        # --- Exploration coverage (derive archetype from strategy_name prefix) ---
        cov_rows = safe_query(cur, """
            SELECT
                CASE
                    WHEN strategy_name LIKE 'ORB%' THEN 'ORB'
                    WHEN strategy_name LIKE 'MA%' THEN 'MA_Crossover'
                    WHEN strategy_name LIKE 'EOD%' THEN 'EOD_Fade'
                    WHEN strategy_name LIKE 'Lunch%' OR strategy_name LIKE 'LUNCH%' THEN 'Lunch_Reversal'
                    WHEN strategy_name LIKE 'Gap%' OR strategy_name LIKE 'GAP%' THEN 'Gap_Fill'
                    WHEN strategy_name LIKE 'MR%' THEN 'Mean_Reversion'
                    WHEN strategy_name LIKE 'MOM%' OR strategy_name LIKE 'Momentum%' THEN 'Momentum'
                    ELSE 'Other'
                END as archetype,
                COUNT(*) as tested,
                SUM(CASE WHEN net_profit > 0 THEN 1 ELSE 0 END) as profitable,
                ROUND(MAX(sharpe_ratio), 4) as best_sharpe,
                ROUND(MAX(net_profit), 2) as best_profit
            FROM backtest_runs
            GROUP BY archetype
            ORDER BY tested DESC
        """)
        result["exploration"] = [dict(r) for r in cov_rows]

        conn.close()

    except Exception as e:
        result["error"] = str(e)

    return result


def get_winner_detail(db_path, winner_id):
    """Get full detail for a single winner including equity curve."""
    result = {"winner": None, "equity_curve": None}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = safe_query(cur, """
            SELECT * FROM winning_strategies WHERE id = ?
        """, (winner_id,))
        if rows:
            result["winner"] = dict(rows[0])

        eq_rows = safe_query(cur, """
            SELECT data_json FROM equity_curves
            WHERE strategy_id = ?
            ORDER BY id DESC LIMIT 1
        """, (winner_id,))
        if eq_rows:
            result["equity_curve"] = dict(eq_rows[0]).get("data_json")

        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def get_history(db_path, limit=100):
    """Get extended cycle history."""
    result = {"cycles": []}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = safe_query(cur, f"""
            SELECT cycle_num, started_at, finished_at, duration_seconds,
                   ideas_generated, backtests_run,
                   stage1_passed, stage2_passed, stage3_passed,
                   stage4_passed, stage5_passed,
                   rejected, errors, best_sharpe, best_strategy_name, gpu_used
            FROM cycle_log ORDER BY id DESC LIMIT {int(limit)}
        """)
        result["cycles"] = [dict(r) for r in rows]
        conn.close()
    except Exception as e:
        result["error"] = str(e)
    return result


def get_stage_strategies(db_path, stage_name, limit=20):
    """Get strategies at a specific pipeline stage for drill-down."""
    result = {"stage": stage_name, "strategies": []}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get from strategy_lifecycle
        # Try schema with rejection_reason first (production), fallback to notes (test)
        rows = safe_query(cur, """
            SELECT strategy_name, current_stage, degradation_strikes,
                   s1_passed_at, s2_passed_at, s3_passed_at, s4_passed_at, s5_passed_at,
                   created_at, updated_at, rejection_reason as notes
            FROM strategy_lifecycle
            WHERE current_stage = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (stage_name, limit))
        if not rows:
            # Fallback: try with 'notes' column (test DBs)
            rows = safe_query(cur, """
                SELECT strategy_name, current_stage, degradation_strikes,
                       s1_passed_at, s2_passed_at, s3_passed_at, s4_passed_at, s5_passed_at,
                       created_at, updated_at, notes
                FROM strategy_lifecycle
                WHERE current_stage = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (stage_name, limit))
        strategies = [dict(r) for r in rows]

        # Enrich with backtest_runs metrics if available
        for s in strategies:
            name = s.get("strategy_name", "")
            bt_rows = safe_query(cur, """
                SELECT sharpe_ratio, net_profit, total_trades, win_rate, profit_factor,
                       max_drawdown, max_drawdown_pct, timestamp
                FROM backtest_runs
                WHERE strategy_name = ?
                ORDER BY id DESC LIMIT 1
            """, (name,))
            if bt_rows:
                s["metrics"] = dict(bt_rows[0])

        result["strategies"] = strategies
        result["count"] = len(strategies)

        conn.close()
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No DB path provided"}))
        sys.exit(1)

    db_path = sys.argv[1]
    if not os.path.exists(db_path):
        print(json.dumps({"error": "DB not found"}))
        sys.exit(1)

    command = sys.argv[2] if len(sys.argv) > 2 else "dashboard"

    if command == "dashboard":
        print(json.dumps(get_dashboard_data(db_path)))
    elif command == "winner":
        winner_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        print(json.dumps(get_winner_detail(db_path, winner_id)))
    elif command == "history":
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        print(json.dumps(get_history(db_path, limit)))
    elif command == "stage":
        stage_name = sys.argv[3] if len(sys.argv) > 3 else "STAGE1_PASS"
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        print(json.dumps(get_stage_strategies(db_path, stage_name, limit)))
    else:
        print(json.dumps(get_dashboard_data(db_path)))


if __name__ == "__main__":
    main()
