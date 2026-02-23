"""
Marcus Database Backup & Restore

Exports all tables from marcus_registry.db to a compressed JSON archive.
Use --restore on a new machine to recreate the database from the export.

Usage:
    python marcus_db_backup.py              # Export database
    python marcus_db_backup.py --verify     # Export + verify row counts
    python marcus_db_backup.py --restore    # Restore database from latest export
"""

import argparse
import gzip
import json
import os
import sqlite3
import sys
from datetime import datetime

# Resolve paths relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUANT_LAB = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DB_PATH = os.path.join(_QUANT_LAB, "StrategyPipeline", "src", "backtesting", "marcus_registry.db")
_BACKUP_DIR = os.path.join(_SCRIPT_DIR, "db_backups")
_EXPORT_PATH = os.path.join(_BACKUP_DIR, "marcus_registry_export.json.gz")

# All tables and their CREATE statements (for restore)
TABLE_SCHEMAS = {
    "backtest_runs": """CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        strategy_name TEXT, symbol TEXT, interval TEXT, params_json TEXT,
        total_return REAL, cagr REAL, sharpe_ratio REAL, max_drawdown REAL,
        calmar_ratio REAL, profit_factor REAL, var_95 REAL, ending_equity REAL,
        data_range_start TEXT, data_range_end TEXT, regime TEXT, notes TEXT,
        source_code TEXT, hash_id TEXT UNIQUE, priority INTEGER DEFAULT 0,
        max_drawdown_pct REAL, win_rate REAL, total_trades INTEGER, net_profit REAL
    )""",
    "winning_strategies": """CREATE TABLE IF NOT EXISTS winning_strategies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        strategy_name TEXT NOT NULL, archetype TEXT, symbol TEXT, interval TEXT,
        sharpe_ratio REAL, total_return REAL, net_profit REAL,
        max_drawdown REAL, max_drawdown_pct REAL, win_rate REAL,
        profit_factor REAL, total_trades INTEGER, win_trades INTEGER,
        loss_trades INTEGER, avg_trade_pnl REAL, params_json TEXT,
        source_code TEXT, pine_script TEXT, equity_curve_json TEXT,
        quality_score REAL, quality_notes TEXT, monte_carlo_var95 REAL,
        permutation_pvalue REAL, regime_analysis_json TEXT,
        data_range_start TEXT, data_range_end TEXT, notes TEXT, tags TEXT,
        is_active INTEGER DEFAULT 1, priority INTEGER DEFAULT 0,
        hash_id TEXT UNIQUE
    )""",
    "equity_curves": """CREATE TABLE IF NOT EXISTS equity_curves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id INTEGER NOT NULL, curve_type TEXT DEFAULT 'backtest',
        data_json TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (strategy_id) REFERENCES winning_strategies(id)
    )""",
    "trade_logs": """CREATE TABLE IF NOT EXISTS trade_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id INTEGER NOT NULL, trades_json TEXT NOT NULL,
        total_trades INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (strategy_id) REFERENCES winning_strategies(id)
    )""",
    "strategy_lifecycle": """CREATE TABLE IF NOT EXISTS strategy_lifecycle (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id INTEGER, strategy_hash TEXT NOT NULL, strategy_name TEXT,
        archetype TEXT, current_stage TEXT DEFAULT 'CANDIDATE',
        s1_passed_at TEXT, s1_metrics_json TEXT,
        s2_passed_at TEXT, s2_metrics_json TEXT,
        s3_passed_at TEXT, s3_metrics_json TEXT,
        s4_passed_at TEXT, s4_metrics_json TEXT,
        s5_passed_at TEXT, s5_metrics_json TEXT,
        degradation_strikes INTEGER DEFAULT 0, rejection_reason TEXT,
        archived_at TEXT, created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    )""",
    "strategy_graveyard": """CREATE TABLE IF NOT EXISTS strategy_graveyard (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_hash TEXT UNIQUE NOT NULL, strategy_name TEXT,
        killed_at_stage TEXT, reason TEXT, best_sharpe REAL,
        total_trades INTEGER, created_at TEXT DEFAULT (datetime('now'))
    )""",
    "cycle_log": """CREATE TABLE IF NOT EXISTS cycle_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_num INTEGER, started_at TEXT, finished_at TEXT,
        duration_seconds REAL, ideas_generated INTEGER DEFAULT 0,
        backtests_run INTEGER DEFAULT 0, stage1_passed INTEGER DEFAULT 0,
        stage2_passed INTEGER DEFAULT 0, stage3_passed INTEGER DEFAULT 0,
        stage4_passed INTEGER DEFAULT 0, stage5_passed INTEGER DEFAULT 0,
        rejected INTEGER DEFAULT 0, disposed INTEGER DEFAULT 0,
        errors INTEGER DEFAULT 0, best_sharpe REAL, best_strategy_name TEXT,
        gpu_used INTEGER DEFAULT 0, notes TEXT
    )""",
    "system_health": """CREATE TABLE IF NOT EXISTS system_health (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT, report_json TEXT
    )""",
    "marcus_messages": """CREATE TABLE IF NOT EXISTS marcus_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL, sent_at TEXT DEFAULT (datetime('now')),
        acknowledged INTEGER DEFAULT 0, message_type TEXT DEFAULT 'user_guidance',
        status TEXT DEFAULT 'queued', applied_at TEXT, applied_to TEXT,
        result_notes TEXT
    )""",
    "marcus_focus_config": """CREATE TABLE IF NOT EXISTS marcus_focus_config (
        key TEXT PRIMARY KEY, value TEXT NOT NULL,
        updated_at TEXT DEFAULT (datetime('now'))
    )""",
    "marcus_thought_summaries": """CREATE TABLE IF NOT EXISTS marcus_thought_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_num INTEGER, timestamp TEXT DEFAULT (datetime('now')),
        summary_type TEXT, summary_text TEXT NOT NULL,
        context_json TEXT, source TEXT DEFAULT 'synthesized'
    )""",
}


def export_db(db_path=_DB_PATH, output_path=_EXPORT_PATH):
    """Export all tables from the database to a compressed JSON file."""
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    export = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "source_db": os.path.basename(db_path),
            "source_size_bytes": os.path.getsize(db_path),
            "tables": {},
        },
        "tables": {},
    }

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
    ).fetchall()]

    for table in tables:
        rows = conn.execute(f"SELECT * FROM [{table}]").fetchall()
        columns = [desc[0] for desc in conn.execute(f"SELECT * FROM [{table}] LIMIT 0").description]
        data = [dict(zip(columns, row)) for row in rows]
        export["tables"][table] = data
        export["metadata"]["tables"][table] = len(data)
        print(f"  {table}: {len(data)} rows")

    conn.close()

    # Write compressed
    json_bytes = json.dumps(export, default=str).encode("utf-8")
    with gzip.open(output_path, "wb", compresslevel=6) as f:
        f.write(json_bytes)

    raw_mb = len(json_bytes) / (1024 * 1024)
    gz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported {len(tables)} tables -> {output_path}")
    print(f"  Raw JSON: {raw_mb:.1f} MB -> Compressed: {gz_mb:.1f} MB ({gz_mb/raw_mb*100:.0f}%)")
    return export["metadata"]


def verify_export(db_path=_DB_PATH, export_path=_EXPORT_PATH):
    """Verify the export matches the source database."""
    if not os.path.exists(export_path):
        print(f"ERROR: Export not found at {export_path}")
        sys.exit(1)

    with gzip.open(export_path, "rb") as f:
        export = json.loads(f.read().decode("utf-8"))

    conn = sqlite3.connect(db_path)
    ok = True
    for table, expected_count in export["metadata"]["tables"].items():
        actual_count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        status = "OK" if actual_count == expected_count else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(f"  {table}: export={expected_count}, db={actual_count} [{status}]")

    conn.close()
    if ok:
        print("\nVerification PASSED - all row counts match")
    else:
        print("\nVerification FAILED - row count mismatches detected")
        sys.exit(1)


def restore_db(export_path=_EXPORT_PATH, db_path=_DB_PATH):
    """Restore the database from a compressed JSON export."""
    if not os.path.exists(export_path):
        print(f"ERROR: Export not found at {export_path}")
        sys.exit(1)

    if os.path.exists(db_path):
        backup = db_path + f".pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(db_path, backup)
        print(f"Existing database backed up to: {backup}")

    print(f"Loading export from {export_path}...")
    with gzip.open(export_path, "rb") as f:
        export = json.loads(f.read().decode("utf-8"))

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Create tables from schemas
    for table_name, schema in TABLE_SCHEMAS.items():
        conn.execute(schema)
    conn.commit()

    # Insert data
    for table_name, rows in export["tables"].items():
        if not rows:
            print(f"  {table_name}: 0 rows (empty)")
            continue

        columns = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_names = ", ".join([f"[{c}]" for c in columns])
        sql = f"INSERT OR REPLACE INTO [{table_name}] ({col_names}) VALUES ({placeholders})"

        batch = [tuple(row.get(c) for c in columns) for row in rows]
        conn.executemany(sql, batch)
        print(f"  {table_name}: {len(rows)} rows restored")

    conn.commit()
    conn.close()

    db_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\nDatabase restored to {db_path} ({db_size:.1f} MB)")
    print(f"Source export: {export['metadata']['exported_at']}")


def main():
    parser = argparse.ArgumentParser(description="Marcus Database Backup & Restore")
    parser.add_argument("--verify", action="store_true", help="Export and verify integrity")
    parser.add_argument("--restore", action="store_true", help="Restore database from export")
    parser.add_argument("--db", default=_DB_PATH, help="Path to database file")
    parser.add_argument("--export", default=_EXPORT_PATH, help="Path to export file")
    args = parser.parse_args()

    if args.restore:
        print("=== Marcus Database Restore ===")
        restore_db(export_path=args.export, db_path=args.db)
    else:
        print("=== Marcus Database Export ===")
        export_db(db_path=args.db, output_path=args.export)
        if args.verify:
            print("\n=== Verifying Export ===")
            verify_export(db_path=args.db, export_path=args.export)


if __name__ == "__main__":
    main()
