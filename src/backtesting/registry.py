"""
Strategy Registry & Winning Strategy Database
===============================================
Comprehensive knowledge base for all backtested strategies.

Stores:
- Full performance metrics (Sharpe, drawdown, win rate, PF, etc.)
- Strategy source code snapshots
- Serialized equity curves (viewable as snapshots)
- Parameter configurations
- Market regime context
- Validation/quality gate results

Provides:
- Leaderboard of top strategies
- Duplicate detection (hash-based)
- Equity curve retrieval for visualization
- Strategy comparison
- Auto-migration for schema changes
"""

import sqlite3
import json
import os
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_float(val, default=0.0) -> float:
    """Safely convert to float, handling None/NaN/Inf."""
    if val is None:
        return default
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0) -> int:
    """Safely convert to int, handling None."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class StrategyRegistry:
    """
    Enhanced Trading Knowledge Base with comprehensive strategy storage.

    Persists metrics, source code, equity curves, notes, and market regime context.
    Provides duplicate detection, leaderboard, and strategy comparison.
    """

    def __init__(self, db_path: str = "backtests.db"):
        self.db_path = db_path
        self._init_db()
        self._migrate_db()

    def _init_db(self):
        """Initialize SQLite database schema with full strategy storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode for better concurrent read/write performance
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set busy timeout to 30 seconds (prevents "database is locked" errors)
        cursor.execute("PRAGMA busy_timeout=30000")
        # Enable foreign key enforcement
        cursor.execute("PRAGMA foreign_keys=ON")

        # Main backtest runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT,
                symbol TEXT,
                interval TEXT,
                params_json TEXT,
                total_return REAL,
                cagr REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                calmar_ratio REAL,
                profit_factor REAL,
                var_95 REAL,
                ending_equity REAL,
                data_range_start TEXT,
                data_range_end TEXT,
                regime TEXT,
                notes TEXT,
                source_code TEXT,
                hash_id TEXT UNIQUE
            )
        """)

        # Winning strategies table - curated strategies that passed quality gate
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS winning_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy_name TEXT NOT NULL,
                archetype TEXT,
                symbol TEXT,
                interval TEXT,
                -- Performance metrics
                sharpe_ratio REAL,
                total_return REAL,
                net_profit REAL,
                max_drawdown REAL,
                max_drawdown_pct REAL,
                win_rate REAL,
                profit_factor REAL,
                total_trades INTEGER,
                win_trades INTEGER,
                loss_trades INTEGER,
                avg_trade_pnl REAL,
                -- Configuration
                params_json TEXT,
                -- Code snapshot
                source_code TEXT,
                pine_script TEXT,
                -- Equity curve (JSON-serialized list of {date, equity} pairs)
                equity_curve_json TEXT,
                -- Validation
                quality_score REAL,
                quality_notes TEXT,
                monte_carlo_var95 REAL,
                permutation_pvalue REAL,
                regime_analysis_json TEXT,
                -- Metadata
                data_range_start TEXT,
                data_range_end TEXT,
                notes TEXT,
                tags TEXT,
                is_active INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 0,
                -- Deduplication
                hash_id TEXT UNIQUE
            )
        """)

        # Equity curve snapshots (for large curves, stored separately)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                curve_type TEXT DEFAULT 'backtest',
                data_json TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES winning_strategies(id)
            )
        """)

        # Trade log snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                trades_json TEXT NOT NULL,
                total_trades INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES winning_strategies(id)
            )
        """)

        # System Health Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                report_json TEXT
            )
        """)

        # Strategy lifecycle tracking (Marcus autonomous agent)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER,
                strategy_hash TEXT NOT NULL,
                strategy_name TEXT,
                archetype TEXT,
                current_stage TEXT DEFAULT 'CANDIDATE',
                s1_passed_at TEXT, s1_metrics_json TEXT,
                s2_passed_at TEXT, s2_metrics_json TEXT,
                s3_passed_at TEXT, s3_metrics_json TEXT,
                s4_passed_at TEXT, s4_metrics_json TEXT,
                s5_passed_at TEXT, s5_metrics_json TEXT,
                degradation_strikes INTEGER DEFAULT 0,
                rejection_reason TEXT,
                archived_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Graveyard - hashes of permanently discarded strategies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_graveyard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_hash TEXT UNIQUE NOT NULL,
                strategy_name TEXT,
                killed_at_stage TEXT,
                reason TEXT,
                best_sharpe REAL,
                total_trades INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Cycle execution log (for daemon dashboard)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycle_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_num INTEGER,
                started_at TEXT,
                finished_at TEXT,
                duration_seconds REAL,
                ideas_generated INTEGER DEFAULT 0,
                backtests_run INTEGER DEFAULT 0,
                stage1_passed INTEGER DEFAULT 0,
                stage2_passed INTEGER DEFAULT 0,
                stage3_passed INTEGER DEFAULT 0,
                stage4_passed INTEGER DEFAULT 0,
                stage5_passed INTEGER DEFAULT 0,
                rejected INTEGER DEFAULT 0,
                disposed INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                best_sharpe REAL,
                best_strategy_name TEXT,
                gpu_used INTEGER DEFAULT 0,
                notes TEXT
            )
        """)

        # Create indexes for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_sharpe ON backtest_runs(sharpe_ratio DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_name ON backtest_runs(strategy_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_winning_sharpe ON winning_strategies(sharpe_ratio DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_winning_active ON winning_strategies(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_stage ON strategy_lifecycle(current_stage)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_hash ON strategy_lifecycle(strategy_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_graveyard_hash ON strategy_graveyard(strategy_hash)")

        conn.commit()
        conn.close()

    def _migrate_db(self):
        """Auto-migrate existing database to add any missing columns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Migrate backtest_runs table
        cursor.execute("PRAGMA table_info(backtest_runs)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        new_cols = {
            'regime': 'TEXT',
            'notes': 'TEXT',
            'source_code': 'TEXT',
            'hash_id': 'TEXT',
            'priority': 'INTEGER DEFAULT 0',
            'max_drawdown_pct': 'REAL',
            'win_rate': 'REAL',
            'total_trades': 'INTEGER',
            'net_profit': 'REAL',
        }

        for col, col_type in new_cols.items():
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE backtest_runs ADD COLUMN {col} {col_type}")
                    logger.info(f"Migrated backtest_runs: added column {col}")
                    if col == 'hash_id':
                        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_hash_id ON backtest_runs(hash_id)")
                except Exception as e:
                    logger.debug(f"Column {col} migration note: {e}")

        conn.commit()
        conn.close()

    # =========================================================================
    # Hash & Duplicate Detection
    # =========================================================================

    def _generate_hash(self, strategy_name: str, symbol: str, interval: str,
                       params: Dict[str, Any], data_range: tuple) -> str:
        """Generates a unique hash for a backtest configuration."""
        hash_str = f"{strategy_name}_{symbol}_{interval}_{json.dumps(params, sort_keys=True, default=str)}_{data_range}"
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def check_duplicate(self, strategy_name: str, symbol: str, interval: str,
                        params: Dict[str, Any], data_range: tuple) -> Optional[Dict[str, Any]]:
        """Checks if a backtest with identical config already exists."""
        hash_id = self._generate_hash(strategy_name, symbol, interval, params, data_range)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT timestamp, sharpe_ratio, total_return FROM backtest_runs WHERE hash_id = ?",
                (hash_id,)
            )
            row = cursor.fetchone()
            if row:
                return {'timestamp': row[0], 'sharpe_ratio': row[1], 'total_return': row[2]}
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
        finally:
            conn.close()
        return None

    # =========================================================================
    # Backtest Run Storage (all runs, pass or fail)
    # =========================================================================

    def save_run(self, strategy_name: str, symbol: str, interval: str,
                 params: Dict[str, Any], stats: Dict[str, Any],
                 data_range: tuple, regime: str = "UNKNOWN",
                 notes: str = "", source_code: str = ""):
        """Persist a backtest run with full context to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date, end_date = data_range
        hash_id = self._generate_hash(strategy_name, symbol, interval, params, data_range)

        try:
            cursor.execute("""
                INSERT INTO backtest_runs (
                    strategy_name, symbol, interval, params_json,
                    total_return, cagr, sharpe_ratio, max_drawdown,
                    calmar_ratio, profit_factor, var_95, ending_equity,
                    data_range_start, data_range_end, regime, notes, source_code, hash_id,
                    win_rate, total_trades, net_profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, symbol, interval, json.dumps(params, default=str),
                _safe_float(stats.get('Total Return', stats.get('total_return'))),
                _safe_float(stats.get('CAGR', stats.get('cagr'))),
                _safe_float(stats.get('Sharpe Ratio', stats.get('sharpe_ratio'))),
                _safe_float(stats.get('Max Drawdown', stats.get('max_drawdown'))),
                _safe_float(stats.get('Calmar Ratio', stats.get('calmar_ratio'))),
                _safe_float(stats.get('Profit Factor', stats.get('profit_factor'))),
                _safe_float(stats.get('VaR (95%)', stats.get('var_95'))),
                _safe_float(stats.get('Ending Equity', stats.get('ending_equity', stats.get('final_equity')))),
                str(start_date), str(end_date),
                regime, notes, source_code, hash_id,
                _safe_float(stats.get('Win Rate', stats.get('win_rate'))),
                _safe_int(stats.get('Total Trades', stats.get('total_trades'))),
                _safe_float(stats.get('Net Profit', stats.get('net_profit'))),
            ))
            conn.commit()
            logger.info(f"Registry: archived {strategy_name}")
        except sqlite3.IntegrityError:
            logger.debug(f"Duplicate run skipped: {strategy_name}")
        except Exception as e:
            logger.error(f"Registry save error: {e}")
        finally:
            conn.close()

    def save_batch(self, runs: List[Dict[str, Any]]):
        """Batch insert multiple runs."""
        if not runs:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data_to_insert = []
        for run in runs:
            params = run.get('params', {})
            stats = run.get('stats', {})
            data_range = run.get('data_range', ('', ''))

            hash_id = self._generate_hash(
                run.get('strategy_name', ''),
                run.get('symbol', ''),
                run.get('interval', ''),
                params, data_range
            )

            data_to_insert.append((
                run.get('strategy_name', ''), run.get('symbol', ''),
                run.get('interval', ''), json.dumps(params, default=str),
                _safe_float(stats.get('Total Return')), _safe_float(stats.get('CAGR')),
                _safe_float(stats.get('Sharpe Ratio')), _safe_float(stats.get('Max Drawdown')),
                _safe_float(stats.get('Calmar Ratio')), _safe_float(stats.get('Profit Factor')),
                _safe_float(stats.get('VaR (95%)')), _safe_float(stats.get('Ending Equity')),
                str(data_range[0]), str(data_range[1]),
                run.get('regime', 'UNKNOWN'), run.get('notes', ''),
                run.get('source_code', ''), hash_id,
            ))

        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO backtest_runs (
                    strategy_name, symbol, interval, params_json,
                    total_return, cagr, sharpe_ratio, max_drawdown,
                    calmar_ratio, profit_factor, var_95, ending_equity,
                    data_range_start, data_range_end, regime, notes, source_code, hash_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            conn.commit()
            logger.info(f"Registry: batch archived {cursor.rowcount} runs")
        except Exception as e:
            logger.error(f"Batch save error: {e}")
        finally:
            conn.close()

    # =========================================================================
    # Winning Strategy Storage (curated, quality-gated strategies)
    # =========================================================================

    def save_winning_strategy(
        self,
        strategy_name: str,
        archetype: str = "",
        symbol: str = "",
        interval: str = "",
        metrics: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        source_code: str = "",
        pine_script: str = "",
        equity_curve: Any = None,
        trade_log: List[Dict] = None,
        quality_notes: str = "",
        data_range: Tuple[str, str] = ("", ""),
        notes: str = "",
        tags: str = "",
        monte_carlo_var95: float = 0.0,
        permutation_pvalue: float = 1.0,
        regime_analysis: Dict = None,
    ) -> Optional[int]:
        """
        Save a winning (quality-gated) strategy with full context.

        Returns the strategy ID, or None if save failed.
        """
        metrics = metrics or {}
        params = params or {}

        hash_id = self._generate_hash(strategy_name, symbol, interval, params, data_range)

        # Serialize equity curve
        equity_json = ""
        if equity_curve is not None:
            equity_json = self._serialize_equity_curve(equity_curve)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO winning_strategies (
                    strategy_name, archetype, symbol, interval,
                    sharpe_ratio, total_return, net_profit, max_drawdown, max_drawdown_pct,
                    win_rate, profit_factor, total_trades, win_trades, loss_trades, avg_trade_pnl,
                    params_json, source_code, pine_script, equity_curve_json,
                    quality_score, quality_notes, monte_carlo_var95, permutation_pvalue,
                    regime_analysis_json,
                    data_range_start, data_range_end, notes, tags, hash_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, archetype, symbol, interval,
                _safe_float(metrics.get('sharpe_ratio')),
                _safe_float(metrics.get('total_return')),
                _safe_float(metrics.get('net_profit')),
                _safe_float(metrics.get('max_drawdown')),
                _safe_float(metrics.get('max_drawdown_pct', metrics.get('max_drawdown'))),
                _safe_float(metrics.get('win_rate')),
                _safe_float(metrics.get('profit_factor')),
                _safe_int(metrics.get('total_trades')),
                _safe_int(metrics.get('win_trades')),
                _safe_int(metrics.get('loss_trades')),
                _safe_float(metrics.get('avg_trade_pnl')),
                json.dumps(params, default=str),
                source_code, pine_script, equity_json,
                _safe_float(metrics.get('quality_score', metrics.get('sharpe_ratio'))),
                quality_notes, monte_carlo_var95, permutation_pvalue,
                json.dumps(regime_analysis or {}, default=str),
                str(data_range[0]), str(data_range[1]),
                notes, tags, hash_id,
            ))
            conn.commit()
            strategy_id = cursor.lastrowid

            # Save equity curve separately if large
            if equity_json and len(equity_json) > 1000:
                self._save_equity_curve(cursor, conn, strategy_id, equity_json)

            # Save trade log if provided
            if trade_log:
                self._save_trade_log(cursor, conn, strategy_id, trade_log)

            logger.info(f"Winning strategy saved: {strategy_name} (ID={strategy_id})")
            return strategy_id

        except Exception as e:
            logger.error(f"Failed to save winning strategy: {e}")
            return None
        finally:
            conn.close()

    def _serialize_equity_curve(self, equity_curve) -> str:
        """Convert equity curve to JSON string."""
        try:
            if isinstance(equity_curve, pd.Series):
                data = [
                    {"date": str(idx), "equity": float(val)}
                    for idx, val in equity_curve.items()
                    if not np.isnan(val)
                ]
                # Downsample if too many points (keep ~500 for display)
                if len(data) > 500:
                    step = len(data) // 500
                    data = data[::step] + [data[-1]]
                return json.dumps(data)

            elif isinstance(equity_curve, pd.DataFrame):
                if 'equity' in equity_curve.columns:
                    return self._serialize_equity_curve(equity_curve['equity'])
                return json.dumps([])

            elif isinstance(equity_curve, list):
                # Already a list of dicts
                if len(equity_curve) > 500:
                    step = len(equity_curve) // 500
                    equity_curve = equity_curve[::step] + [equity_curve[-1]]
                return json.dumps(equity_curve, default=str)

            return json.dumps([])
        except Exception as e:
            logger.warning(f"Could not serialize equity curve: {e}")
            return json.dumps([])

    def _save_equity_curve(self, cursor, conn, strategy_id: int, equity_json: str):
        """Store equity curve data in separate table."""
        try:
            cursor.execute(
                "INSERT INTO equity_curves (strategy_id, curve_type, data_json) VALUES (?, ?, ?)",
                (strategy_id, 'backtest', equity_json)
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not save equity curve: {e}")

    def _save_trade_log(self, cursor, conn, strategy_id: int, trade_log: List[Dict]):
        """Store trade log in separate table."""
        try:
            trades_json = json.dumps(trade_log, default=str)
            cursor.execute(
                "INSERT INTO trade_logs (strategy_id, trades_json, total_trades) VALUES (?, ?, ?)",
                (strategy_id, trades_json, len(trade_log))
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not save trade log: {e}")

    # =========================================================================
    # Queries & Leaderboard
    # =========================================================================

    def get_leaderboard(self, limit: int = 10, min_trades: int = 0) -> pd.DataFrame:
        """Retrieve top strategies from all backtest runs."""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT strategy_name, symbol, interval, sharpe_ratio, total_return,
                   profit_factor, win_rate, total_trades, max_drawdown,
                   regime, notes, timestamp
            FROM backtest_runs
            WHERE total_trades >= ? OR total_trades IS NULL
            ORDER BY sharpe_ratio DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(min_trades, limit))
        conn.close()
        return df

    def get_winning_strategies(self, limit: int = 20, active_only: bool = True) -> pd.DataFrame:
        """Retrieve winning (quality-gated) strategies."""
        conn = sqlite3.connect(self.db_path)
        where = "WHERE is_active = 1" if active_only else ""
        query = f"""
            SELECT id, strategy_name, archetype, symbol, interval,
                   sharpe_ratio, total_return, net_profit, max_drawdown,
                   win_rate, profit_factor, total_trades,
                   quality_score, data_range_start, data_range_end,
                   notes, tags, timestamp
            FROM winning_strategies
            {where}
            ORDER BY sharpe_ratio DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df

    def get_strategy_detail(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get full details of a winning strategy including code and equity curve."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM winning_strategies WHERE id = ?", (strategy_id,))
            row = cursor.fetchone()
            if not row:
                return None

            columns = [desc[0] for desc in cursor.description]
            detail = dict(zip(columns, row))

            # Parse JSON fields
            for json_field in ['params_json', 'equity_curve_json', 'regime_analysis_json']:
                if detail.get(json_field):
                    try:
                        detail[json_field] = json.loads(detail[json_field])
                    except (json.JSONDecodeError, TypeError):
                        pass

            return detail

        except Exception as e:
            logger.error(f"Failed to get strategy detail: {e}")
            return None
        finally:
            conn.close()

    def get_equity_curve(self, strategy_id: int) -> Optional[pd.DataFrame]:
        """Retrieve equity curve as a DataFrame for visualization."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Try equity_curves table first
            cursor.execute(
                "SELECT data_json FROM equity_curves WHERE strategy_id = ? ORDER BY id DESC LIMIT 1",
                (strategy_id,)
            )
            row = cursor.fetchone()

            if not row:
                # Fall back to inline equity curve in winning_strategies
                cursor.execute(
                    "SELECT equity_curve_json FROM winning_strategies WHERE id = ?",
                    (strategy_id,)
                )
                row = cursor.fetchone()

            if not row or not row[0]:
                return None

            data = json.loads(row[0])
            if not data:
                return None

            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to get equity curve: {e}")
            return None
        finally:
            conn.close()

    def get_strategy_code(self, strategy_id: int) -> Optional[str]:
        """Retrieve the source code snapshot for a winning strategy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT source_code FROM winning_strategies WHERE id = ?",
                (strategy_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get strategy code: {e}")
            return None
        finally:
            conn.close()

    def get_historical_average(self, strategy_name: str, symbol: Optional[str] = None) -> Dict[str, float]:
        """Calculate average metrics for a strategy/symbol combination."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT AVG(sharpe_ratio), AVG(total_return), AVG(max_drawdown) FROM backtest_runs WHERE strategy_name = ?"
        params_list: List[Any] = [strategy_name]

        if symbol:
            query += " AND symbol = ?"
            params_list.append(symbol)

        try:
            cursor.execute(query, params_list)
            row = cursor.fetchone()
            if row and row[0] is not None:
                return {
                    'avg_sharpe': _safe_float(row[0]),
                    'avg_return': _safe_float(row[1]),
                    'avg_drawdown': _safe_float(row[2]),
                }
        except Exception as e:
            logger.error(f"Historical average query failed: {e}")
        finally:
            conn.close()

        return {}

    def compare_strategies(self, strategy_ids: List[int]) -> pd.DataFrame:
        """Compare multiple winning strategies side by side."""
        conn = sqlite3.connect(self.db_path)
        placeholders = ','.join('?' * len(strategy_ids))
        query = f"""
            SELECT id, strategy_name, sharpe_ratio, total_return, max_drawdown,
                   win_rate, profit_factor, total_trades, quality_score
            FROM winning_strategies
            WHERE id IN ({placeholders})
            ORDER BY sharpe_ratio DESC
        """
        df = pd.read_sql_query(query, conn, params=strategy_ids)
        conn.close()
        return df

    # =========================================================================
    # System Health
    # =========================================================================

    def log_diagnostic(self, status: str, report: Dict[str, Any]):
        """Persist a system health check report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO system_health (status, report_json) VALUES (?, ?)",
                (status, json.dumps(report, default=str))
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to archive diagnostic: {e}")
        finally:
            conn.close()

    # =========================================================================
    # Summary / Stats
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire registry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        summary = {}

        try:
            cursor.execute("SELECT COUNT(*) FROM backtest_runs")
            summary['total_runs'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM winning_strategies WHERE is_active = 1")
            summary['winning_strategies'] = cursor.fetchone()[0]

            cursor.execute("SELECT MAX(sharpe_ratio) FROM winning_strategies WHERE is_active = 1")
            row = cursor.fetchone()
            summary['best_sharpe'] = _safe_float(row[0]) if row else 0.0

            cursor.execute("""
                SELECT strategy_name, sharpe_ratio
                FROM winning_strategies WHERE is_active = 1
                ORDER BY sharpe_ratio DESC LIMIT 1
            """)
            row = cursor.fetchone()
            summary['best_strategy'] = row[0] if row else 'None'

        except Exception as e:
            logger.error(f"Summary query failed: {e}")
        finally:
            conn.close()

        return summary

    def get_total_backtest_count(self) -> int:
        """Get total number of backtest runs. Used for Deflated Sharpe Ratio n_trials."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM backtest_runs")
            return cursor.fetchone()[0]
        except Exception:
            return 0
        finally:
            conn.close()

    # =========================================================================
    # Lifecycle & Graveyard Queries (Marcus Autonomous Agent)
    # =========================================================================

    def log_cycle(self, cycle_data: Dict[str, Any]) -> int:
        """Log a completed research cycle. Returns cycle log ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO cycle_log (
                    cycle_num, started_at, finished_at, duration_seconds,
                    ideas_generated, backtests_run,
                    stage1_passed, stage2_passed, stage3_passed, stage4_passed, stage5_passed,
                    rejected, disposed, errors,
                    best_sharpe, best_strategy_name, gpu_used, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_data.get('cycle_num', 0),
                cycle_data.get('started_at', ''),
                cycle_data.get('finished_at', ''),
                _safe_float(cycle_data.get('duration_seconds')),
                _safe_int(cycle_data.get('ideas_generated')),
                _safe_int(cycle_data.get('backtests_run')),
                _safe_int(cycle_data.get('stage1_passed')),
                _safe_int(cycle_data.get('stage2_passed')),
                _safe_int(cycle_data.get('stage3_passed')),
                _safe_int(cycle_data.get('stage4_passed')),
                _safe_int(cycle_data.get('stage5_passed')),
                _safe_int(cycle_data.get('rejected')),
                _safe_int(cycle_data.get('disposed')),
                _safe_int(cycle_data.get('errors')),
                _safe_float(cycle_data.get('best_sharpe')),
                cycle_data.get('best_strategy_name', ''),
                1 if cycle_data.get('gpu_used') else 0,
                cycle_data.get('notes', ''),
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log cycle: {e}")
            return -1
        finally:
            conn.close()

    def get_cycle_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent cycle execution history for dashboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM cycle_log ORDER BY id DESC LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get cycle history: {e}")
            return []
        finally:
            conn.close()

    def get_pipeline_counts(self) -> Dict[str, int]:
        """Count strategies in each lifecycle stage for the dashboard funnel."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT current_stage, COUNT(*) FROM strategy_lifecycle
                WHERE current_stage NOT IN ('REJECTED', 'DELETED')
                GROUP BY current_stage
            """)
            counts = dict(cursor.fetchall())

            # Also count total tested and graveyard
            cursor.execute("SELECT COUNT(*) FROM backtest_runs")
            counts['total_tested'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM strategy_graveyard")
            counts['graveyard'] = cursor.fetchone()[0]

            return counts
        except Exception as e:
            logger.error(f"Failed to get pipeline counts: {e}")
            return {}
        finally:
            conn.close()

    def get_graveyard_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent graveyard entries (killed strategies)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT strategy_name, killed_at_stage, reason, best_sharpe,
                       total_trades, created_at
                FROM strategy_graveyard
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get graveyard entries: {e}")
            return []
        finally:
            conn.close()

    def get_exploration_coverage(self) -> Dict[str, Any]:
        """P2-4: Get exploration coverage stats by archetype.

        Returns a dict of archetype -> {tested, profitable, best_sharpe, best_name}
        showing how thoroughly each strategy space has been explored.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Parse archetype from strategy name or regime
            cursor.execute("""
                SELECT
                    CASE
                        WHEN strategy_name LIKE 'ORB_%' THEN 'orb_breakout'
                        WHEN strategy_name LIKE 'MA_%' THEN 'ma_crossover'
                        WHEN strategy_name LIKE 'EOD_%' THEN 'eod_momentum'
                        WHEN strategy_name LIKE 'Lunch_%' THEN 'lunch_hour_breakout'
                        WHEN strategy_name LIKE 'GapFade_%' THEN 'gap_fill_fade'
                        ELSE 'other'
                    END as archetype,
                    COUNT(*) as tested,
                    SUM(CASE WHEN net_profit > 0 THEN 1 ELSE 0 END) as profitable,
                    MAX(sharpe_ratio) as best_sharpe,
                    -- Find the name of the strategy with the best Sharpe
                    (SELECT b2.strategy_name FROM backtest_runs b2
                     WHERE b2.sharpe_ratio = MAX(backtest_runs.sharpe_ratio)
                     AND (CASE
                        WHEN b2.strategy_name LIKE 'ORB_%' THEN 'orb_breakout'
                        WHEN b2.strategy_name LIKE 'MA_%' THEN 'ma_crossover'
                        WHEN b2.strategy_name LIKE 'EOD_%' THEN 'eod_momentum'
                        WHEN b2.strategy_name LIKE 'Lunch_%' THEN 'lunch_hour_breakout'
                        WHEN b2.strategy_name LIKE 'GapFade_%' THEN 'gap_fill_fade'
                        ELSE 'other'
                     END) = (CASE
                        WHEN backtest_runs.strategy_name LIKE 'ORB_%' THEN 'orb_breakout'
                        WHEN backtest_runs.strategy_name LIKE 'MA_%' THEN 'ma_crossover'
                        WHEN backtest_runs.strategy_name LIKE 'EOD_%' THEN 'eod_momentum'
                        WHEN backtest_runs.strategy_name LIKE 'Lunch_%' THEN 'lunch_hour_breakout'
                        WHEN backtest_runs.strategy_name LIKE 'GapFade_%' THEN 'gap_fill_fade'
                        ELSE 'other'
                     END)
                     LIMIT 1) as best_name
                FROM backtest_runs
                GROUP BY archetype
                ORDER BY tested DESC
            """)
            columns = ['archetype', 'tested', 'profitable', 'best_sharpe', 'best_name']
            rows = cursor.fetchall()
            return {
                'by_archetype': [dict(zip(columns, row)) for row in rows],
                'total_tested': sum(r[1] for r in rows),
                'total_profitable': sum(r[2] or 0 for r in rows),
            }
        except Exception as e:
            logger.error(f"Exploration coverage query failed: {e}")
            return {'by_archetype': [], 'total_tested': 0, 'total_profitable': 0}
        finally:
            conn.close()

    def get_next_cycle_num(self) -> int:
        """Get the next cycle number."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT MAX(cycle_num) FROM cycle_log")
            row = cursor.fetchone()
            return (row[0] or 0) + 1
        except Exception as e:
            logger.error(f"Failed to get next cycle num: {e}")
            return 1
        finally:
            conn.close()
