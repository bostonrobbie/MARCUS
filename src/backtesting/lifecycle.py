"""
Strategy Lifecycle Manager
===========================
State machine that manages the full lifecycle of strategies in the Marcus
autonomous research pipeline.

States:
    CANDIDATE  -> Just generated, untested
    TESTING    -> Currently being backtested
    STAGE1_PASS -> Passed basic profitability (standard costs)
    STAGE2_PASS -> Passed gauntlet stress test (2x costs)
    STAGE3_PASS -> Passed regime split (profitable in all periods)
    STAGE4_PASS -> Passed parameter sensitivity (robust to variations)
    STAGE5_PASS -> Passed complementarity (low correlation with existing)
    DEPLOYED   -> In paper/live trading
    DEGRADED   -> Was deployed, performance declining
    ARCHIVED   -> Retired but kept for reference
    REJECTED   -> Failed a gate, pending cleanup
    DELETED    -> Permanently removed (hash kept in graveyard)

Disposal Logic:
    - Stage 1 fail -> immediate delete (clearly garbage)
    - Stage 2+ fail -> archive (marginal, may learn from it)
    - 3 degradation strikes -> archive
    - Graveyard stores hashes of all killed ideas to prevent re-testing
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# Valid lifecycle stages in order
STAGES = [
    'CANDIDATE', 'TESTING',
    'STAGE1_PASS', 'STAGE2_PASS', 'STAGE3_PASS', 'STAGE4_PASS', 'STAGE5_PASS',
    'DEPLOYED', 'DEGRADED', 'ARCHIVED', 'REJECTED', 'DELETED'
]

ACTIVE_STAGES = {'STAGE5_PASS', 'DEPLOYED'}
TERMINAL_STAGES = {'ARCHIVED', 'REJECTED', 'DELETED'}


class StrategyLifecycleManager:
    """
    Manages the full lifecycle of strategies from idea to deployment to disposal.

    Uses the strategy_lifecycle and strategy_graveyard tables in the registry DB.
    """

    def __init__(self, db_path: str, config=None):
        self.db_path = db_path
        self.config = config
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_tables(self):
        """Verify lifecycle tables exist (created by registry._init_db)."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM strategy_lifecycle")
            cursor.execute("SELECT COUNT(*) FROM strategy_graveyard")
        except sqlite3.OperationalError:
            # Tables don't exist yet - they'll be created by StrategyRegistry
            logger.warning("Lifecycle tables not found. Run StrategyRegistry init first.")
        finally:
            conn.close()

    # =========================================================================
    # Registration
    # =========================================================================

    def register_candidate(self, strategy_hash: str, strategy_name: str,
                           archetype: str = "") -> int:
        """Register a new strategy candidate. Returns lifecycle ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO strategy_lifecycle
                    (strategy_hash, strategy_name, archetype, current_stage)
                VALUES (?, ?, ?, 'CANDIDATE')
            """, (strategy_hash, strategy_name, archetype))
            conn.commit()
            lifecycle_id = cursor.lastrowid
            logger.info(f"Registered candidate: {strategy_name} (LC-{lifecycle_id})")
            return lifecycle_id
        except Exception as e:
            logger.error(f"Failed to register candidate: {e}")
            return -1
        finally:
            conn.close()

    # =========================================================================
    # State Transitions
    # =========================================================================

    def promote(self, lifecycle_id: int, to_stage: str, metrics: Dict[str, Any] = None) -> bool:
        """
        Promote a strategy to the next stage. Records metrics at each gate.
        Returns True if promotion succeeded.
        """
        if to_stage not in STAGES:
            logger.error(f"Invalid stage: {to_stage}")
            return False

        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            # Get current state
            cursor.execute(
                "SELECT current_stage, strategy_name FROM strategy_lifecycle WHERE id = ?",
                (lifecycle_id,)
            )
            row = cursor.fetchone()
            if not row:
                logger.error(f"Lifecycle ID {lifecycle_id} not found")
                return False

            current_stage, name = row

            # Determine which stage column to update
            stage_col_map = {
                'STAGE1_PASS': ('s1_passed_at', 's1_metrics_json'),
                'STAGE2_PASS': ('s2_passed_at', 's2_metrics_json'),
                'STAGE3_PASS': ('s3_passed_at', 's3_metrics_json'),
                'STAGE4_PASS': ('s4_passed_at', 's4_metrics_json'),
                'STAGE5_PASS': ('s5_passed_at', 's5_metrics_json'),
            }

            now = datetime.now().isoformat()
            metrics_json = json.dumps(metrics or {}, default=str)

            if to_stage in stage_col_map:
                ts_col, metrics_col = stage_col_map[to_stage]
                cursor.execute(f"""
                    UPDATE strategy_lifecycle
                    SET current_stage = ?, {ts_col} = ?, {metrics_col} = ?, updated_at = ?
                    WHERE id = ?
                """, (to_stage, now, metrics_json, now, lifecycle_id))
            else:
                cursor.execute("""
                    UPDATE strategy_lifecycle
                    SET current_stage = ?, updated_at = ?
                    WHERE id = ?
                """, (to_stage, now, lifecycle_id))

            conn.commit()
            logger.info(f"PROMOTED: {name} ({current_stage} -> {to_stage})")
            return True

        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return False
        finally:
            conn.close()

    def set_testing(self, lifecycle_id: int) -> bool:
        """Mark strategy as currently being tested."""
        return self.promote(lifecycle_id, 'TESTING')

    def reject(self, lifecycle_id: int, reason: str, stage: str) -> None:
        """
        Reject a strategy.
        - Stage 1 failures get DELETED and sent to graveyard
        - Stage 2+ failures get ARCHIVED (may have useful data)
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT strategy_hash, strategy_name, s1_metrics_json FROM strategy_lifecycle WHERE id = ?",
                (lifecycle_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            strategy_hash, name, s1_json = row
            best_sharpe = 0.0
            total_trades = 0
            if s1_json:
                try:
                    m = json.loads(s1_json)
                    best_sharpe = float(m.get('sharpe_ratio', 0))
                    total_trades = int(m.get('total_trades', 0))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            now = datetime.now().isoformat()

            delete_it = (
                self.config and self.config.delete_stage1_failures and
                stage in ('STAGE1_PASS', 'TESTING', 'CANDIDATE')
            ) or stage in ('TESTING', 'CANDIDATE')

            if delete_it:
                # Send to graveyard and mark as DELETED
                self._send_to_graveyard(conn, cursor, strategy_hash, name,
                                        stage, reason, best_sharpe, total_trades)
                cursor.execute("""
                    UPDATE strategy_lifecycle
                    SET current_stage = 'DELETED', rejection_reason = ?, updated_at = ?
                    WHERE id = ?
                """, (reason, now, lifecycle_id))
            else:
                # Archive it - keep data for reference
                cursor.execute("""
                    UPDATE strategy_lifecycle
                    SET current_stage = 'ARCHIVED', rejection_reason = ?,
                        archived_at = ?, updated_at = ?
                    WHERE id = ?
                """, (reason, now, now, lifecycle_id))

            conn.commit()
            action = "DELETED" if delete_it else "ARCHIVED"
            logger.info(f"{action}: {name} at {stage} - {reason}")

        except Exception as e:
            logger.error(f"Rejection failed: {e}")
        finally:
            conn.close()

    def mark_degraded(self, lifecycle_id: int, reason: str) -> None:
        """Mark a DEPLOYED strategy as degraded. Increments strike counter."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT degradation_strikes, strategy_name FROM strategy_lifecycle WHERE id = ?",
                (lifecycle_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            strikes, name = row
            new_strikes = (strikes or 0) + 1
            now = datetime.now().isoformat()

            max_strikes = 3
            if self.config:
                max_strikes = self.config.max_degradation_strikes

            if new_strikes >= max_strikes:
                # Auto-archive after too many strikes
                cursor.execute("""
                    UPDATE strategy_lifecycle
                    SET current_stage = 'ARCHIVED', degradation_strikes = ?,
                        rejection_reason = ?, archived_at = ?, updated_at = ?
                    WHERE id = ?
                """, (new_strikes, f"Degraded {new_strikes}x: {reason}", now, now, lifecycle_id))
                logger.warning(f"AUTO-ARCHIVED: {name} after {new_strikes} degradation strikes")
            else:
                cursor.execute("""
                    UPDATE strategy_lifecycle
                    SET current_stage = 'DEGRADED', degradation_strikes = ?, updated_at = ?
                    WHERE id = ?
                """, (new_strikes, now, lifecycle_id))
                logger.info(f"DEGRADED: {name} (strike {new_strikes}/{max_strikes}) - {reason}")

            conn.commit()

        except Exception as e:
            logger.error(f"Degradation marking failed: {e}")
        finally:
            conn.close()

    def archive(self, lifecycle_id: int, reason: str) -> None:
        """Move strategy to ARCHIVED state."""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        try:
            cursor.execute("""
                UPDATE strategy_lifecycle
                SET current_stage = 'ARCHIVED', rejection_reason = ?,
                    archived_at = ?, updated_at = ?
                WHERE id = ?
            """, (reason, now, now, lifecycle_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Archive failed: {e}")
        finally:
            conn.close()

    # =========================================================================
    # Queries
    # =========================================================================

    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """All strategies in STAGE5_PASS or DEPLOYED state."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT id, strategy_id, strategy_hash, strategy_name, archetype,
                       current_stage, s5_metrics_json, created_at
                FROM strategy_lifecycle
                WHERE current_stage IN ('STAGE5_PASS', 'DEPLOYED')
                ORDER BY created_at DESC
            """)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get active strategies: {e}")
            return []
        finally:
            conn.close()

    def get_pipeline_status(self) -> Dict[str, int]:
        """Count of strategies in each state. Used by dashboard funnel."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT current_stage, COUNT(*)
                FROM strategy_lifecycle
                GROUP BY current_stage
            """)
            return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {}
        finally:
            conn.close()

    def get_lifecycle_by_hash(self, strategy_hash: str) -> Optional[Dict[str, Any]]:
        """Look up a lifecycle record by strategy hash."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM strategy_lifecycle WHERE strategy_hash = ? ORDER BY id DESC LIMIT 1",
                (strategy_hash,)
            )
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
        except Exception as e:
            logger.error(f"Lifecycle lookup failed: {e}")
            return None
        finally:
            conn.close()

    # =========================================================================
    # Disposal Logic
    # =========================================================================

    def run_disposal_sweep(self) -> Dict[str, int]:
        """
        Periodic cleanup:
        1. Clean old REJECTED/DELETED records (keep hashes in graveyard)
        2. Archive DEGRADED strategies with max strikes
        3. Enforce max_active_strategies cap

        Returns disposal counts.
        """
        report = {'cleaned': 0, 'archived': 0, 'capped': 0}
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            now = datetime.now()
            cleanup_days = 7
            if self.config:
                cleanup_days = self.config.rejected_cleanup_days

            cutoff = (now - timedelta(days=cleanup_days)).isoformat()

            # 1. Clean old DELETED records (graveyard hash already saved)
            cursor.execute("""
                DELETE FROM strategy_lifecycle
                WHERE current_stage = 'DELETED' AND updated_at < ?
            """, (cutoff,))
            report['cleaned'] = cursor.rowcount

            # 2. Archive DEGRADED with max strikes
            max_strikes = 3
            if self.config:
                max_strikes = self.config.max_degradation_strikes

            cursor.execute("""
                UPDATE strategy_lifecycle
                SET current_stage = 'ARCHIVED',
                    archived_at = ?,
                    rejection_reason = 'Auto-archived: max degradation strikes',
                    updated_at = ?
                WHERE current_stage = 'DEGRADED' AND degradation_strikes >= ?
            """, (now.isoformat(), now.isoformat(), max_strikes))
            report['archived'] = cursor.rowcount

            # 3. Enforce max active strategies cap
            max_active = 20
            if self.config:
                max_active = self.config.max_active_strategies

            cursor.execute("""
                SELECT id FROM strategy_lifecycle
                WHERE current_stage IN ('STAGE5_PASS', 'DEPLOYED')
                ORDER BY
                    CASE WHEN current_stage = 'DEPLOYED' THEN 0 ELSE 1 END,
                    s5_passed_at DESC
            """)
            active_ids = [row[0] for row in cursor.fetchall()]

            if len(active_ids) > max_active:
                to_archive = active_ids[max_active:]
                placeholders = ','.join('?' * len(to_archive))
                cursor.execute(f"""
                    UPDATE strategy_lifecycle
                    SET current_stage = 'ARCHIVED',
                        archived_at = ?,
                        rejection_reason = 'Capped: exceeded max active strategies',
                        updated_at = ?
                    WHERE id IN ({placeholders})
                """, [now.isoformat(), now.isoformat()] + to_archive)
                report['capped'] = cursor.rowcount

            conn.commit()
            logger.info(f"Disposal sweep: cleaned={report['cleaned']}, "
                        f"archived={report['archived']}, capped={report['capped']}")

        except Exception as e:
            logger.error(f"Disposal sweep failed: {e}")
        finally:
            conn.close()

        return report

    # =========================================================================
    # Graveyard
    # =========================================================================

    def _send_to_graveyard(self, conn, cursor, strategy_hash: str,
                           strategy_name: str, killed_at_stage: str,
                           reason: str, best_sharpe: float = 0.0,
                           total_trades: int = 0) -> None:
        """Save strategy hash to graveyard to prevent re-testing."""
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO strategy_graveyard
                    (strategy_hash, strategy_name, killed_at_stage, reason,
                     best_sharpe, total_trades)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (strategy_hash, strategy_name, killed_at_stage, reason,
                  best_sharpe, total_trades))
            # Don't commit here - caller handles transaction
        except Exception as e:
            logger.error(f"Graveyard insert failed: {e}")

    def is_in_graveyard(self, strategy_hash: str) -> bool:
        """Check if this idea hash was already tried and killed."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM strategy_graveyard WHERE strategy_hash = ?",
                (strategy_hash,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Graveyard check failed: {e}")
            return False
        finally:
            conn.close()

    def get_graveyard_count(self) -> int:
        """How many strategies have been permanently killed."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM strategy_graveyard")
            return cursor.fetchone()[0]
        except Exception as e:
            return 0
        finally:
            conn.close()

    def get_disposal_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Recent disposals from both graveyard and archived strategies."""
        conn = self._get_conn()
        cursor = conn.cursor()
        entries = []
        try:
            # Graveyard entries (permanently killed)
            cursor.execute("""
                SELECT strategy_name, killed_at_stage as stage, reason,
                       best_sharpe, created_at, 'KILLED' as action
                FROM strategy_graveyard
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            entries.extend([dict(zip(columns, row)) for row in cursor.fetchall()])

            # Recently archived
            cursor.execute("""
                SELECT strategy_name, current_stage as stage, rejection_reason as reason,
                       0.0 as best_sharpe, archived_at as created_at, 'ARCHIVED' as action
                FROM strategy_lifecycle
                WHERE current_stage = 'ARCHIVED' AND archived_at IS NOT NULL
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            entries.extend([dict(zip(columns, row)) for row in cursor.fetchall()])

            # Sort by date, most recent first
            entries.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return entries[:limit]

        except Exception as e:
            logger.error(f"Failed to get disposal log: {e}")
            return []
        finally:
            conn.close()
