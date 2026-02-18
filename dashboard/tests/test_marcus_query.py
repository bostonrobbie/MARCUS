#!/usr/bin/env python3
"""
QA Tests for marcus-query.py - the SQLite query backend for the Marcus dashboard.

Covers:
- Unit tests: each query function individually
- Integration tests: full dashboard data payload
- Edge cases: empty DB, missing tables, corrupt data
- Regression: leaderboard column name fix (timestamp vs created_at)

Run: python -m pytest tests/test_marcus_query.py -v
  or: python tests/test_marcus_query.py
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

# Add parent directory to path so we can import marcus-query
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Real DB path for integration tests ──────────────────────────────────
REAL_DB = os.path.join(
    'C:', os.sep, 'Users', 'User', 'Desktop', 'Zero_Human_HQ',
    'Quant_Lab', 'StrategyPipeline', 'src', 'backtesting', 'marcus_registry.db'
)
QUERY_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'marcus-query.py')


def create_test_db(path):
    """Create a minimal test DB with known data."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Create tables matching the real schema
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            strategy_name TEXT NOT NULL,
            symbol TEXT, interval TEXT, params_json TEXT,
            total_return REAL, cagr REAL, sharpe_ratio REAL,
            max_drawdown REAL, calmar_ratio REAL, profit_factor REAL,
            var_95 REAL, ending_equity REAL,
            data_range_start TEXT, data_range_end TEXT,
            regime TEXT, notes TEXT, source_code TEXT, hash_id TEXT,
            priority INTEGER, max_drawdown_pct REAL, win_rate REAL,
            total_trades INTEGER, net_profit REAL
        );

        CREATE TABLE IF NOT EXISTS winning_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            strategy_name TEXT NOT NULL,
            archetype TEXT, symbol TEXT, interval TEXT,
            sharpe_ratio REAL, total_return REAL, net_profit REAL,
            max_drawdown REAL, max_drawdown_pct REAL,
            win_rate REAL, profit_factor REAL,
            total_trades INTEGER, win_trades INTEGER, loss_trades INTEGER,
            avg_trade_pnl REAL, params_json TEXT, source_code TEXT,
            pine_script TEXT, equity_curve_json TEXT,
            quality_score REAL, quality_notes TEXT,
            monte_carlo_var95 REAL, permutation_pvalue REAL,
            regime_analysis_json TEXT,
            data_range_start TEXT, data_range_end TEXT,
            notes TEXT, tags TEXT,
            is_active INTEGER DEFAULT 1, priority INTEGER DEFAULT 0,
            hash_id TEXT
        );

        CREATE TABLE IF NOT EXISTS equity_curves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            curve_type TEXT DEFAULT 'backtest',
            data_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS strategy_lifecycle (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            current_stage TEXT NOT NULL,
            degradation_strikes INTEGER DEFAULT 0,
            s1_passed_at TEXT, s2_passed_at TEXT, s3_passed_at TEXT,
            s4_passed_at TEXT, s5_passed_at TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy_graveyard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            killed_at_stage TEXT,
            reason TEXT,
            best_sharpe REAL,
            total_trades INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            hash_id TEXT
        );

        CREATE TABLE IF NOT EXISTS cycle_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_num INTEGER,
            started_at TEXT, finished_at TEXT,
            duration_seconds REAL,
            ideas_generated INTEGER, backtests_run INTEGER,
            stage1_passed INTEGER, stage2_passed INTEGER,
            stage3_passed INTEGER, stage4_passed INTEGER,
            stage5_passed INTEGER,
            rejected INTEGER, errors INTEGER,
            best_sharpe REAL, best_strategy_name TEXT,
            gpu_used INTEGER DEFAULT 0, notes TEXT
        );

        CREATE TABLE IF NOT EXISTS system_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            component TEXT, event_type TEXT,
            severity TEXT, message TEXT, metadata TEXT
        );
    """)

    # Insert test data
    cur.execute("""
        INSERT INTO winning_strategies (strategy_name, archetype, symbol, interval,
            sharpe_ratio, total_return, net_profit, max_drawdown_pct, win_rate,
            profit_factor, total_trades, win_trades, loss_trades, avg_trade_pnl,
            quality_score, is_active, timestamp)
        VALUES ('TEST_ORB_30min', 'ORB', 'NQ', '5m',
            0.32, 0.52, 44687.0, 0.11, 53.7, 1.02, 745, 400, 345, 60.0,
            0.32, 1, '2026-02-17 17:38:21')
    """)

    cur.execute("""
        INSERT INTO equity_curves (strategy_id, curve_type, data_json)
        VALUES (1, 'backtest', '[{"date":"2011-01-01","equity":100000},{"date":"2025-12-31","equity":144687}]')
    """)

    cur.execute("""
        INSERT INTO cycle_log (cycle_num, started_at, finished_at, duration_seconds,
            ideas_generated, backtests_run, stage1_passed, stage2_passed,
            stage3_passed, stage4_passed, stage5_passed, rejected, errors,
            best_sharpe, best_strategy_name, gpu_used)
        VALUES (1, '2026-02-17 12:00:00', '2026-02-17 12:00:15', 15.0,
            10, 10, 3, 1, 1, 1, 1, 9, 0, 0.32, 'TEST_ORB_30min', 0)
    """)

    cur.execute("""
        INSERT INTO strategy_lifecycle (strategy_name, current_stage, s1_passed_at,
            s2_passed_at, s5_passed_at, created_at, updated_at)
        VALUES ('TEST_ORB_30min', 'STAGE5_PASS', '2026-02-17 12:00:05',
            '2026-02-17 12:00:08', '2026-02-17 12:00:14',
            '2026-02-17 12:00:00', '2026-02-17 12:00:14')
    """)

    cur.execute("""
        INSERT INTO strategy_graveyard (strategy_name, killed_at_stage, reason,
            best_sharpe, total_trades)
        VALUES ('DEAD_MA_10', 'STAGE1_FAIL', 'net_profit < 0', -0.05, 100)
    """)

    for i in range(5):
        cur.execute("""
            INSERT INTO backtest_runs (strategy_name, sharpe_ratio, net_profit,
                total_trades, profit_factor, max_drawdown_pct, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f'ORB_test_{i}', 0.1 + i * 0.05, 5000 + i * 1000,
              300 + i * 50, 1.0 + i * 0.01, 0.1, 50 + i))

    conn.commit()
    conn.close()
    return path


class TestMarcusQueryUnit(unittest.TestCase):
    """Unit tests against a controlled test database."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, 'test_marcus.db')
        create_test_db(cls.db_path)

    def _run_query(self, *args):
        """Run marcus-query.py with args and return parsed JSON."""
        cmd = [sys.executable, QUERY_SCRIPT, self.db_path] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        return json.loads(result.stdout)

    def test_dashboard_returns_all_keys(self):
        """Dashboard command returns all expected top-level keys."""
        data = self._run_query('dashboard')
        expected_keys = ['cycles', 'pipeline', 'pipeline_cumulative', 'leaderboard',
                         'winners', 'graveyard', 'stats', 'in_progress', 'archived',
                         'exploration']
        for key in expected_keys:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_winners_populated(self):
        """Winners array is populated with correct fields."""
        data = self._run_query('dashboard')
        self.assertGreater(len(data['winners']), 0, "No winners found")
        w = data['winners'][0]
        self.assertEqual(w['strategy_name'], 'TEST_ORB_30min')
        self.assertAlmostEqual(w['sharpe_ratio'], 0.32, places=2)
        self.assertTrue(w['has_equity_curve'], "Winner should have equity curve")

    def test_leaderboard_not_empty(self):
        """Leaderboard should have at least the winner."""
        data = self._run_query('dashboard')
        self.assertGreater(len(data['leaderboard']), 0, "Empty leaderboard")
        # First entry should be the winner
        self.assertEqual(data['leaderboard'][0]['source'], 'winner')

    def test_leaderboard_uses_timestamp_not_created_at(self):
        """REGRESSION: Leaderboard must use 'timestamp' column, aliased as created_at."""
        data = self._run_query('dashboard')
        lb = data['leaderboard']
        for row in lb:
            if row.get('source') == 'winner':
                self.assertIn('created_at', row)
                self.assertIsNotNone(row['created_at'])

    def test_pipeline_cumulative(self):
        """Pipeline cumulative counts from cycle_log."""
        data = self._run_query('dashboard')
        pc = data['pipeline_cumulative']
        self.assertEqual(pc['CANDIDATE'], 10)
        self.assertEqual(pc['STAGE1_PASS'], 3)
        self.assertEqual(pc['STAGE5_PASS'], 1)

    def test_pipeline_current_lifecycle(self):
        """Pipeline current from strategy_lifecycle."""
        data = self._run_query('dashboard')
        p = data['pipeline']
        self.assertEqual(p.get('STAGE5_PASS', 0), 1)

    def test_stats_aggregation(self):
        """Stats are correctly aggregated from cycle_log."""
        data = self._run_query('dashboard')
        s = data['stats']
        self.assertEqual(s['total_cycles'], 1)
        self.assertEqual(s['total_ideas'], 10)
        self.assertEqual(s['total_s1'], 3)
        self.assertEqual(s['total_s2'], 1)
        self.assertEqual(s['total_s5'], 1)
        self.assertIn('kill_rate_pct', s)
        self.assertIn('s1_pass_rate', s)
        self.assertIn('s2_pass_rate', s)

    def test_graveyard_populated(self):
        """Graveyard has failed strategies."""
        data = self._run_query('dashboard')
        gv = data['graveyard']
        self.assertGreater(len(gv), 0)
        self.assertEqual(gv[0]['strategy_name'], 'DEAD_MA_10')

    def test_in_progress_populated(self):
        """In-progress shows active lifecycle entries."""
        data = self._run_query('dashboard')
        ip = data['in_progress']
        self.assertGreater(len(ip), 0)

    def test_exploration_coverage(self):
        """Exploration coverage derived from strategy names."""
        data = self._run_query('dashboard')
        expl = data['exploration']
        self.assertGreater(len(expl), 0)
        # All our test data starts with ORB_ or TEST_
        archetypes = {e['archetype'] for e in expl}
        self.assertIn('ORB', archetypes)

    def test_winner_detail(self):
        """Winner detail endpoint returns full data + equity curve."""
        data = self._run_query('winner', '1')
        self.assertIsNotNone(data['winner'])
        self.assertEqual(data['winner']['strategy_name'], 'TEST_ORB_30min')
        self.assertIsNotNone(data['equity_curve'])
        ec = json.loads(data['equity_curve'])
        self.assertEqual(len(ec), 2)

    def test_history(self):
        """History endpoint returns cycle records."""
        data = self._run_query('history', '10')
        self.assertGreater(len(data['cycles']), 0)
        self.assertEqual(data['cycles'][0]['cycle_num'], 1)

    def test_no_db_returns_error(self):
        """Missing DB returns error JSON."""
        cmd = [sys.executable, QUERY_SCRIPT, '/nonexistent/path.db']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        self.assertIn('error', data)


class TestMarcusQueryEmptyDB(unittest.TestCase):
    """Edge case: completely empty database with tables but no data."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, 'empty_marcus.db')
        conn = sqlite3.connect(cls.db_path)
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, strategy_name TEXT, sharpe_ratio REAL, net_profit REAL, total_trades INTEGER, profit_factor REAL, max_drawdown_pct REAL, win_rate REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE winning_strategies (id INTEGER PRIMARY KEY, strategy_name TEXT, sharpe_ratio REAL, profit_factor REAL, max_drawdown_pct REAL, total_trades INTEGER, net_profit REAL, win_rate REAL, timestamp DATETIME, is_active INTEGER DEFAULT 1, archetype TEXT, symbol TEXT, interval TEXT, total_return REAL, max_drawdown REAL, win_trades INTEGER, loss_trades INTEGER, avg_trade_pnl REAL, params_json TEXT, source_code TEXT, pine_script TEXT, equity_curve_json TEXT, quality_score REAL, quality_notes TEXT, monte_carlo_var95 REAL, permutation_pvalue REAL, regime_analysis_json TEXT, data_range_start TEXT, data_range_end TEXT, notes TEXT, tags TEXT, priority INTEGER DEFAULT 0, hash_id TEXT);
            CREATE TABLE equity_curves (id INTEGER PRIMARY KEY, strategy_id INTEGER, curve_type TEXT, data_json TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE strategy_lifecycle (id INTEGER PRIMARY KEY, strategy_name TEXT, current_stage TEXT, degradation_strikes INTEGER DEFAULT 0, s1_passed_at TEXT, s2_passed_at TEXT, s3_passed_at TEXT, s4_passed_at TEXT, s5_passed_at TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP, notes TEXT);
            CREATE TABLE strategy_graveyard (id INTEGER PRIMARY KEY, strategy_name TEXT, killed_at_stage TEXT, reason TEXT, best_sharpe REAL, total_trades INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, hash_id TEXT);
            CREATE TABLE cycle_log (id INTEGER PRIMARY KEY, cycle_num INTEGER, started_at TEXT, finished_at TEXT, duration_seconds REAL, ideas_generated INTEGER, backtests_run INTEGER, stage1_passed INTEGER, stage2_passed INTEGER, stage3_passed INTEGER, stage4_passed INTEGER, stage5_passed INTEGER, rejected INTEGER, errors INTEGER, best_sharpe REAL, best_strategy_name TEXT, gpu_used INTEGER DEFAULT 0, notes TEXT);
        """)
        conn.commit()
        conn.close()

    def _run_query(self, *args):
        cmd = [sys.executable, QUERY_SCRIPT, self.db_path] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return json.loads(result.stdout)

    def test_empty_db_no_crash(self):
        """Empty DB returns valid JSON without crashing."""
        data = self._run_query('dashboard')
        self.assertEqual(len(data['winners']), 0)
        self.assertEqual(len(data['leaderboard']), 0)
        self.assertEqual(len(data['cycles']), 0)

    def test_empty_stats_valid(self):
        """Stats are zeroed out, not null."""
        data = self._run_query('dashboard')
        s = data['stats']
        self.assertEqual(s.get('total_cycles', 0), 0)

    def test_winner_detail_missing(self):
        """Winner detail for non-existent ID returns null."""
        data = self._run_query('winner', '999')
        self.assertIsNone(data['winner'])


class TestMarcusQueryIntegration(unittest.TestCase):
    """Integration tests against the REAL production database."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_DB):
            raise unittest.SkipTest("Real Marcus DB not found")

    def _run_query(self, *args):
        cmd = [sys.executable, QUERY_SCRIPT, REAL_DB] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        return json.loads(result.stdout)

    def test_real_db_dashboard(self):
        """Real DB returns complete dashboard data."""
        data = self._run_query('dashboard')
        self.assertIn('winners', data)
        self.assertIn('leaderboard', data)
        self.assertIn('stats', data)
        self.assertNotIn('error', data)

    def test_real_db_has_winners(self):
        """Real DB should have at least 1 winner (pipeline was verified working)."""
        data = self._run_query('dashboard')
        self.assertGreater(len(data['winners']), 0, "Expected at least 1 winner in real DB")

    def test_real_db_leaderboard_not_empty(self):
        """Real leaderboard should have entries."""
        data = self._run_query('dashboard')
        self.assertGreater(len(data['leaderboard']), 0, "Expected leaderboard entries")

    def test_real_db_pipeline_cumulative_flow(self):
        """Cumulative pipeline should show decreasing flow."""
        data = self._run_query('dashboard')
        pc = data['pipeline_cumulative']
        # More candidates than S1 passes, more S1 than S2, etc.
        self.assertGreaterEqual(pc.get('CANDIDATE', 0), pc.get('STAGE1_PASS', 0))
        self.assertGreaterEqual(pc.get('STAGE1_PASS', 0), pc.get('STAGE2_PASS', 0))

    def test_real_db_stats_accuracy(self):
        """Real stats should have reasonable values."""
        data = self._run_query('dashboard')
        s = data['stats']
        self.assertGreater(s.get('total_cycles', 0), 0)
        self.assertGreater(s.get('total_ideas', 0), 0)
        self.assertGreater(s.get('best_sharpe_ever', 0), 0)
        # Kill rate should be high (>90%)
        self.assertGreater(s.get('kill_rate_pct', 0), 80)

    def test_real_db_winner_equity_curve(self):
        """At least one winner should have an equity curve."""
        data = self._run_query('dashboard')
        winners = data['winners']
        has_equity = any(w.get('has_equity_curve') for w in winners)
        self.assertTrue(has_equity, "Expected at least 1 winner with equity curve")

    def test_real_db_winner_detail(self):
        """Winner detail endpoint works with real data."""
        data = self._run_query('dashboard')
        if data['winners']:
            wid = data['winners'][0]['id']
            detail = self._run_query('winner', str(wid))
            self.assertIsNotNone(detail['winner'])
            self.assertEqual(detail['winner']['id'], wid)

    def test_real_db_history(self):
        """History endpoint returns cycles."""
        data = self._run_query('history', '50')
        self.assertGreater(len(data['cycles']), 0)

    def test_real_db_exploration_coverage(self):
        """Exploration coverage shows at least ORB archetype."""
        data = self._run_query('dashboard')
        expl = data['exploration']
        self.assertGreater(len(expl), 0)
        archetypes = {e['archetype'] for e in expl}
        self.assertIn('ORB', archetypes, "Expected ORB in exploration data")


class TestAPIEndpoints(unittest.TestCase):
    """Test the HTTP API endpoints (requires serve-web.js running on port 3456)."""

    API_BASE = 'http://localhost:3456'

    @classmethod
    def setUpClass(cls):
        """Check if server is running."""
        import urllib.request
        try:
            urllib.request.urlopen(cls.API_BASE + '/api/marcus', timeout=3)
        except Exception:
            raise unittest.SkipTest("serve-web.js not running on port 3456")

    def _get(self, path):
        import urllib.request
        url = self.API_BASE + path
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())

    def _post(self, path, data):
        import urllib.request
        url = self.API_BASE + path
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def test_marcus_endpoint(self):
        """GET /api/marcus returns full dashboard data."""
        data = self._get('/api/marcus')
        self.assertIn('daemon', data)
        self.assertIn('stats', data)
        self.assertIn('winners', data)
        self.assertIn('leaderboard', data)

    def test_marcus_daemon_status(self):
        """Daemon status is one of RUNNING/SLOW/STALE/OFFLINE."""
        data = self._get('/api/marcus')
        self.assertIn(data['daemon']['status'], ['RUNNING', 'SLOW', 'STALE', 'OFFLINE'])

    def test_marcus_winner_endpoint(self):
        """GET /api/marcus/winner?id=N returns winner detail."""
        data = self._get('/api/marcus/winner?id=1')
        self.assertIn('winner', data)

    def test_marcus_history_endpoint(self):
        """GET /api/marcus/history returns cycles."""
        data = self._get('/api/marcus/history?limit=10')
        self.assertIn('cycles', data)

    def test_marcus_events_endpoint(self):
        """GET /api/marcus/events returns event array."""
        data = self._get('/api/marcus/events?limit=10')
        self.assertIn('events', data)

    def test_marcus_daemon_endpoint(self):
        """GET /api/marcus/daemon returns daemon info."""
        data = self._get('/api/marcus/daemon')
        self.assertIn('status', data)

    def test_marcus_command_endpoint(self):
        """POST /api/marcus/command accepts commands."""
        data = self._post('/api/marcus/command', {'command': 'get_directives'})
        self.assertTrue(data.get('ok'))
        self.assertIn('directives', data)

    def test_marcus_config_endpoint(self):
        """GET /api/marcus/config returns config."""
        data = self._get('/api/marcus/config')
        self.assertIn('config', data)
        self.assertIn('status', data)


class TestStageEndpoint(unittest.TestCase):
    """Tests for the new /api/marcus/stage drill-down endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, 'stage_test_marcus.db')
        create_test_db(cls.db_path)
        # Add additional lifecycle entries at various stages
        conn = sqlite3.connect(cls.db_path)
        cur = conn.cursor()
        for i in range(5):
            cur.execute("""
                INSERT INTO strategy_lifecycle (strategy_name, current_stage, created_at, updated_at)
                VALUES (?, 'STAGE1_PASS', datetime('now'), datetime('now'))
            """, (f'S1_strat_{i}',))
        for i in range(3):
            cur.execute("""
                INSERT INTO strategy_lifecycle (strategy_name, current_stage, created_at, updated_at)
                VALUES (?, 'STAGE2_PASS', datetime('now'), datetime('now'))
            """, (f'S2_strat_{i}',))
        for i in range(2):
            cur.execute("""
                INSERT INTO strategy_lifecycle (strategy_name, current_stage, created_at, updated_at, notes)
                VALUES (?, 'REJECTED', datetime('now'), datetime('now'), ?)
            """, (f'REJECTED_strat_{i}', 'Sharpe CI crosses zero'))
        conn.commit()
        conn.close()

    def _run_query(self, *args):
        cmd = [sys.executable, QUERY_SCRIPT, self.db_path] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        return json.loads(result.stdout)

    def test_stage_returns_expected_structure(self):
        """Stage command returns stage name, strategies array, count."""
        data = self._run_query('stage', 'STAGE1_PASS', '10')
        self.assertIn('stage', data)
        self.assertEqual(data['stage'], 'STAGE1_PASS')
        self.assertIn('strategies', data)
        self.assertIn('count', data)

    def test_stage_returns_correct_strategies(self):
        """Stage command returns strategies at the specified stage."""
        data = self._run_query('stage', 'STAGE1_PASS', '20')
        # We inserted 5 STAGE1_PASS strategies
        self.assertGreaterEqual(data['count'], 5)
        for s in data['strategies']:
            self.assertEqual(s['current_stage'], 'STAGE1_PASS')

    def test_stage_respects_limit(self):
        """Stage command respects the limit parameter."""
        data = self._run_query('stage', 'STAGE1_PASS', '2')
        self.assertLessEqual(len(data['strategies']), 2)

    def test_stage_empty_for_nonexistent(self):
        """Stage query for non-existent stage returns empty list."""
        data = self._run_query('stage', 'NONEXISTENT_STAGE', '10')
        self.assertEqual(data['count'], 0)
        self.assertEqual(len(data['strategies']), 0)

    def test_stage_strategies_have_lifecycle_fields(self):
        """Stage strategies include lifecycle fields."""
        data = self._run_query('stage', 'STAGE1_PASS', '5')
        if data['strategies']:
            s = data['strategies'][0]
            self.assertIn('strategy_name', s)
            self.assertIn('current_stage', s)
            self.assertIn('created_at', s)
            self.assertIn('updated_at', s)

    def test_stage_rejected_includes_notes(self):
        """Rejected strategies should include rejection notes."""
        data = self._run_query('stage', 'REJECTED', '10')
        self.assertGreater(data['count'], 0)
        # At least one should have notes
        notes_found = any(s.get('notes') for s in data['strategies'])
        self.assertTrue(notes_found, "Rejected strategies should have rejection notes")


class TestStatisticalAuditFields(unittest.TestCase):
    """Tests for statistical audit fields in winner data."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, 'audit_test_marcus.db')
        create_test_db(cls.db_path)
        # Update winner with statistical audit fields
        conn = sqlite3.connect(cls.db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE winning_strategies SET
                quality_notes = 'Passed all 5 gates | Sharpe CI: [0.150, 0.490] | Perm p=0.020 | DSR=0.970 | Robustness=72 | MC VaR95=-0.120',
                monte_carlo_var95 = -0.12,
                permutation_pvalue = 0.02
            WHERE id = 1
        """)
        conn.commit()
        conn.close()

    def _run_query(self, *args):
        cmd = [sys.executable, QUERY_SCRIPT, self.db_path] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, f"Script failed: {result.stderr}")
        return json.loads(result.stdout)

    def test_winner_has_quality_notes(self):
        """Winner should have quality_notes with statistical summary."""
        data = self._run_query('winner', '1')
        w = data['winner']
        self.assertIsNotNone(w.get('quality_notes'))
        self.assertIn('Sharpe CI', w['quality_notes'])
        self.assertIn('Perm p=', w['quality_notes'])
        self.assertIn('DSR=', w['quality_notes'])
        self.assertIn('Robustness=', w['quality_notes'])
        self.assertIn('MC VaR95=', w['quality_notes'])

    def test_winner_has_mc_var95(self):
        """Winner should have Monte Carlo VaR95 field."""
        data = self._run_query('winner', '1')
        w = data['winner']
        self.assertIsNotNone(w.get('monte_carlo_var95'))
        self.assertAlmostEqual(w['monte_carlo_var95'], -0.12, places=2)

    def test_winner_has_permutation_pvalue(self):
        """Winner should have permutation p-value field."""
        data = self._run_query('winner', '1')
        w = data['winner']
        self.assertIsNotNone(w.get('permutation_pvalue'))
        self.assertAlmostEqual(w['permutation_pvalue'], 0.02, places=2)

    def test_dashboard_winners_include_audit_fields(self):
        """Dashboard winners endpoint includes quality_notes."""
        data = self._run_query('dashboard')
        self.assertGreater(len(data['winners']), 0)
        w = data['winners'][0]
        self.assertIn('quality_notes', w)
        self.assertIn('monte_carlo_var95', w)


class TestStageAPIEndpoint(unittest.TestCase):
    """Test the HTTP /api/marcus/stage endpoint (requires serve-web.js running)."""

    API_BASE = 'http://localhost:3456'

    @classmethod
    def setUpClass(cls):
        import urllib.request
        try:
            urllib.request.urlopen(cls.API_BASE + '/api/marcus', timeout=3)
        except Exception:
            raise unittest.SkipTest("serve-web.js not running on port 3456")

    def _get(self, path):
        import urllib.request
        url = self.API_BASE + path
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())

    def test_stage_endpoint_returns_json(self):
        """GET /api/marcus/stage returns valid JSON."""
        data = self._get('/api/marcus/stage?stage=STAGE1_PASS&limit=5')
        self.assertIn('stage', data)
        self.assertIn('strategies', data)
        self.assertIn('count', data)

    def test_stage_endpoint_default_params(self):
        """GET /api/marcus/stage with no params uses defaults."""
        data = self._get('/api/marcus/stage')
        self.assertIn('stage', data)
        self.assertEqual(data['stage'], 'STAGE1_PASS')


if __name__ == '__main__':
    unittest.main(verbosity=2)
