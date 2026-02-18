"""
Marcus Live Dashboard
======================
Auto-refreshing HTML dashboard that reads from SQLite and generates a single
self-contained HTML file. Designed to be opened in any browser and auto-refresh.

Sections:
    1. Header Bar - Daemon status, heartbeat, GPU, uptime
    2. Pipeline Funnel - Visual: Ideas -> S1 -> S2 -> S3 -> S4 -> S5 -> Deployed
    3. Cycle History - Last 20 research cycles
    4. Strategy Leaderboard - Top active strategies
    5. Equity Curves - Plotly chart of top performers
    6. Disposal Log - Recent kills with reasons
    7. System Health - Recent events from monitor
    8. Research Progress - Cumulative stats
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .marcus_config import MarcusConfig
from .registry import StrategyRegistry
from .lifecycle import StrategyLifecycleManager
from .monitor import PipelineMonitor

logger = logging.getLogger(__name__)


class MarcusDashboard:
    """
    Generates the Marcus live HTML dashboard.
    Reads all data from SQLite (read-only). Daemon calls rebuild() periodically.
    """

    def __init__(self, config: MarcusConfig):
        self.config = config
        self.registry = StrategyRegistry(config.db_path)
        self.lifecycle = StrategyLifecycleManager(config.db_path, config)
        self.monitor = PipelineMonitor(config.logs_dir)

    def rebuild(self):
        """Generate complete dashboard HTML and write to disk."""
        try:
            data = self._gather_data()
            html = self._render(data)
            self._write(html)
            logger.info(f"Dashboard rebuilt: {self.config.dashboard_path}")
        except Exception as e:
            logger.error(f"Dashboard rebuild failed: {e}")

    # =========================================================================
    # Data Gathering
    # =========================================================================

    def _gather_data(self) -> Dict[str, Any]:
        """Query all dashboard data from SQLite and monitor."""
        return {
            'daemon_status': self._get_daemon_status(),
            'cycle_history': self.registry.get_cycle_history(limit=20),
            'pipeline_counts': self.registry.get_pipeline_counts(),
            'pipeline_status': self.lifecycle.get_pipeline_status(),
            'leaderboard': self._get_leaderboard(),
            'disposal_log': self.lifecycle.get_disposal_log(limit=20),
            'graveyard': self.registry.get_graveyard_entries(limit=20),
            'recent_events': self.monitor.get_recent_events(limit=20),
            'equity_curves': self._get_top_equity_curves(limit=5),
            'research_stats': self._get_research_stats(),
        }

    def _get_daemon_status(self) -> Dict[str, Any]:
        """Get daemon liveness and GPU status."""
        heartbeat = self.monitor.get_last_heartbeat()
        errors_24h = self.monitor.get_error_count(hours=24)

        # Parse heartbeat to check if alive
        status = "UNKNOWN"
        if heartbeat != "UNKNOWN":
            try:
                hb_time = datetime.fromisoformat(heartbeat)
                age_seconds = (datetime.now() - hb_time).total_seconds()
                if age_seconds < 600:  # 10 min
                    status = "RUNNING"
                elif age_seconds < 3600:  # 1 hour
                    status = "SLOW"
                else:
                    status = "STALE"
            except (ValueError, TypeError):
                status = "UNKNOWN"

        # GPU check
        gpu_status = "Unknown"
        try:
            from .accelerate import get_gpu_info, GPU_AVAILABLE
            if GPU_AVAILABLE:
                info = get_gpu_info()
                gpu_status = f"GPU Active ({info.get('device_name', 'CUDA')})"
            else:
                gpu_status = "CPU Only"
        except ImportError:
            gpu_status = "Module unavailable"

        return {
            'status': status,
            'heartbeat': heartbeat,
            'errors_24h': errors_24h,
            'gpu_status': gpu_status,
        }

    def _get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get top strategies for the leaderboard.

        P2-1: Falls back to top strategies from backtest_runs when
        winning_strategies is empty (before any strategy has passed S5).
        """
        try:
            df = self.registry.get_winning_strategies(limit=15, active_only=False)
            if not df.empty:
                return df.to_dict('records')

            # P2-1 FALLBACK: Use top strategies from backtest_runs
            logger.info("No winning strategies yet — falling back to top backtest_runs")
            df = self.registry.get_leaderboard(limit=15, min_trades=200)
            if not df.empty:
                return df.to_dict('records')
            return []
        except Exception as e:
            logger.error(f"Leaderboard query failed: {e}")
            return []

    def _get_top_equity_curves(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get equity curve data for top strategies (for Plotly chart)."""
        curves = []
        try:
            df = self.registry.get_winning_strategies(limit=limit, active_only=False)
            if df.empty:
                return curves

            for _, row in df.iterrows():
                strat_id = row.get('id')
                name = row.get('strategy_name', f'ID-{strat_id}')
                if strat_id is None:
                    continue

                eq = self.registry.get_equity_curve(int(strat_id))
                if eq is not None and not eq.empty and 'equity' in eq.columns:
                    # Convert to JSON-serializable format
                    dates = eq.index.astype(str).tolist() if hasattr(eq.index, 'astype') else list(range(len(eq)))
                    values = eq['equity'].tolist()
                    curves.append({
                        'name': name,
                        'dates': dates,
                        'values': values,
                    })
        except Exception as e:
            logger.error(f"Equity curve fetch failed: {e}")

        return curves

    def _get_research_stats(self) -> Dict[str, Any]:
        """Get cumulative research statistics."""
        cycles = self.registry.get_cycle_history(limit=1000)
        if not cycles:
            return {
                'total_cycles': 0, 'total_ideas': 0, 'total_backtests': 0,
                'total_s1_passed': 0, 'total_s2_passed': 0,
                'total_s5_passed': 0, 'total_rejected': 0, 'total_errors': 0,
                'kill_rate_pct': 100.0, 'avg_cycle_seconds': 0,
                'best_sharpe_ever': 0.0,
            }

        total_ideas = sum(c.get('ideas_generated', 0) for c in cycles)
        total_backtests = sum(c.get('backtests_run', 0) for c in cycles)
        total_s1 = sum(c.get('stage1_passed', 0) for c in cycles)
        total_s2 = sum(c.get('stage2_passed', 0) for c in cycles)
        total_s5 = sum(c.get('stage5_passed', 0) for c in cycles)
        total_rejected = sum(c.get('rejected', 0) for c in cycles)
        total_errors = sum(c.get('errors', 0) for c in cycles)
        durations = [c.get('duration_seconds', 0) for c in cycles if c.get('duration_seconds')]
        best_sharpe = max((c.get('best_sharpe', 0) or 0) for c in cycles) if cycles else 0

        kill_rate = 100.0
        if total_backtests > 0:
            kill_rate = (1.0 - total_s5 / total_backtests) * 100

        return {
            'total_cycles': len(cycles),
            'total_ideas': total_ideas,
            'total_backtests': total_backtests,
            'total_s1_passed': total_s1,
            'total_s2_passed': total_s2,
            'total_s5_passed': total_s5,
            'total_rejected': total_rejected,
            'total_errors': total_errors,
            'kill_rate_pct': kill_rate,
            'avg_cycle_seconds': sum(durations) / len(durations) if durations else 0,
            'best_sharpe_ever': best_sharpe,
        }

    # =========================================================================
    # HTML Rendering
    # =========================================================================

    def _render(self, data: Dict[str, Any]) -> str:
        """Render the complete dashboard HTML."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = data['daemon_status']
        stats = data['research_stats']
        pipeline = data['pipeline_counts']

        # Status color
        status_color = {
            'RUNNING': '#00ff41', 'SLOW': '#ffaa00',
            'STALE': '#ff4444', 'UNKNOWN': '#888888',
        }.get(status['status'], '#888888')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="300">
    <title>MARCUS - Autonomous Research Agent</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Consolas', 'Courier New', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #00ff41;
        }}
        .header h1 {{
            font-size: 24px;
            color: #00ff41;
            text-shadow: 0 0 10px rgba(0,255,65,0.3);
        }}
        .header .subtitle {{ color: #888; font-size: 12px; }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .header-stats {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}
        .header-stat {{
            text-align: center;
        }}
        .header-stat .value {{
            font-size: 18px;
            font-weight: bold;
        }}
        .header-stat .label {{
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
        }}

        /* Grid layout */
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            padding: 20px;
        }}
        .grid-full {{
            grid-column: 1 / -1;
        }}

        /* Cards */
        .card {{
            background: #141414;
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
        }}
        .card-header {{
            background: #1a1a1a;
            padding: 12px 16px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-header h2 {{
            font-size: 14px;
            color: #00ff41;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .card-header .badge {{
            background: #333;
            color: #aaa;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
        }}
        .card-body {{
            padding: 16px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        th {{
            background: #1a1a1a;
            color: #00ff41;
            padding: 8px 10px;
            text-align: left;
            font-weight: normal;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 6px 10px;
            border-bottom: 1px solid #222;
        }}
        tr:hover td {{
            background: #1a1a1a;
        }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .pass {{ color: #00ff41; }}
        .fail {{ color: #ff4444; }}
        .warn {{ color: #ffaa00; }}
        .muted {{ color: #666; }}

        /* Funnel */
        .funnel {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            padding: 20px 10px;
            flex-wrap: wrap;
        }}
        .funnel-stage {{
            text-align: center;
            padding: 12px 8px;
            border-radius: 6px;
            min-width: 80px;
        }}
        .funnel-stage .count {{
            font-size: 24px;
            font-weight: bold;
        }}
        .funnel-stage .label {{
            font-size: 10px;
            text-transform: uppercase;
            margin-top: 4px;
        }}
        .funnel-arrow {{
            color: #444;
            font-size: 20px;
        }}

        /* Scrollable card body */
        .scroll-body {{
            max-height: 350px;
            overflow-y: auto;
        }}
        .scroll-body::-webkit-scrollbar {{
            width: 6px;
        }}
        .scroll-body::-webkit-scrollbar-thumb {{
            background: #333;
            border-radius: 3px;
        }}

        /* Event log */
        .event {{
            padding: 6px 10px;
            border-bottom: 1px solid #1a1a1a;
            font-size: 11px;
            display: flex;
            gap: 10px;
        }}
        .event .ts {{ color: #555; min-width: 140px; }}
        .event .comp {{ color: #00aaff; min-width: 100px; }}
        .event.error {{ background: rgba(255,0,0,0.05); }}
        .event.error .msg {{ color: #ff6666; }}

        /* Stats grid */
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }}
        .stat-box {{
            background: #1a1a1a;
            padding: 14px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #222;
        }}
        .stat-box .value {{
            font-size: 22px;
            font-weight: bold;
            color: #00ff41;
        }}
        .stat-box .label {{
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            margin-top: 4px;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 12px;
            color: #444;
            font-size: 11px;
            border-top: 1px solid #222;
        }}
    </style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div>
        <h1>MARCUS</h1>
        <div class="subtitle">Autonomous Quant Research Agent</div>
    </div>
    <div class="header-stats">
        <div class="header-stat">
            <div class="value" style="color: {status_color};">
                <span class="status-badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color};">{status['status']}</span>
            </div>
            <div class="label">Daemon</div>
        </div>
        <div class="header-stat">
            <div class="value" style="color: #00aaff;">{status['gpu_status'].split('(')[0].strip()}</div>
            <div class="label">Compute</div>
        </div>
        <div class="header-stat">
            <div class="value">{stats['total_cycles']}</div>
            <div class="label">Cycles</div>
        </div>
        <div class="header-stat">
            <div class="value" style="color: {'#ff4444' if status['errors_24h'] > 0 else '#00ff41'};">{status['errors_24h']}</div>
            <div class="label">Errors (24h)</div>
        </div>
        <div class="header-stat">
            <div class="value">{stats.get('best_sharpe_ever', 0):.2f}</div>
            <div class="label">Best Sharpe</div>
        </div>
    </div>
</div>

<div class="grid">

    <!-- PIPELINE FUNNEL -->
    <div class="card grid-full">
        <div class="card-header">
            <h2>Pipeline Funnel</h2>
            <span class="badge">Kill Rate: {stats['kill_rate_pct']:.1f}%</span>
        </div>
        <div class="card-body">
            {self._render_funnel(pipeline, data['pipeline_status'])}
        </div>
    </div>

    <!-- RESEARCH STATS -->
    <div class="card grid-full">
        <div class="card-header">
            <h2>Research Progress</h2>
            <span class="badge">Cumulative</span>
        </div>
        <div class="card-body">
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="value">{stats['total_ideas']}</div>
                    <div class="label">Ideas Generated</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats['total_backtests']}</div>
                    <div class="label">Backtests Run</div>
                </div>
                <div class="stat-box">
                    <div class="value" style="color: #00ff41;">{stats['total_s5_passed']}</div>
                    <div class="label">Stage 5 Passed</div>
                </div>
                <div class="stat-box">
                    <div class="value" style="color: #ff4444;">{stats['total_rejected']}</div>
                    <div class="label">Rejected</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats['total_s1_passed']}</div>
                    <div class="label">S1 Passed</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats['total_s2_passed']}</div>
                    <div class="label">S2 Passed</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats.get('total_errors', 0)}</div>
                    <div class="label">Total Errors</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats['avg_cycle_seconds']:.0f}s</div>
                    <div class="label">Avg Cycle Time</div>
                </div>
            </div>
        </div>
    </div>

    <!-- EQUITY CURVES -->
    <div class="card grid-full">
        <div class="card-header">
            <h2>Top Strategy Equity Curves</h2>
            <span class="badge">{len(data['equity_curves'])} curves</span>
        </div>
        <div class="card-body">
            <div id="equity-chart" style="height: 350px;"></div>
        </div>
    </div>

    <!-- STRATEGY LEADERBOARD -->
    <div class="card">
        <div class="card-header">
            <h2>Strategy Leaderboard</h2>
            <span class="badge">{len(data['leaderboard'])} strategies</span>
        </div>
        <div class="card-body scroll-body">
            {self._render_leaderboard(data['leaderboard'])}
        </div>
    </div>

    <!-- CYCLE HISTORY -->
    <div class="card">
        <div class="card-header">
            <h2>Cycle History</h2>
            <span class="badge">Last {len(data['cycle_history'])}</span>
        </div>
        <div class="card-body scroll-body">
            {self._render_cycle_history(data['cycle_history'])}
        </div>
    </div>

    <!-- DISPOSAL LOG -->
    <div class="card">
        <div class="card-header">
            <h2>Disposal Log</h2>
            <span class="badge">{len(data['disposal_log'])} recent kills</span>
        </div>
        <div class="card-body scroll-body">
            {self._render_disposal_log(data['disposal_log'])}
        </div>
    </div>

    <!-- SYSTEM HEALTH -->
    <div class="card">
        <div class="card-header">
            <h2>System Health</h2>
            <span class="badge">Last {len(data['recent_events'])} events</span>
        </div>
        <div class="card-body scroll-body">
            {self._render_events(data['recent_events'])}
        </div>
    </div>

</div>

<!-- FOOTER -->
<div class="footer">
    MARCUS Autonomous Research Agent | Dashboard generated {now} | Auto-refresh every 5 minutes
</div>

<!-- PLOTLY CHART SCRIPT -->
<script>
{self._render_equity_chart_js(data['equity_curves'])}
</script>

</body>
</html>"""

    # =========================================================================
    # Component Renderers
    # =========================================================================

    def _render_funnel(self, pipeline: Dict, status: Dict) -> str:
        """Render the pipeline funnel visualization."""
        stages = [
            ('Ideas', pipeline.get('total_tested', 0), '#888888'),
            ('S1', status.get('STAGE1_PASS', 0), '#44aaff'),
            ('S2', status.get('STAGE2_PASS', 0), '#00aaff'),
            ('S3', status.get('STAGE3_PASS', 0), '#00ccaa'),
            ('S4', status.get('STAGE4_PASS', 0), '#88cc00'),
            ('S5', status.get('STAGE5_PASS', 0), '#00ff41'),
            ('Deployed', status.get('DEPLOYED', 0), '#ffaa00'),
            ('Graveyard', pipeline.get('graveyard', 0), '#ff4444'),
        ]

        html = '<div class="funnel">'
        for i, (label, count, color) in enumerate(stages):
            if i > 0 and i < len(stages) - 1:
                html += '<span class="funnel-arrow">&rarr;</span>'
            elif i == len(stages) - 1:
                html += '<span class="funnel-arrow" style="margin-left:20px;">|</span>'

            html += f"""
            <div class="funnel-stage" style="background: {color}11; border: 1px solid {color}44;">
                <div class="count" style="color: {color};">{count}</div>
                <div class="label" style="color: {color}99;">{label}</div>
            </div>"""

        html += '</div>'
        return html

    def _render_leaderboard(self, strategies: List[Dict]) -> str:
        """Render the strategy leaderboard table."""
        if not strategies:
            return '<div style="color: #555; text-align: center; padding: 20px;">No strategies yet. Marcus is working on it...</div>'

        html = """<table>
        <tr>
            <th>#</th><th>Strategy</th><th>Arch</th>
            <th class="num">Sharpe</th><th class="num">PF</th>
            <th class="num">WR%</th><th class="num">DD%</th>
            <th class="num">Trades</th><th class="num">Net $</th>
        </tr>"""

        for i, s in enumerate(strategies):
            sharpe = float(s.get('sharpe_ratio', 0) or 0)
            pf = float(s.get('profit_factor', 0) or 0)
            wr = float(s.get('win_rate', 0) or 0)  # Already stored as percentage (e.g. 49.5)
            dd = abs(float(s.get('max_drawdown', 0) or 0)) * 100
            trades = int(s.get('total_trades', 0) or 0)
            net = float(s.get('net_profit', s.get('ending_equity', 100000) - 100000) or 0)
            name = str(s.get('strategy_name', ''))[:25]
            arch = str(s.get('archetype', s.get('regime', '')))[:10]

            sharpe_class = 'pass' if sharpe >= 1.0 else ('warn' if sharpe >= 0.5 else 'fail')
            net_class = 'pass' if net > 0 else 'fail'

            html += f"""
            <tr>
                <td class="muted">{i + 1}</td>
                <td>{name}</td>
                <td class="muted">{arch}</td>
                <td class="num {sharpe_class}">{sharpe:.2f}</td>
                <td class="num">{pf:.2f}</td>
                <td class="num">{wr:.1f}</td>
                <td class="num">{dd:.1f}</td>
                <td class="num">{trades}</td>
                <td class="num {net_class}">${net:,.0f}</td>
            </tr>"""

        html += '</table>'
        return html

    def _render_cycle_history(self, cycles: List[Dict]) -> str:
        """Render the cycle history table."""
        if not cycles:
            return '<div style="color: #555; text-align: center; padding: 20px;">No cycles executed yet.</div>'

        html = """<table>
        <tr>
            <th>#</th><th>Time</th><th class="num">Ideas</th>
            <th class="num">S1</th><th class="num">S2</th><th class="num">S5</th>
            <th class="num">Rej</th><th class="num">Err</th>
            <th class="num">Best</th><th class="num">Dur</th>
        </tr>"""

        for c in cycles:
            cycle_num = c.get('cycle_num', '?')
            started = str(c.get('started_at', ''))[:16]
            ideas = c.get('ideas_generated', 0)
            s1 = c.get('stage1_passed', 0)
            s2 = c.get('stage2_passed', 0)
            s5 = c.get('stage5_passed', 0)
            rej = c.get('rejected', 0)
            err = c.get('errors', 0)
            best = float(c.get('best_sharpe', 0) or 0)
            dur = float(c.get('duration_seconds', 0) or 0)

            err_class = 'fail' if err > 0 else ''
            s5_class = 'pass' if s5 > 0 else ''

            html += f"""
            <tr>
                <td class="muted">{cycle_num}</td>
                <td class="muted">{started}</td>
                <td class="num">{ideas}</td>
                <td class="num">{s1}</td>
                <td class="num">{s2}</td>
                <td class="num {s5_class}">{s5}</td>
                <td class="num">{rej}</td>
                <td class="num {err_class}">{err}</td>
                <td class="num">{best:.2f}</td>
                <td class="num">{dur:.0f}s</td>
            </tr>"""

        html += '</table>'
        return html

    def _render_disposal_log(self, disposals: List[Dict]) -> str:
        """Render the disposal/graveyard log."""
        if not disposals:
            return '<div style="color: #555; text-align: center; padding: 20px;">No disposals yet.</div>'

        html = """<table>
        <tr>
            <th>Strategy</th><th>Stage</th><th>Reason</th><th>Date</th>
        </tr>"""

        for d in disposals:
            name = str(d.get('strategy_name', ''))[:25]
            stage = d.get('killed_at_stage', d.get('stage', ''))
            reason = str(d.get('reason', ''))[:50]
            date = str(d.get('created_at', d.get('disposed_at', '')))[:16]

            html += f"""
            <tr>
                <td>{name}</td>
                <td class="fail">{stage}</td>
                <td class="muted">{reason}</td>
                <td class="muted">{date}</td>
            </tr>"""

        html += '</table>'
        return html

    def _render_events(self, events: List[Dict]) -> str:
        """Render the system health event log."""
        if not events:
            return '<div style="color: #555; text-align: center; padding: 20px;">No events recorded.</div>'

        html = ''
        for evt in events:
            ts = str(evt.get('timestamp', ''))[:19]
            comp = evt.get('component', '')
            msg = str(evt.get('message', ''))[:80]
            severity = evt.get('severity', 'INFO')

            css_class = 'error' if severity in ('ERROR', 'CRITICAL') else ''
            msg_style = ''
            if severity == 'WARNING':
                msg_style = 'color: #ffaa00;'
            elif severity in ('ERROR', 'CRITICAL'):
                msg_style = 'color: #ff6666;'

            html += f"""
            <div class="event {css_class}">
                <span class="ts">{ts}</span>
                <span class="comp">{comp}</span>
                <span class="msg" style="{msg_style}">{msg}</span>
            </div>"""

        return html

    def _render_equity_chart_js(self, curves: List[Dict]) -> str:
        """Generate Plotly JavaScript for equity curves chart."""
        if not curves:
            return """
            document.getElementById('equity-chart').innerHTML =
                '<div style="color: #555; text-align: center; padding: 80px 0;">No equity curves available yet.</div>';
            """

        colors = ['#00ff41', '#00aaff', '#ffaa00', '#ff4444', '#aa44ff']
        traces = []

        for i, curve in enumerate(curves):
            color = colors[i % len(colors)]
            name_js = json.dumps(curve['name'])
            dates_js = json.dumps(curve['dates'])
            values_js = json.dumps(curve['values'])

            traces.append(f"""{{
                x: {dates_js},
                y: {values_js},
                type: 'scatter',
                mode: 'lines',
                name: {name_js},
                line: {{ color: '{color}', width: 1.5 }}
            }}""")

        traces_str = ',\n'.join(traces)

        return f"""
        var data = [{traces_str}];
        var layout = {{
            paper_bgcolor: '#141414',
            plot_bgcolor: '#141414',
            font: {{ color: '#888', family: 'Consolas, monospace', size: 11 }},
            margin: {{ l: 60, r: 20, t: 10, b: 40 }},
            xaxis: {{
                gridcolor: '#222',
                linecolor: '#333',
            }},
            yaxis: {{
                gridcolor: '#222',
                linecolor: '#333',
                tickprefix: '$',
            }},
            legend: {{
                x: 0.01, y: 0.99,
                bgcolor: 'rgba(20,20,20,0.8)',
                bordercolor: '#333',
                borderwidth: 1,
                font: {{ size: 10 }}
            }},
            hovermode: 'x unified',
        }};
        Plotly.newPlot('equity-chart', data, layout, {{responsive: true, displayModeBar: false}});
        """

    # =========================================================================
    # File Output
    # =========================================================================

    def _write(self, html: str):
        """Write HTML dashboard to disk."""
        os.makedirs(os.path.dirname(self.config.dashboard_path), exist_ok=True)
        with open(self.config.dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html)


# =============================================================================
# Standalone Entry Point
# =============================================================================

if __name__ == "__main__":
    config = MarcusConfig.default()
    dashboard = MarcusDashboard(config)
    dashboard.rebuild()
    print(f"Dashboard generated: {config.dashboard_path}")
