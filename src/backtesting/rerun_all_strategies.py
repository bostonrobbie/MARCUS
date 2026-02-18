"""
Marcus Strategy Re-Tester
=========================
Stops daemon, backs up DB, reruns ALL strategy archetypes through the
validated engine, generates before/after comparison report.

Usage:
    python -m backtesting.rerun_all_strategies
    python -m backtesting.rerun_all_strategies --dry-run      # Show what would be tested, don't run
    python -m backtesting.rerun_all_strategies --archetypes orb_breakout ma_crossover  # Test specific types
    python -m backtesting.rerun_all_strategies --max-ideas 100  # Limit total ideas tested
"""

import os
import sys
import time
import shutil
import signal
import sqlite3
import argparse
import traceback
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.marcus_config import MarcusConfig
from backtesting.registry import StrategyRegistry
from backtesting.lifecycle import StrategyLifecycleManager
from backtesting.vector_engine import VectorEngine, VectorizedNQORB, VectorizedMA
from backtesting.stage2_rigorous_backtest import RigorousBacktester, StrategyMapper
from backtesting.research_engine import AutonomousResearchEngine
from backtesting.monitor import PipelineMonitor


def stop_daemon(config: MarcusConfig):
    """Stop the Marcus daemon if running."""
    print("[1/8] Stopping Marcus daemon...")
    state_file = config.state_file
    if os.path.exists(state_file):
        try:
            import json
            with open(state_file) as f:
                state = json.load(f)
            print(f"  Last heartbeat: {state.get('last_heartbeat_at', 'unknown')}")
            print(f"  Total cycles: {state.get('total_cycles', 0)}")
        except Exception:
            pass

    # Try to kill any running daemon process
    try:
        import subprocess
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True, timeout=10
        )
        # Look for marcus_daemon processes
        for line in result.stdout.split('\n'):
            if 'marcus_daemon' in line.lower():
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    pid = parts[1].strip('"')
                    print(f"  Killing daemon PID {pid}")
                    subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, timeout=10)
    except Exception as e:
        print(f"  Could not check for daemon process: {e}")
    print("  Done.")


def backup_db(config: MarcusConfig) -> str:
    """Back up current DB. Returns backup path."""
    print("[2/8] Backing up database...")
    db_path = config.db_path
    if not os.path.exists(db_path):
        print(f"  No DB found at {db_path}")
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.replace('.db', f'_backup_{timestamp}.db')
    shutil.copy2(db_path, backup_path)
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    print(f"  Backed up to: {backup_path} ({size_mb:.1f} MB)")
    return backup_path


def export_current_results(config: MarcusConfig) -> dict:
    """Export current backtest results for comparison."""
    print("[3/8] Exporting current results...")
    db_path = config.db_path
    if not os.path.exists(db_path):
        print("  No DB to export from.")
        return {'strategies': {}, 'summary': {}}

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM backtest_runs", conn)
    except Exception:
        print("  No backtest_runs table found.")
        return {'strategies': {}, 'summary': {}}
    finally:
        conn.close()

    if df.empty:
        print("  No existing backtest results.")
        return {'strategies': {}, 'summary': {}}

    results = {}
    for _, row in df.iterrows():
        name = row.get('strategy_name', row.get('name', 'unknown'))
        results[name] = {
            'sharpe': row.get('sharpe_ratio', row.get('sharpe', 0)),
            'net_profit': row.get('net_profit', 0),
            'max_drawdown': row.get('max_drawdown', 0),
            'total_trades': row.get('total_trades', row.get('trades', 0)),
            'quality_passed': row.get('quality_passed', False),
        }

    profitable = sum(1 for v in results.values() if v['net_profit'] > 0)
    avg_sharpe = np.mean([v['sharpe'] for v in results.values()]) if results else 0
    best_sharpe = max((v['sharpe'] for v in results.values()), default=0)

    summary = {
        'total': len(results),
        'profitable': profitable,
        'avg_sharpe': avg_sharpe,
        'best_sharpe': best_sharpe,
    }
    print(f"  Exported {len(results)} strategies (profitable: {profitable}, best Sharpe: {best_sharpe:.4f})")

    return {'strategies': results, 'summary': summary}


def clear_runs(config: MarcusConfig):
    """Clear backtest_runs table (keep graveyard/lifecycle for reference)."""
    print("[4/8] Clearing backtest_runs table...")
    conn = sqlite3.connect(config.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM backtest_runs")
        deleted = cursor.rowcount
        conn.commit()
        print(f"  Cleared {deleted} old runs.")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        conn.close()


def generate_all_ideas(config: MarcusConfig, archetypes=None, max_ideas=None) -> list:
    """Generate ALL strategy ideas using the research engine fallback grid."""
    print("[5/8] Generating strategy ideas...")
    engine = AutonomousResearchEngine(config)
    all_ideas = engine._fallback_ideas(99999)  # Get all possible combos

    if archetypes:
        all_ideas = [i for i in all_ideas if i.get('archetype', '') in archetypes]
        print(f"  Filtered to archetypes: {archetypes}")

    if max_ideas and len(all_ideas) > max_ideas:
        # Shuffle and take first N for balanced sampling
        import random
        random.shuffle(all_ideas)
        all_ideas = all_ideas[:max_ideas]

    # Count by archetype
    arch_counts = defaultdict(int)
    for idea in all_ideas:
        arch_counts[idea.get('archetype', 'unknown')] += 1

    print(f"  Generated {len(all_ideas)} ideas:")
    for arch, count in sorted(arch_counts.items()):
        print(f"    {arch}: {count}")

    return all_ideas


def run_backtests(config: MarcusConfig, ideas: list, data: pd.DataFrame) -> list:
    """Run backtests for all ideas."""
    print(f"[6/8] Running {len(ideas)} backtests...")
    results = []
    passed = 0
    errors = 0
    profitable = 0
    best_sharpe = -999
    best_name = ""

    registry = StrategyRegistry(config.db_path)
    lifecycle = StrategyLifecycleManager(config.db_path, config)
    mapper = StrategyMapper()
    backtester = RigorousBacktester(config, registry)

    start_time = time.time()

    # Pre-compute data range once
    data_range = (str(data.index[0].date()), str(data.index[-1].date()))

    for i, idea in enumerate(ideas):
        try:
            # Create strategy
            strategy = mapper.create_vector_strategy(idea)
            if strategy is None:
                continue

            strategy_name = idea.get('strategy_name', 'unknown')

            # Run backtest
            engine = VectorEngine(
                strategy,
                initial_capital=100000,
                commission=2.06,
                slippage=5.0,
                point_value=20.0,
            )
            result = engine.run(data)

            # Extract metrics
            metrics = backtester._extract_metrics(result, strategy_name)
            metrics['strategy_name'] = strategy_name
            metrics['archetype'] = idea.get('archetype', '')

            # Determine regime
            net_pnl = metrics.get('net_profit', 0)
            if net_pnl > 0:
                regime = "STAGE1_PASS"
            else:
                regime = "STAGE1_FAIL"
            notes = f"Net profit ${net_pnl:,.0f}" if net_pnl else ""

            # Build params dict from idea
            params = {k: v for k, v in idea.items()
                      if k not in ('strategy_name', 'archetype', 'description')}

            # Save to DB with correct signature
            try:
                registry.save_run(
                    strategy_name=strategy_name,
                    symbol='NQ',
                    interval='5m',
                    params=params,
                    stats=metrics,  # _extract_metrics returns snake_case keys, save_run handles both
                    data_range=data_range,
                    regime=regime,
                    notes=notes,
                )
            except Exception as e:
                if errors < 3:
                    print(f"  DB SAVE ERROR on {strategy_name}: {e}")

            # P1-4: Create lifecycle entry for batch runs
            try:
                import hashlib, json as _json
                idea_hash = hashlib.sha256(_json.dumps({
                    'archetype': idea.get('archetype', ''),
                    'variant': idea.get('variant', ''),
                    'params': idea.get('params', {}),
                }, sort_keys=True, default=str).encode()).hexdigest()
                lc_id = lifecycle.register_candidate(
                    idea_hash, strategy_name, idea.get('archetype', ''))
                lifecycle.set_testing(lc_id)
                if regime == "STAGE1_PASS":
                    lifecycle.promote(lc_id, 'STAGE1_PASS', {
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'net_profit': metrics.get('net_profit', 0),
                    })
                else:
                    lifecycle.reject(lc_id,
                                     f"Net profit ${net_pnl:,.0f}", 'TESTING')
            except Exception as e:
                pass  # Non-critical; don't break batch run

            results.append(metrics)

            # Track stats
            net_pnl = metrics.get('net_profit', 0)
            sharpe = metrics.get('sharpe_ratio', metrics.get('sharpe', 0))
            if net_pnl > 0:
                profitable += 1
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_name = metrics['strategy_name']

            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == len(ideas):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(ideas) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(ideas)}] Profitable: {profitable}, Best Sharpe: {best_sharpe:.4f} ({best_name}) "
                      f"| {rate:.1f} strats/sec, ETA: {eta:.0f}s")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on {idea.get('strategy_name', '?')}: {e}")
            elif errors == 6:
                print(f"  ... suppressing further error messages")

    elapsed = time.time() - start_time
    print(f"\n  Completed {len(results)} backtests in {elapsed:.1f}s ({errors} errors)")
    print(f"  Profitable: {profitable}/{len(results)}")
    print(f"  Best Sharpe: {best_sharpe:.4f} ({best_name})")

    return results


def generate_comparison_report(old_results: dict, new_results: list, output_path: str = None):
    """Generate before/after comparison report."""
    print("[7/8] Generating comparison report...")

    old_strats = old_results.get('strategies', {})
    old_summary = old_results.get('summary', {})

    # New summary
    new_profitable = sum(1 for r in new_results if r.get('net_profit', 0) > 0)
    new_sharpes = [r.get('sharpe_ratio', r.get('sharpe', 0)) for r in new_results]
    new_avg_sharpe = np.mean(new_sharpes) if new_sharpes else 0
    new_best_sharpe = max(new_sharpes) if new_sharpes else 0

    report = []
    report.append("=" * 70)
    report.append("STRATEGY RE-TEST COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    # Summary table
    report.append(f"{'':>30} {'OLD':>12} {'NEW':>12} {'DELTA':>12}")
    report.append(f"{'-'*66}")

    old_total = old_summary.get('total', 0)
    new_total = len(new_results)
    report.append(f"{'Total strategies':>30} {old_total:>12} {new_total:>12} {new_total - old_total:>+12}")

    old_prof = old_summary.get('profitable', 0)
    report.append(f"{'Profitable strategies':>30} {old_prof:>12} {new_profitable:>12} {new_profitable - old_prof:>+12}")

    old_avg = old_summary.get('avg_sharpe', 0)
    report.append(f"{'Avg Sharpe':>30} {old_avg:>12.4f} {new_avg_sharpe:>12.4f} {new_avg_sharpe - old_avg:>+12.4f}")

    old_best = old_summary.get('best_sharpe', 0)
    report.append(f"{'Best Sharpe':>30} {old_best:>12.4f} {new_best_sharpe:>12.4f} {new_best_sharpe - old_best:>+12.4f}")

    pass_old = old_summary.get('profitable', 0) / max(old_total, 1) * 100
    pass_new = new_profitable / max(new_total, 1) * 100
    report.append(f"{'Profit rate %':>30} {pass_old:>11.1f}% {pass_new:>11.1f}% {pass_new - pass_old:>+11.1f}%")

    # Archetype breakdown
    report.append("")
    report.append("--- BY ARCHETYPE ---")
    arch_stats = defaultdict(lambda: {'count': 0, 'profitable': 0, 'best_sharpe': -999, 'best_name': ''})
    for r in new_results:
        arch = r.get('archetype', 'unknown')
        arch_stats[arch]['count'] += 1
        if r.get('net_profit', 0) > 0:
            arch_stats[arch]['profitable'] += 1
        sharpe = r.get('sharpe_ratio', r.get('sharpe', 0))
        if sharpe > arch_stats[arch]['best_sharpe']:
            arch_stats[arch]['best_sharpe'] = sharpe
            arch_stats[arch]['best_name'] = r.get('strategy_name', '')

    report.append(f"{'Archetype':>25} {'Count':>8} {'Profitable':>12} {'Rate':>8} {'Best Sharpe':>14}")
    for arch, stats in sorted(arch_stats.items()):
        rate = stats['profitable'] / max(stats['count'], 1) * 100
        report.append(f"{arch:>25} {stats['count']:>8} {stats['profitable']:>12} {rate:>7.1f}% {stats['best_sharpe']:>14.4f}")

    # Strategies with significant change
    if old_strats:
        report.append("")
        report.append("--- STRATEGIES WITH >20% SHARPE CHANGE ---")
        changes = []
        for r in new_results:
            name = r.get('strategy_name', '')
            if name in old_strats:
                old_s = old_strats[name].get('sharpe', 0)
                new_s = r.get('sharpe_ratio', r.get('sharpe', 0))
                if old_s != 0:
                    pct_change = (new_s - old_s) / abs(old_s) * 100
                    if abs(pct_change) > 20:
                        changes.append((name, old_s, new_s, pct_change))

        if changes:
            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            report.append(f"{'Strategy':>45} {'Old Sharpe':>12} {'New Sharpe':>12} {'Change':>10}")
            for name, old_s, new_s, pct in changes[:20]:
                report.append(f"{name:>45} {old_s:>12.4f} {new_s:>12.4f} {pct:>+9.1f}%")
        else:
            report.append("  No strategies with >20% change found.")

    # New winners
    report.append("")
    report.append("--- TOP 20 NEW WINNERS (by Sharpe) ---")
    sorted_new = sorted(new_results, key=lambda r: r.get('sharpe_ratio', r.get('sharpe', 0)), reverse=True)
    report.append(f"{'Strategy':>50} {'Sharpe':>10} {'Profit':>12} {'Trades':>8} {'DD':>10}")
    for r in sorted_new[:20]:
        name = r.get('strategy_name', '')
        sharpe = r.get('sharpe_ratio', r.get('sharpe', 0))
        profit = r.get('net_profit', 0)
        trades = r.get('total_trades', r.get('trades', 0))
        dd = r.get('max_drawdown', 0)
        report.append(f"{name:>50} {sharpe:>10.4f} ${profit:>11,.0f} {trades:>8} {dd:>9.2%}")

    report.append("")
    report.append("=" * 70)

    full_report = "\n".join(report)
    print(full_report)

    # Save to file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report)
        print(f"\nReport saved to: {output_path}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="Marcus Strategy Re-Tester")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be tested without running')
    parser.add_argument('--archetypes', nargs='+', default=None,
                        help='Only test specific archetypes (e.g., orb_breakout ma_crossover)')
    parser.add_argument('--max-ideas', type=int, default=None, help='Max ideas to test')
    parser.add_argument('--skip-daemon-stop', action='store_true', help='Skip daemon stop/restart')
    parser.add_argument('--report-path', default=None, help='Output path for comparison report')
    args = parser.parse_args()

    config = MarcusConfig()

    print("=" * 70)
    print("MARCUS STRATEGY RE-TESTER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Step 1: Stop daemon
    if not args.skip_daemon_stop:
        stop_daemon(config)

    # Step 2: Export old results (before any modifications)
    old_results = export_current_results(config)

    # Step 3: Generate ideas (safe to do in dry-run)
    ideas = generate_all_ideas(config, archetypes=args.archetypes, max_ideas=args.max_ideas)

    if args.dry_run:
        print(f"\n[DRY RUN] Would test {len(ideas)} strategies. No data modified. Exiting.")
        return

    # Step 4: Backup DB (only when actually running)
    backup_db(config)

    # Step 5: Clear runs
    clear_runs(config)

    # Step 6: Load data and run backtests
    print(f"\n[LOADING] Loading NQ 5m data...")
    data_file = os.path.join(config.data_dir, "A2API-NQ-m5.csv")
    # FAST LOAD: The CSV timestamps like "2010-06-02 18:05:00 -04:00" already
    # represent ET wall clock time. The offset (-04:00 EDT, -05:00 EST) confirms
    # timezone but the local time is already correct. Instead of the slow
    # pd.to_datetime(utc=True).dt.tz_convert('America/New_York') which takes 5+ min
    # on 1M rows, we just strip the timezone suffix and parse the first 19 chars.
    data = pd.read_csv(data_file)
    data.columns = [c.capitalize() for c in data.columns]
    date_col = 'Time' if 'Time' in data.columns else data.columns[0]
    # Strip timezone offset, keep local ET time (e.g., "2010-06-02 18:05:00 -04:00" → "2010-06-02 18:05:00")
    data[date_col] = pd.to_datetime(data[date_col].str[:19], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index(date_col)
    data.index.name = None
    print(f"  Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    results = run_backtests(config, ideas, data)

    # Step 7: Generate comparison report
    report_path = args.report_path or os.path.join(
        config.reports_dir, f"retest_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    generate_comparison_report(old_results, results, report_path)

    # Step 8: Restart daemon
    if not args.skip_daemon_stop:
        print("\n[8/8] Ready to restart daemon.")
        print("  Run: python -m backtesting.marcus_daemon")
        print("  Or:  net start MarcusAgent")

    print("\nDone!")


if __name__ == "__main__":
    main()
