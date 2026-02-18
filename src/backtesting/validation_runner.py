"""
Marcus Validation Runner
========================
Compare Python backtest trade lists vs TradingView export trade-by-trade.
Includes:
  - PythonTradeExtractor: Extract individual trades from VectorEngine signals
  - TradingViewParser: Parse TV strategy tester CSV export
  - TradeComparator: Compare two trade lists with tolerances
  - SpotChecker: Bar-by-bar signal state for debugging specific dates

Usage:
    python -m backtesting.validation_runner --mode extract
    python -m backtesting.validation_runner --mode compare --py python_trades.csv --tv tv_trades.csv
    python -m backtesting.validation_runner --mode spot --date 2024-01-15
"""

import sys
import os
import csv
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# DATA CLASSES
# ===========================================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float = 0.0
    exit_reason: str = ''  # 'SL', 'TP', 'EOD', 'REVERSAL', 'DAY_RESET'
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0

    def to_dict(self):
        return {
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time) if self.exit_time else '',
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_points': self.pnl_points,
            'pnl_dollars': self.pnl_dollars,
        }


@dataclass
class ComparisonReport:
    total_python: int = 0
    total_tv: int = 0
    matched: int = 0
    unmatched_python: List[Trade] = field(default_factory=list)
    unmatched_tv: List[Trade] = field(default_factory=list)
    price_mismatches: List[dict] = field(default_factory=list)
    direction_mismatches: List[dict] = field(default_factory=list)
    exit_reason_mismatches: List[dict] = field(default_factory=list)


# ===========================================================================
# PYTHON TRADE EXTRACTOR
# ===========================================================================
class PythonTradeExtractor:
    """
    Extract individual trades from VectorEngine signal array.
    Walks the signal array chronologically, identifies entry/exit transitions.
    """

    def __init__(self, point_value=20.0):
        self.point_value = point_value

    def extract_trades(self, df: pd.DataFrame, signals: pd.Series) -> List[Trade]:
        """
        Walk the signal array, identify entries/exits, return trade list.

        Logic:
          - Signal 0→1: LONG entry
          - Signal 0→-1: SHORT entry
          - Signal 1→0 or -1→0: Exit
          - Signal 1→-1 or -1→1: Reversal (close + new entry)
        """
        trades = []
        current_trade = None

        for i in range(1, len(signals)):
            prev_sig = signals.iloc[i - 1]
            curr_sig = signals.iloc[i]
            ts = df.index[i]
            price = df['Close'].iloc[i] if 'Close' in df.columns else df['close'].iloc[i]

            # Exit existing position
            if current_trade is not None and curr_sig != prev_sig:
                current_trade.exit_time = ts
                current_trade.exit_price = price

                # Determine exit reason
                if curr_sig == 0:
                    # Check if it's EOD
                    bar_min = ts.hour * 60 + ts.minute
                    if bar_min >= 945:
                        current_trade.exit_reason = 'EOD'
                    else:
                        current_trade.exit_reason = 'SL/TP'  # Can't distinguish without SL/TP levels
                elif curr_sig != 0 and curr_sig != prev_sig:
                    current_trade.exit_reason = 'REVERSAL'

                # Calculate PnL
                if current_trade.direction == 'LONG':
                    current_trade.pnl_points = current_trade.exit_price - current_trade.entry_price
                else:
                    current_trade.pnl_points = current_trade.entry_price - current_trade.exit_price
                current_trade.pnl_dollars = current_trade.pnl_points * self.point_value

                trades.append(current_trade)
                current_trade = None

            # New entry
            if curr_sig != 0 and prev_sig != curr_sig:
                direction = 'LONG' if curr_sig == 1 else 'SHORT'
                current_trade = Trade(
                    entry_time=ts,
                    exit_time=None,
                    direction=direction,
                    entry_price=price,
                )

        # Handle open position at end of data
        if current_trade is not None:
            current_trade.exit_time = df.index[-1]
            current_trade.exit_price = df['Close'].iloc[-1] if 'Close' in df.columns else df['close'].iloc[-1]
            current_trade.exit_reason = 'DATA_END'
            if current_trade.direction == 'LONG':
                current_trade.pnl_points = current_trade.exit_price - current_trade.entry_price
            else:
                current_trade.pnl_points = current_trade.entry_price - current_trade.exit_price
            current_trade.pnl_dollars = current_trade.pnl_points * self.point_value
            trades.append(current_trade)

        return trades


def export_trades_csv(trades: List[Trade], filepath: str):
    """Export trade list to CSV for comparison."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'entry_time', 'exit_time', 'direction', 'entry_price',
            'exit_price', 'exit_reason', 'pnl_points', 'pnl_dollars'
        ])
        writer.writeheader()
        for t in trades:
            writer.writerow(t.to_dict())
    print(f"Exported {len(trades)} trades to {filepath}")


# ===========================================================================
# TRADINGVIEW PARSER
# ===========================================================================
class TradingViewParser:
    """
    Parse TradingView strategy tester CSV export.
    TV exports trades with columns like:
      Trade #, Type, Signal, Date/Time, Price, Contracts, Profit, ...
    """

    def parse(self, csv_path: str) -> List[Trade]:
        """Read TV export and normalize to same format."""
        trades = []
        raw_entries = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_entries.append(row)

        # TradingView exports entries and exits as separate rows
        # We need to pair them up
        pending_entry = None
        for row in raw_entries:
            trade_type = row.get('Type', row.get('type', '')).strip().lower()
            signal = row.get('Signal', row.get('signal', '')).strip().lower()
            dt_str = row.get('Date/Time', row.get('date/time', row.get('DateTime', '')))
            price = float(row.get('Price', row.get('price', 0)))

            # Parse datetime
            try:
                ts = pd.Timestamp(dt_str)
            except (ValueError, TypeError):
                continue

            if 'entry' in signal or 'entry' in trade_type:
                # New entry
                if pending_entry is not None:
                    # Previous entry without exit — force close
                    pending_entry.exit_reason = 'MISSING_EXIT'
                    trades.append(pending_entry)

                direction = 'LONG' if 'long' in trade_type or 'long' in signal else 'SHORT'
                pending_entry = Trade(
                    entry_time=ts,
                    exit_time=None,
                    direction=direction,
                    entry_price=price,
                )

            elif 'exit' in signal or 'exit' in trade_type or 'close' in signal:
                if pending_entry is not None:
                    pending_entry.exit_time = ts
                    pending_entry.exit_price = price

                    # Try to determine exit reason from comment
                    comment = row.get('Comment', row.get('comment', '')).strip()
                    if 'SL' in comment:
                        pending_entry.exit_reason = 'SL'
                    elif 'TP' in comment:
                        pending_entry.exit_reason = 'TP'
                    elif 'EOD' in comment:
                        pending_entry.exit_reason = 'EOD'
                    else:
                        pending_entry.exit_reason = 'UNKNOWN'

                    # Calculate PnL
                    if pending_entry.direction == 'LONG':
                        pending_entry.pnl_points = pending_entry.exit_price - pending_entry.entry_price
                    else:
                        pending_entry.pnl_points = pending_entry.entry_price - pending_entry.exit_price
                    pending_entry.pnl_dollars = pending_entry.pnl_points * 20.0

                    trades.append(pending_entry)
                    pending_entry = None

        # Handle trailing entry
        if pending_entry is not None:
            pending_entry.exit_reason = 'OPEN'
            trades.append(pending_entry)

        return trades


# ===========================================================================
# TRADE COMPARATOR
# ===========================================================================
class TradeComparator:
    """Compare two trade lists with configurable tolerances."""

    def __init__(self, price_tol: float = 0.5, time_tol_minutes: int = 10):
        self.price_tol = price_tol  # NQ points
        self.time_tol = pd.Timedelta(minutes=time_tol_minutes)

    def compare(self, python_trades: List[Trade], tv_trades: List[Trade]) -> ComparisonReport:
        """
        Match trades by entry_time (±tolerance), compare all fields.
        """
        report = ComparisonReport(
            total_python=len(python_trades),
            total_tv=len(tv_trades),
        )

        tv_used = set()

        for py_trade in python_trades:
            best_match = None
            best_time_diff = pd.Timedelta.max

            for j, tv_trade in enumerate(tv_trades):
                if j in tv_used:
                    continue

                # Match by direction and entry time proximity
                if py_trade.direction != tv_trade.direction:
                    continue

                time_diff = abs(py_trade.entry_time - tv_trade.entry_time)
                if time_diff <= self.time_tol and time_diff < best_time_diff:
                    best_match = (j, tv_trade)
                    best_time_diff = time_diff

            if best_match:
                j, tv_trade = best_match
                tv_used.add(j)
                report.matched += 1

                # Check price discrepancy
                entry_diff = abs(py_trade.entry_price - tv_trade.entry_price)
                if entry_diff > self.price_tol:
                    report.price_mismatches.append({
                        'time': py_trade.entry_time,
                        'direction': py_trade.direction,
                        'py_entry': py_trade.entry_price,
                        'tv_entry': tv_trade.entry_price,
                        'diff': entry_diff,
                    })

                # Check exit price
                if tv_trade.exit_price > 0 and py_trade.exit_price > 0:
                    exit_diff = abs(py_trade.exit_price - tv_trade.exit_price)
                    if exit_diff > self.price_tol:
                        report.price_mismatches.append({
                            'time': py_trade.exit_time,
                            'direction': py_trade.direction,
                            'py_exit': py_trade.exit_price,
                            'tv_exit': tv_trade.exit_price,
                            'diff': exit_diff,
                            'type': 'exit',
                        })

                # Check exit reason
                if tv_trade.exit_reason and py_trade.exit_reason:
                    if tv_trade.exit_reason != py_trade.exit_reason:
                        report.exit_reason_mismatches.append({
                            'time': py_trade.entry_time,
                            'py_reason': py_trade.exit_reason,
                            'tv_reason': tv_trade.exit_reason,
                        })
            else:
                report.unmatched_python.append(py_trade)

        # Unmatched TV trades
        for j, tv_trade in enumerate(tv_trades):
            if j not in tv_used:
                report.unmatched_tv.append(tv_trade)

        return report

    def print_report(self, report: ComparisonReport):
        """Print detailed comparison report."""
        print("\n" + "=" * 70)
        print("TRADE COMPARISON REPORT")
        print("=" * 70)

        print(f"\nPython trades: {report.total_python}")
        print(f"TV trades:     {report.total_tv}")
        print(f"Matched:       {report.matched}")
        match_rate = (report.matched / max(report.total_python, 1)) * 100
        print(f"Match rate:    {match_rate:.1f}%")

        if report.unmatched_python:
            print(f"\n--- Unmatched Python trades ({len(report.unmatched_python)}) ---")
            for t in report.unmatched_python[:10]:
                print(f"  {t.entry_time} {t.direction} @ {t.entry_price:.2f}")

        if report.unmatched_tv:
            print(f"\n--- Unmatched TV trades ({len(report.unmatched_tv)}) ---")
            for t in report.unmatched_tv[:10]:
                print(f"  {t.entry_time} {t.direction} @ {t.entry_price:.2f}")

        if report.price_mismatches:
            print(f"\n--- Price Mismatches ({len(report.price_mismatches)}) ---")
            for m in report.price_mismatches[:10]:
                if 'type' in m and m['type'] == 'exit':
                    print(f"  EXIT {m['time']} {m['direction']}: Py={m['py_exit']:.2f} TV={m['tv_exit']:.2f} diff={m['diff']:.2f}")
                else:
                    print(f"  ENTRY {m['time']} {m['direction']}: Py={m['py_entry']:.2f} TV={m['tv_entry']:.2f} diff={m['diff']:.2f}")

        if report.exit_reason_mismatches:
            print(f"\n--- Exit Reason Mismatches ({len(report.exit_reason_mismatches)}) ---")
            for m in report.exit_reason_mismatches[:10]:
                print(f"  {m['time']}: Py={m['py_reason']} TV={m['tv_reason']}")

        # Summary
        print(f"\n{'='*70}")
        if match_rate >= 95:
            print(f"VERDICT: EXCELLENT — {match_rate:.1f}% match rate")
        elif match_rate >= 80:
            print(f"VERDICT: GOOD — {match_rate:.1f}% match rate (some discrepancies to investigate)")
        else:
            print(f"VERDICT: POOR — {match_rate:.1f}% match rate (significant discrepancies!)")
        print(f"{'='*70}")


# ===========================================================================
# SPOT CHECKER
# ===========================================================================
class SpotChecker:
    """
    Deep-dive specific dates to find why trades differ.
    Prints bar-by-bar signal state for one trading day.
    """

    def check_date(self, df: pd.DataFrame, date_str: str, strategy, engine):
        """
        Print bar-by-bar analysis for one trading day.

        Args:
            df: Full price DataFrame
            date_str: e.g., "2024-01-15"
            strategy: VectorizedNQORB instance
            engine: VectorEngine instance
        """
        # Filter to single day
        day_mask = df.index.date == pd.Timestamp(date_str).date()
        day_df = df[day_mask].copy()

        if len(day_df) == 0:
            print(f"No data for {date_str}")
            return

        # Run backtest on full data first (indicators need history)
        result = engine.run(df)
        signals = strategy.generate_signals(df)

        # Get signals for this day
        day_signals = signals[day_mask]
        day_equity = result['equity_curve'][day_mask]

        # Print header
        print(f"\n{'='*100}")
        print(f"SPOT CHECK: {date_str}")
        print(f"{'='*100}")
        print(f"{'Time':<12} {'O':>10} {'H':>10} {'L':>10} {'C':>10} {'Signal':>8} {'Note'}")
        print(f"{'-'*100}")

        prev_sig = 0
        for i in range(len(day_df)):
            ts = day_df.index[i]
            time_str = ts.strftime("%H:%M")
            o = day_df['Open'].iloc[i] if 'Open' in day_df.columns else day_df['open'].iloc[i]
            h = day_df['High'].iloc[i] if 'High' in day_df.columns else day_df['high'].iloc[i]
            l = day_df['Low'].iloc[i] if 'Low' in day_df.columns else day_df['low'].iloc[i]
            c = day_df['Close'].iloc[i] if 'Close' in day_df.columns else day_df['close'].iloc[i]
            sig = day_signals.iloc[i]

            # Annotation
            note = ""
            bar_min = ts.hour * 60 + ts.minute
            if bar_min >= 570 and bar_min < 585:
                note = "ORB"
            if sig != prev_sig:
                if sig == 1:
                    note += " ← LONG ENTRY"
                elif sig == -1:
                    note += " ← SHORT ENTRY"
                elif prev_sig != 0:
                    note += " ← EXIT"
            if bar_min >= 945:
                note += " EOD"

            print(f"{time_str:<12} {o:>10.2f} {h:>10.2f} {l:>10.2f} {c:>10.2f} {sig:>8} {note}")
            prev_sig = sig

        # Day summary
        day_trades = PythonTradeExtractor().extract_trades(day_df, day_signals)
        print(f"\nTrades: {len(day_trades)}")
        for t in day_trades:
            print(f"  {t.direction} @ {t.entry_price:.2f} → {t.exit_price:.2f} ({t.exit_reason}) PnL={t.pnl_points:+.2f} pts")


# ===========================================================================
# MAIN RUNNER
# ===========================================================================
def run_python_extract(data_path: str, output_path: str,
                       orb_start="09:30", orb_end="09:45",
                       ema_filter=50, sl_atr_mult=2.0, tp_atr_mult=4.0,
                       atr_max_mult=2.5):
    """
    Run Python backtest and extract trade list.
    """
    from backtesting.vector_engine import VectorizedNQORB, VectorEngine

    print(f"Loading data from {data_path}...")
    # Load CSV directly (SmartDataHandler needs symbol_list; we have a direct path)
    df = pd.read_csv(data_path)
    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns]
    # Parse datetime and strip timezone (preserve ET wall clock time)
    date_col = 'Time' if 'Time' in df.columns else df.columns[0]
    # Convert to ET wall clock time (not UTC!) to match ORB session times
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.set_index(date_col)
    df.index.name = None
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    print(f"\nRunning ORB backtest: {orb_start}-{orb_end}, EMA={ema_filter}, SL={sl_atr_mult}x, TP={tp_atr_mult}x...")
    strategy = VectorizedNQORB(
        orb_start=orb_start, orb_end=orb_end,
        ema_filter=ema_filter,
        sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
        atr_max_mult=atr_max_mult,
    )
    engine = VectorEngine(strategy, initial_capital=100000, commission=2.06, slippage=5.0, point_value=20.0)
    result = engine.run(df)

    signals = strategy.generate_signals(df)
    extractor = PythonTradeExtractor(point_value=20.0)
    trades = extractor.extract_trades(df, signals)

    print(f"\nExtracted {len(trades)} trades")
    if trades:
        longs = sum(1 for t in trades if t.direction == 'LONG')
        shorts = sum(1 for t in trades if t.direction == 'SHORT')
        total_pnl = sum(t.pnl_dollars for t in trades)
        wins = sum(1 for t in trades if t.pnl_points > 0)
        print(f"  Longs: {longs}, Shorts: {shorts}")
        print(f"  Win rate: {wins/len(trades)*100:.1f}%")
        print(f"  Total PnL: ${total_pnl:,.2f}")

    export_trades_csv(trades, output_path)
    return trades, result, df, strategy, engine


def main():
    parser = argparse.ArgumentParser(description="Marcus Validation Runner")
    parser.add_argument('--mode', choices=['extract', 'compare', 'spot'], default='extract')
    parser.add_argument('--data', default=r'C:\Users\User\Desktop\Zero_Human_HQ\Quant_Lab\data\A2API-NQ-m5.csv')
    parser.add_argument('--py', default='python_trades.csv', help='Python trades CSV')
    parser.add_argument('--tv', default='tv_trades.csv', help='TradingView trades CSV')
    parser.add_argument('--output', default='python_trades.csv')
    parser.add_argument('--date', default=None, help='Date for spot check (YYYY-MM-DD)')

    # Strategy params
    parser.add_argument('--orb-start', default='09:30')
    parser.add_argument('--orb-end', default='09:45')
    parser.add_argument('--ema', type=int, default=50)
    parser.add_argument('--sl', type=float, default=2.0)
    parser.add_argument('--tp', type=float, default=4.0)
    parser.add_argument('--atr-max', type=float, default=2.5)

    args = parser.parse_args()

    if args.mode == 'extract':
        run_python_extract(
            args.data, args.output,
            orb_start=args.orb_start, orb_end=args.orb_end,
            ema_filter=args.ema, sl_atr_mult=args.sl, tp_atr_mult=args.tp,
            atr_max_mult=args.atr_max,
        )

    elif args.mode == 'compare':
        py_trades = []
        with open(args.py, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                py_trades.append(Trade(
                    entry_time=pd.Timestamp(row['entry_time']),
                    exit_time=pd.Timestamp(row['exit_time']) if row['exit_time'] else None,
                    direction=row['direction'],
                    entry_price=float(row['entry_price']),
                    exit_price=float(row['exit_price']),
                    exit_reason=row['exit_reason'],
                    pnl_points=float(row['pnl_points']),
                    pnl_dollars=float(row['pnl_dollars']),
                ))

        tv_parser = TradingViewParser()
        tv_trades = tv_parser.parse(args.tv)

        comparator = TradeComparator(price_tol=0.5, time_tol_minutes=10)
        report = comparator.compare(py_trades, tv_trades)
        comparator.print_report(report)

    elif args.mode == 'spot':
        if not args.date:
            print("Error: --date required for spot mode")
            return

        trades, result, df, strategy, engine = run_python_extract(
            args.data, '/dev/null',
            orb_start=args.orb_start, orb_end=args.orb_end,
            ema_filter=args.ema, sl_atr_mult=args.sl, tp_atr_mult=args.tp,
            atr_max_mult=args.atr_max,
        )

        checker = SpotChecker()
        checker.check_date(df, args.date, strategy, engine)


if __name__ == "__main__":
    main()
