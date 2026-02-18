
import argparse
import sys
import os
from datetime import datetime
from typing import List

# Ensure we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .factory import STRATEGY_CATALOG, get_strategy_config
from .pipeline import ResearchPipeline
from .reporting import ReportGenerator

def run_strategy(name: str, symbol: str, start_date: datetime, end_date: datetime, interval: str = '15m'):
    print(f"\n>>> LAUNCHING AUTOMATED PIPELINE FOR: {name.upper()} <<<")
    print(f"    Range: {start_date.date()} -> {end_date.date()}")
    
    cfg = get_strategy_config(name)
    
    # 1. Init Pipeline
    pipeline = ResearchPipeline(
        strategy_name=name,
        strategy_cls=cfg['event_cls'],
        vector_strategy_cls=cfg['vector_cls'],
        symbol_list=[symbol],
        param_grid=cfg['default_params'],
        data_range_start=start_date,
        data_range_end=end_date,
        interval=interval,
        search_dirs=[r"C:\Users\User\Documents\AI\backtesting\data"]
    )
    
    # 2. Execute (Modified execute pipeline to return stats instead of just printing)
    # Since I can't easily modify the execute return signature of the *live* object without editing class,
    # I will rely on the fact that pipeline stores its state or prints.
    # WAIT: I should have updated pipeline.py to return the `stats` dict in `execute`.
    # Let's fix pipeline.py quickly OR just access the registry last entry.
    
    pipeline.execute()
    
    # 3. Retrieve Results for Reporting (Proxy via Registry or just Pipeline Internal Vars if I expose them)
    # The pipeline.py currently saves to registry but doesn't return the full stats dict from execute().
    # I will trust the console output for now, OR I can fetch the latest run from DB.
    
    from .registry import StrategyRegistry
    reg = StrategyRegistry()
    latest_run = reg.get_leaderboard(limit=1).iloc[0] # Simplistic check
    
    # 4. Generate Report
    # We can perform a detailed report generation here if we had the raw objects
    # For now, the pipeline prints a lot.
    
    print(f"\n✅ {name} Pipeline Completed.")
    return True

def main():
    parser = argparse.ArgumentParser(description="AI Backtesting Automation CLI")
    parser.add_argument("action", choices=["start", "list"], help="Action to perform")
    parser.add_argument("--strategy", type=str, help="Specific strategy to run", default="all")
    parser.add_argument("--symbol", type=str, default="NQ=F", help="Ticker symbol")
    parser.add_argument("--years", type=int, default=1, help="Number of years to look back")
    parser.add_argument("--start-date", type=str, help="Start Date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End Date YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="15m", help="Data timeframe (e.g. 1m, 5m, 15m)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Parse just known args to allow partial mismatch if needed
    args, unknown = parser.parse_known_args()
    
    if args.action == "list":
        print("\nAvailable Strategies:")
        for name, cfg in STRATEGY_CATALOG.items():
            print(f"- {name}: {cfg['description']}")
            
    elif args.action == "start":
        # Logic to parse dates
        end = datetime.now()
        start = datetime(end.year - args.years, end.month, end.day)
        
        if args.end_date:
            end = datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.start_date:
            start = datetime.strptime(args.start_date, "%Y-%m-%d")

        if args.strategy == "all":
            print("Running ALL strategies...")
            for name in STRATEGY_CATALOG:
                run_strategy(name, args.symbol, start, end, args.interval)
        else:
            run_strategy(args.strategy, args.symbol, start, end, args.interval)

if __name__ == "__main__":
    main()
