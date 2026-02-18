import sys
import os
import json
from datetime import datetime

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

sys.path.insert(0, SRC_DIR)

import backtesting.boot # Inject Environment Fixes

from backtesting.diagnostics import SystemDiagnostics
from backtesting.registry import StrategyRegistry

def run_qa():
    print("="*60)
    print(f"QA SUITE INITIALIZED: {datetime.now()}")
    print("="*60)
    
    # Load Config
    config_path = os.path.join(PROJECT_ROOT, 'config.json')
    search_dirs = [DATA_DIR]
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            search_dirs.extend(cfg.get('data', {}).get('search_dirs', []))
    
    # Use the first valid directory as primary
    valid_dir = DATA_DIR
    for d in search_dirs:
        # Resolve relative to project root if needed
        # Handle Windows paths with backslashes properly
        full_d = d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)
        
        if os.path.exists(full_d):
            # Check for ANY csv
            try:
                if any(f.lower().endswith('.csv') for f in os.listdir(full_d)):
                    valid_dir = full_d
                    break
            except Exception:
                continue
            
    diag = SystemDiagnostics(valid_dir)
    report = diag.run_all()
    
    # Analyze Report
    fails = [c for c in report['checks'] if c['status'] == 'FAIL']
    
    # Graceful degradation: If GPU failed, move to warnings (don't stop pipeline)
    gpu_fails = [f for f in fails if f['category'] == 'Hardware']
    if gpu_fails:
        print("\n[NOTE] GPU Validation Failed. Falling back to CPU mode.")
        for f in gpu_fails:
            f['status'] = 'WARNING' # Downgrade to Warning
        
        # update lists
        fails = [c for c in report['checks'] if c['status'] == 'FAIL']
    
    warnings = [c for c in report['checks'] if c['status'] == 'WARNING']
    
    status = 'FAIL' if fails else ('WARNING' if warnings else 'PASS')
    
    # ... (Print Summary) ...

    
    print("\n" + "="*60)
    print("QA SUMMARY")
    print("="*60)
    print(f"Total Checks: {len(report['checks'])}")
    print(f"Passed:       {len(report['checks']) - len(fails) - len(warnings)}")
    print(f"Warnings:     {len(warnings)}")
    print(f"Failures:     {len(fails)}")
    
    if fails:
        print("\n[CRITICAL FAILURES]")
        for f in fails:
            print(f" - {f['category']}: {f['message']}")
            if f.get('details'):
                 print(f"   Details: {f['details']}")
                 
    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f" - {w['message']}")
            
    # Save Report to File
    filename = f"QA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[MEMORY] Diagnostics saved to {path}")
    
    # Save to System Memory (DB)
    try:
        reg = StrategyRegistry(os.path.join(PROJECT_ROOT, "backtests.db"))
        reg.log_diagnostic(status, report)
    except Exception as e:
        print(f"[MEMORY FAIL] Could not log to DB: {e}")
    
    exit_code = 1 if fails else 0
    sys.exit(exit_code)

if __name__ == "__main__":
    run_qa()
