import os
import sys
import subprocess
import glob
import pandas as pd
from datetime import datetime

# Bootstrapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
import backtesting.boot

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def run_command(cmd):
    """Run a shell command and stream output."""
    print(f"\n[AUTO-PILOT] Executing: {' '.join(cmd)}")
    # Merge stderr into stdout so we see errors
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', cwd=ROOT_DIR, env=env)
    
    stdout_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            stdout_lines.append(line.strip())
            
    rc = process.poll()
    if rc != 0:
        print("[AUTO-PILOT] Process failed.")
        return False, stdout_lines
    return True, stdout_lines

def auto_research():
    print("="*60)
    print(f"AUTO-PILOT RESEARCH SESSION: {datetime.now()}")
    print("="*60)
    
    # 1. Environment Check / QA Suite
    print("\n>>> Running Pre-Flight QA Suite...")
    qa_cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'qa_suite.py')]
    success, qa_output = run_command(qa_cmd)
    
    if not success:
        print("[CRITICAL] QA Suite Failed. Aborting Auto-Pilot.")
        print("See QA Report in outputs/ folder for details.")
        return

    print("[PASS] QA Suite Verified. Proceeding to Research.")

    # 2. Data Scan
    # Load Config
    config_path = os.path.join(ROOT_DIR, 'config.json')
    search_dirs = [DATA_DIR]
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            search_dirs.extend(cfg.get('data', {}).get('search_dirs', []))

    csv_files = []
    for d in search_dirs:
         # Resolve
         full_d = d if os.path.isabs(d) else os.path.join(ROOT_DIR, d)
         if os.path.exists(full_d):
             csv_files.extend(glob.glob(os.path.join(full_d, "*.csv")))

    # Extract unique symbols
    symbols = []
    for f in csv_files:
        name = os.path.basename(f)
        # Handle 'A2API-SYMBOL-m1.csv' or 'SYMBOL.csv'
        if name.startswith('A2API-'):
            parts = name.split('-')
            if len(parts) >= 2:
                symbols.append(parts[1].upper())
        else:
            symbols.append(name.replace('.csv', '').split('_')[0].upper())
            
    symbols = list(set(symbols))
    
    if not symbols:
        print("[ERROR] No data found in any config directory.")
        return

    print(f"[DATA] Found Symbols: {symbols}")
    
    # 3. Strategy Rotation
    # We define a list of configs to test
    strategies = [
        {'name': 'NQORB', 'symbol': 'NQ=F', 'args': ['--strategy', 'NQORB', '--optimize', '--gpu']},
        # {'name': 'MA', 'symbol': 'SPY', 'args': ['--strategy', 'MA', '--optimize']} # Vectorized MA matches 
    ]
    
    results = []
    
    for strat in strategies:
        if strat['symbol'] not in symbols and strat['symbol'] != 'NQ': # NQ usually mapped from file
             # Try to find a partial match?
             pass
        
        print(f"\n>>> Running Optimization for {strat['name']}...")
        
        # Build Command
        # Assumes running with the same python interpreter
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'runner.py')] + strat['args']
        
        # Add Data Dir
        cmd += ['--data_dir', DATA_DIR]
        
        # Add Symbol
        # For NQORB usually symbol is implied or passed? Runner expects --symbol
        cmd += ['--symbol', strat['symbol']]
        
        success, output = run_command(cmd)
        
        if success:
            # Parse output for best result? 
            # Or reliance on 'optimization_results.csv' being generated.
            # Runner saves 'optimization_results.csv' in CWD (ROOT_DIR)
            res_file = os.path.join(ROOT_DIR, 'optimization_results.csv')
            if os.path.exists(res_file):
                df = pd.read_csv(res_file)
                best = df.iloc[0]
                
                # Normalize keys (GridSearch uses 'Total Return', Runner uses 'Return')
                total_return = best.get('Total Return')
                if pd.isna(total_return) or total_return is None:
                    total_return = best.get('Return', 0.0)
                
                results.append({
                    'strategy': strat['name'],
                    'symbol': strat['symbol'],
                    'return': total_return,
                    'params': best.get('Params', 'N/A')
                })
                
                # Archive Best Result
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"{strat['name']}_{strat['symbol']}_{ts}.csv"
                df.to_csv(os.path.join(OUTPUT_DIR, archive_name))
                print(f"[SUCCESS] Archived results to outputs/{archive_name}")
                
                # --- AI Feedback Integration ---
                # Use normalized return from above
                
                # Heuristic for AI Attention
                # If we have Sharpe info from results? Optimization CSV might vary.
                best_sharpe = best.get('Sharpe Ratio', 0)
                
                if total_return < 0.05: # Less than 5% return
                    print("\n[AI ASSISTANT] Strategy Underperformance Detected.")
                    
                    # Construct minimal stats for prompts
                    stats = {
                        'Total Return': total_return,
                        'Sharpe Ratio': best_sharpe,
                        'Max Drawdown': best.get('Max Drawdown', 0)
                    }
                    
                    # We need to import AIOptimizationPrompter
                    # Do dynamic import to avoid scope issues
                    sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
                    from backtesting.ai_assistant import AIOptimizationPrompter
                    
                    prompter = AIOptimizationPrompter(stats)
                    prompt_file = f"AI_PROMPT_{strat['name']}_{ts}.txt"
                    prompter.save_prompt(strat['name'], strat['symbol'], filepath=os.path.join(OUTPUT_DIR, prompt_file))
                    print(f"--> ACTION: Feed outputs/{prompt_file} to your LLM for code improvements.")

            else:
                print("[WARNING] No optimization results found.")
        else:
            print("[FAIL] Optimization crashed.")

    # 4. Generate Daily Report
    report_path = os.path.join(OUTPUT_DIR, f"Research_Report_{datetime.now().strftime('%Y%m%d')}.md")
    with open(report_path, "w") as f:
        f.write(f"# Auto-Pilot Research Report - {datetime.now().date()}\n\n")
        f.write("| Strategy | Symbol | Top Return | Best Params |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for res in results:
            f.write(f"| {res['strategy']} | {res['symbol']} | {res['return']:.2%} | `{res['params']}` |\n")
    
    print(f"\n[COMPLETE] Report generated: {report_path}")

if __name__ == "__main__":
    auto_research()
