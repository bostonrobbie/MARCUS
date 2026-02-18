from typing import Dict, List
import pandas as pd

from .statistics import StatisticalSignificance


class ValidationSuite:
    """
    Automated Auditor for Trading Strategies.
    Checks for:
    1. Statistical Significance (Sample Size)
    2. Consistency (Sharpe Ratio)
    3. Robustness (Optimistic vs Pessimistic Divergence)
    """
    
    def __init__(self, stats: Dict, trade_log: List[Dict]):
        self.stats = stats
        self.trade_log = trade_log
        self.report = {
            'status': 'PENDING',
            'checks': [],
            'warnings': [],
            'errors': []
        }

    def run_basic_checks(self, returns=None):
        """Runs single-pass checks available from one backtest run."""
        self._check_sample_size()
        self._check_consistency()
        if returns is not None:
            self._check_statistical_significance(returns)
        self._determine_status()
        return self.report

    def _check_statistical_significance(self, returns, alpha=0.05):
        """
        Check if Sharpe ratio is statistically significant.

        Args:
            returns: Array or Series of strategy returns
            alpha: Significance level (default 0.05 for 95% confidence)
        """
        import numpy as np

        if len(returns) < 10:
            self.report['warnings'].append(
                "Insufficient data for statistical significance testing."
            )
            return

        returns_arr = np.array(returns)
        sharpe = self.stats.get('Sharpe Ratio', 0)

        # Calculate p-value
        p_value = StatisticalSignificance.sharpe_pvalue(sharpe, len(returns_arr))

        # Calculate confidence interval
        sharpe_calc, ci_lower, ci_upper = StatisticalSignificance.sharpe_confidence_interval(
            returns_arr, confidence=1-alpha
        )

        passed = p_value < alpha and ci_lower > 0

        check_result = {
            'name': 'Statistical Significance',
            'value': f"p={p_value:.4f}, CI=[{ci_lower:.2f}, {ci_upper:.2f}]",
            'threshold': f"p<{alpha}, CI>0",
            'passed': passed
        }
        self.report['checks'].append(check_result)

        if not passed:
            if p_value >= alpha:
                self.report['warnings'].append(
                    f"Sharpe ratio is NOT statistically significant (p={p_value:.3f} >= {alpha}). "
                    "Results may be due to random chance."
                )
            if ci_lower <= 0:
                self.report['warnings'].append(
                    f"Sharpe ratio 95% CI [{ci_lower:.2f}, {ci_upper:.2f}] includes zero. "
                    "Cannot confidently say strategy has positive edge."
                )

    def _check_sample_size(self, min_trades=30):
        n_trades = len(self.trade_log)
        passed = n_trades >= min_trades
        
        check_result = {
            'name': 'Sample Size',
            'value': n_trades,
            'threshold': min_trades,
            'passed': passed
        }
        self.report['checks'].append(check_result)
        
        if not passed:
            msg = f"Insufficient sample size: {n_trades} trades (Goal: >{min_trades}). Results may be noise."
            self.report['warnings'].append(msg)

    def _check_consistency(self, min_sharpe=0.5):
        sharpe = self.stats.get('Sharpe Ratio', 0)
        # Handle NaN sharpe
        if pd.isna(sharpe): sharpe = -999
        
        passed = sharpe >= min_sharpe
        
        check_result = {
            'name': 'Sharpe Consistency',
            'value': f"{sharpe:.2f}",
            'threshold': min_sharpe,
            'passed': passed
        }
        self.report['checks'].append(check_result)
        
        if not passed:
            self.report['warnings'].append(f"Low risk-adjusted return: Sharpe {sharpe:.2f} < {min_sharpe}")

    def compare_robustness(self, pessimistic_stats: Dict):
        """
        Compares current (assumed Optimistic) stats with Pessimistic stats.
        """
        opt_dd = abs(self.stats.get('Max Drawdown', 0))
        pess_dd = abs(pessimistic_stats.get('Max Drawdown', 0))
        
        # Guard against zero division
        if opt_dd == 0: opt_dd = 0.0001
        
        degradation = (pess_dd - opt_dd) / opt_dd
        
        # Threshold: Fail if Pessimistic is > 2x worse (100% degradation)
        passed = degradation < 1.0 
        
        check_result = {
            'name': 'Execution Robustness',
            'value': f"{degradation:.1%} Degradation",
            'threshold': '100%',
            'passed': passed
        }
        self.report['checks'].append(check_result)
        
        if not passed:
            self.report['errors'].append(
                f"Strategy is FRAGILE: Max Drawdown doubles in pessimistic conditions "
                f"({opt_dd:.1%} -> {pess_dd:.1%})."
            )

    def _determine_status(self):
        if self.report['errors']:
            self.report['status'] = 'FAIL'
        elif self.report['warnings']:
            self.report['status'] = 'WARNING'
        else:
            self.report['status'] = 'PASS'

    def print_cert(self):
        print("\n" + "="*40)
        print(f"STRATEGY CERTIFICATION: [{self.report['status']}]")
        print("="*40)
        for check in self.report['checks']:
            mark = "[OK]" if check['passed'] else "[X] "
            print(f"{mark} {check['name']}: {check['value']}")
        
        if self.report['warnings']:
            print("\nWarnings:")
            for w in self.report['warnings']:
                print(f" - {w}")
                
        if self.report['errors']:
            print("\nCRITICAL FAILURES:")
            for e in self.report['errors']:
                print(f" ! {e}")
        print("="*40 + "\n")

    def analyze_regimes(self, equity_curve: pd.DataFrame, regime_series: pd.Series) -> Dict:
        """
        Segment performance by Regime.
        """
        print("\n" + "-"*40)
        print("REGIME ANALYSIS")
        print("-" * 40)
        
        # Merge Equity and Regime
        # Alignt timestamps
        df = equity_curve.copy()
        if 'equity' not in df.columns:
            return {}
            
        # Join on index (assuming both are datetime index)
        # regime_series might len != df len if data_handler truncated
        # We assume they align roughly or we reindex
        
        df['regime'] = regime_series.reindex(df.index, method='ffill')
        
        # Calculate Daily Returns
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        results = {}
        
        # Group by Regime
        # Regime Enum to string
        for regime_val in df['regime'].unique():
            if pd.isna(regime_val): continue
            
            mask = df['regime'] == regime_val
            sub_df = df[mask]
            
            if len(sub_df) < 5: continue # Skip insignificant samples
            
            # regime_val is Enum, get name
            r_name = regime_val.name if hasattr(regime_val, 'name') else str(regime_val)
            
            cum_ret = (1 + sub_df['returns']).prod() - 1.0
            avg_ret = sub_df['returns'].mean()
            std_ret = sub_df['returns'].std()
            sharpe = (avg_ret / std_ret * (252**0.5)) if std_ret > 0 else 0
            
            results[r_name] = {
                'Return': f"{cum_ret:.2%}",
                'Sharpe': f"{sharpe:.2f}",
                'Days': len(sub_df)
            }
            
            print(f"Regime: {r_name:<15} | Return: {results[r_name]['Return']:<8} | Sharpe: {results[r_name]['Sharpe']:<5} | Days: {len(sub_df)}")
            
            # Check for Major Failures (e.g. > 20% loss in a regime)
            if cum_ret < -0.20:
                print(f"  [WARNING] Strategy performs poorly in {r_name} ({cum_ret:.2%})")
                
        return results

class MonteCarloValidator:
    """
    Simulates alternative realities by restructuring trade sequence.
    Goal: Assess if 'Max Drawdown' was luck or skill.
    """
    def __init__(self, trade_log: List[Dict], n_sims: int = 1000):
        self.trade_log = trade_log
        self.n_sims = n_sims

    def run(self):
        import numpy as np
        
        returns = []
        for t in self.trade_log:
             if 'return' in t:
                 returns.append(t['return'])
             elif 'pnl' in t and 'entry_value' in t:
                 returns.append(t['pnl'] / t['entry_value'])
                 
        if not returns:
            print("[Monte Carlo] No trade returns found to simulate.")
            return {}
            
        returns = np.array(returns)
        sim_dds = []

        print(f"\n[Monte Carlo] Running {self.n_sims} simulations on {len(returns)} trades...")
        
        for _ in range(self.n_sims):
            # Shuffle with replacement (Bootstrap)
            sim_rets = np.random.choice(returns, size=len(returns), replace=True)
            
            # Construct Equity Curve
            equity = np.cumprod(1 + sim_rets)
            peak = np.maximum.accumulate(equity)
            # Guard: replace zero peaks with nan to avoid division by zero
            safe_peak = np.where(peak == 0, np.nan, peak)
            dd = np.where(np.isnan(safe_peak), 0, (equity - peak) / safe_peak)
            max_dd = np.min(dd)
            sim_dds.append(max_dd)
            
        sim_dds = np.array(sim_dds)
        
        # Metrics
        worst_case_95 = np.percentile(sim_dds, 5) # 5th percentile (e.g. -0.25)
        avg_dd = np.mean(sim_dds)
        
        print("-" * 40)
        print("MONTE CARLO RESULTS")
        print("-" * 40)
        print(f"Original Max DD:  (Depends on sequence)")
        print(f"Avg Simulated DD: {avg_dd:.2%}")
        print(f"95% Worst Case:   {worst_case_95:.2%}")
        
        if worst_case_95 < -0.50:
            print("[FAIL] Risk of Ruin > 50% in worst-case scenarios.")
        else:
            print("[PASS] Strategy survives worst-case shuffling.")
        print("="*40 + "\n")
        
        return {
            'avg_dd': avg_dd,
            'worst_case_95': worst_case_95
        }

class SensitivityTester:
    def __init__(self, strategy_cls, symbol_list, data_dirs, base_params, runner_func):
        self.strategy_cls = strategy_cls
        self.symbol_list = symbol_list
        self.data_dirs = data_dirs
        self.base_params = base_params
        self.runner_func = runner_func # Dependency injection to avoid circular import
        
    def run(self):
        print("\n" + "="*40)
        print(f"SENSITIVITY ANALYSIS: {self.strategy_cls.__name__}")
        print("="*40)
        
        results = []
        variations = self._generate_variations()
        
        print(f"Testing {len(variations)} parameter variations...")
        
        for params in variations:
            # We run without plotting, validation, or verbose output
            # Just capture the stats
            stats, _ = self.runner_func(
                self.strategy_cls, 
                self.symbol_list, 
                self.data_dirs, 
                plot=False, 
                mode='OPTIMISTIC', 
                check_validation=False, 
                **params
            )
            
            res = {
                'params': str(params),
                'return': stats.get('Total Return', 0),
                'sharpe': stats.get('Sharpe Ratio', 0),
                'dd': stats.get('Max Drawdown', 0)
            }
            results.append(res)
            print(f"Params: {params} -> Return: {res['return']:.2%}, Sharpe: {res['sharpe']:.2f}")
            
        self._analyze_stability(results)

    def _generate_variations(self):
        """Generates +/- 10% variations for integer/float parameters."""
        variations = [self.base_params.copy()]
        
        for key, value in self.base_params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # +10%
                v_up = self.base_params.copy()
                if isinstance(value, int):
                    v_up[key] = int(value * 1.1) + 1
                else:
                    v_up[key] = value * 1.1
                variations.append(v_up)
                
                # -10%
                v_down = self.base_params.copy()
                if isinstance(value, int):
                    v_down[key] = int(value * 0.9)
                else:
                    v_down[key] = value * 0.9
                variations.append(v_down)
                
        # Remove duplicates
        unique_vars = []
        seen = set()
        for v in variations:
            s = str(sorted(v.items()))
            if s not in seen:
                seen.add(s)
                unique_vars.append(v)
                
        return unique_vars

    def _analyze_stability(self, results):
        df = pd.DataFrame(results)
        if df.empty: return
        
        avg_ret = df['return'].mean()
        std_ret = df['return'].std()
        
        print("-" * 40)
        print("STABILITY REPORT")
        print("-" * 40)
        print(f"Base Params:   {self.base_params}")
        print(f"Mean Return:   {avg_ret:.2%}")
        print(f"Std Dev (Stability): {std_ret:.2%}")
        
        # Heuristic: If StdDev > 0.5 * Mean, it's unstable
        if abs(avg_ret) > 0 and std_ret > 0.5 * abs(avg_ret):
            print("[FAIL] Strategy is UNSTABLE. Small param changes cause wild result swings.")
        else:
            print("[PASS] Strategy appears ROBUST to parameter noise.")
        print("="*40 + "\n")
