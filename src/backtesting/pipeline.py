
import pandas as pd
import json
import os
from typing import Dict, Any, List, Type
from datetime import datetime
from queue import Queue

from .data import SmartDataHandler
from .optimizer import VectorizedGridSearch, WalkForwardOptimizer
from .engine import BacktestEngine
from .portfolio import Portfolio
from .execution import SimulatedExecutionHandler, FixedCommission, AssetAwareCommissionModel, VolatilitySlippageModel
from .registry import StrategyRegistry
from .strategy import Strategy
from .vector_engine import VectorEngine, VectorizedNQORB, VectorizedMA
from .validation import ValidationSuite, MonteCarloValidator, SensitivityTester
from .regime import MarketRegimeDetector
from .advisor import StrategyAdvisor
# from .pine_generator import PineScriptGenerator
from .skeptic import StrategySkeptic
from .dashboard import DashboardGenerator
# NOTE: Removed duplicate 'from .vector_engine import ...' that was here
from .statistics import analyze_strategy_significance
from .attribution import analyze_trades
from .wfo_analytics import analyze_wfo_results
from .insights import generate_insights, InsightsEngine, InsightsDashboard
from .decision_agent import ValidationDecisionAgent, DecisionContext, Decision
from .code_review_agent import CodeReviewAgent
from .param_agent import ParameterAgent
from .logic_agent import LogicAgent

class ResearchPipeline:
    """
    The Unified Pipeline connecting Data -> Vector Research -> Event Validation -> Registry.
    Now supports WFO, Monte Carlo, Sensitivity, and TV Emulation.
    """
    def __init__(self, 
                 strategy_name: str, 
                 strategy_cls: Type[Strategy],
                 symbol_list: List[str],
                 param_grid: Dict[str, List],
                 data_range_start: datetime, 
                 data_range_end: datetime,
                 interval: str = '15m',
                 initial_capital: float = 100000.0,
                 search_dirs: List[str] = None,
                 config: Dict = None):
        
        self.strategy_name = strategy_name
        self.strategy_cls = strategy_cls
        self.symbol_list = symbol_list
        self.param_grid = param_grid
        self.start_date = pd.to_datetime(data_range_start)
        self.end_date = pd.to_datetime(data_range_end)
        self.interval = interval
        self.initial_capital = initial_capital
        self.search_dirs = search_dirs if search_dirs else []
        self.config = config if config else {}
        
        self.registry = StrategyRegistry()
        self.data_handler = None
        
        # Resolve Vector Strategy Class
        self.vector_strategy_cls = self._resolve_vector_strat(strategy_name)

    def _resolve_vector_strat(self, name: str):
        mapping = {
            'MovingAverageCrossover': VectorizedMA,
            'NqOrb': VectorizedNQORB,
            'NqOrb15m': VectorizedNQORB,
            'CleanOrb15m': VectorizedNQORB,
            'CleanOrb5m': VectorizedNQORB,
        }
        return mapping.get(name, VectorizedNQORB) # Default to NQORB if unknown

    def load_data(self):
        """Step 1: Ingest Data"""
        print(f"\n[1] Ingesting Data for {self.symbol_list}...")
        self.data_handler = SmartDataHandler(
            symbol_list=self.symbol_list,
            search_dirs=self.search_dirs,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval
        )
        # Verify data exists
        for sym in self.symbol_list:
            if self.data_handler.symbol_data.get(sym) is None:
                raise ValueError(f"No data found for {sym}")
        print("    Data Loaded Successfully.")

    def run_optimization(self, wfo: bool = False) -> pd.DataFrame:
        """Step 2: Run Optimization (Vector or WFO)"""
        print(f"\n[2] Running Optimization ({self.strategy_name})...")
        
        if wfo:
            return self._run_wfo()
        else:
            return self._run_grid_search()

    def _run_grid_search(self) -> pd.DataFrame:
        optimizer = VectorizedGridSearch(
            data_handler_cls=SmartDataHandler,
            data_handler_args=(self.symbol_list, self.search_dirs, self.start_date, self.end_date, self.interval),
            strategy_cls=self.strategy_cls,
            vector_strategy_cls=self.vector_strategy_cls,
            param_grid=self.param_grid,
            initial_capital=self.initial_capital,
            n_jobs=-1
        )
        results = optimizer.run()
        print(f"    Grid Search Complete. Tested {len(results)} combinations.")
        return results

    def _run_wfo(self):
        print("    Mode: Walk-Forward Optimization")
        wfo = WalkForwardOptimizer(
            strategy_cls=self.strategy_cls,
            symbol_list=self.symbol_list,
            search_dirs=self.search_dirs,
            param_grid=self.param_grid,
            train_days=90, # Configurable?
            test_days=30,
            step_days=30,
            interval=self.interval,
            vector_engine_cls=VectorEngine,
            vector_strategy_cls=self.vector_strategy_cls
        )
        results, stitched = wfo.run()
        return results, stitched

    def run_event_certification(self, params: Dict[str, Any]):
        """Step 3: Event-Driven Certification (The Sanity Check)"""
        print(f"\n[3] Running Event-Driven Certification...")
        print(f"    Params: {params}")
        
        # Setup Event Engine components
        events = Queue()
        
        # Create fresh DataHandler
        data = SmartDataHandler(
            self.symbol_list, 
            self.search_dirs, 
            self.start_date, 
            self.end_date, 
            self.interval
        )
        
        # Execution Config
        exec_mode = self.config.get('execution_mode', 'TV_BROKER_EMULATOR')
        
        # Commission & Slippage (Asset Aware)
        comm_model = AssetAwareCommissionModel(instrument_specs=self.config.get('instrument_specs'))
        slip_model = VolatilitySlippageModel(data, factor=0.05) if self.config.get('use_vol_slippage') else None
        
        portfolio = Portfolio(data, events, self.initial_capital)
        strategy = self.strategy_cls(data, events, **params)
        execution = SimulatedExecutionHandler(
            events, 
            data, 
            commission_model=comm_model,
            slippage_model=slip_model,
            mode=exec_mode
        )
        
        engine = BacktestEngine(data, strategy, portfolio, execution)
        engine.run()
        
        # Metrics
        if not portfolio.equity_curve:
            total_return = 0.0
            final_equity = self.initial_capital
        else:
            final_equity = portfolio.equity_curve[-1]['equity']
            total_return = (final_equity / self.initial_capital) - 1.0
        
        print(f"    Certification Result: Return = {total_return:.2%}, Equity = {final_equity:,.2f}")
        
        # Extract Trade Log (Filter for Realized PnL)
        trade_log = [t for t in portfolio.trade_log if t['realized_pnl'] != 0.0]
        # Actually Portfolio has 'closed_trades' list usually.
        
        return total_return, final_equity, portfolio

    def run_robustness_checks(self, portfolio: Portfolio, best_params: Dict) -> Dict:
        """Step 4: Advanced Robustness (Monte Carlo, Sensitivity, Regimes)"""
        print(f"\n[4] Running Robustness Checks...")
        metrics = {}
        
        # 0. Define Primary Symbol
        sym = self.symbol_list[0]

        # 1. Regime Analysis
        print("    [A] Regime Analysis")
        equity_curve = pd.DataFrame(portfolio.equity_curve)
        if not equity_curve.empty:
            if 'datetime' in equity_curve.columns:
                equity_curve['datetime'] = pd.to_datetime(equity_curve['datetime'])
                equity_curve.set_index('datetime', inplace=True)
            
            # Need regime series
            # We need the full dataframe for the primary symbol
            sym = self.symbol_list[0]
            if sym in self.data_handler.symbol_data:
                df = self.data_handler.symbol_data[sym]
                regime_series = MarketRegimeDetector().get_regime_series(df)
                
                suite = ValidationSuite({}, [])
                regime_stats = suite.analyze_regimes(equity_curve, regime_series)
                metrics['regime_stats'] = regime_stats
        
        # 2. Monte Carlo
        print("    [B] Monte Carlo Simulation")
        # We need a trade log. Portfolio object should have it.
        # Let's assume best_tearsheet logic from runner is analogous to portfolio.closed_trades
        trades = getattr(portfolio, 'closed_trades', []) # List[Dict]
        if trades:
            mc = MonteCarloValidator(trades, n_sims=1000)
            mc_res = mc.run()
            metrics.update(mc_res)
        else:
            print("    [SKIP] No trades to simulate.")

        # 3. Strategy Advisor
        print("    [C] Strategy Advisor")
        advisor = StrategyAdvisor(metrics, best_params)
        advice = advisor.analyze()
        metrics['advice'] = advice
        for item in advice:
            print(f"      💡 {item}")

        # 4. Pine Script Generation
        print("    [D] Generating Pine Script (Round-Trip)")
        # Instantiate strategy temporarily or use class method if static?
        # Ideally we instantiate it to access the PINE_TEMPLATE
        # We can use the strategy_cls directly if PINE_TEMPLATE is static, but export_to_pine is instance method in base?
        # Base method checks hasattr(self...
        
        try:
            # We need to pass valid data_handler because Strategy.__init__ accesses bars.symbol_list
            # Mock queue is fine.
            from queue import Queue
            temp_strat = self.strategy_cls(self.data_handler, Queue(), **best_params)
            
            # Merge defaults from the instance back into params for the template
            # This ensures keys like 'ema_filter' exist even if not optimized
            full_params = best_params.copy()
            # We assume strategy attributes match param names
            for k in ['ema_filter', 'adx_filter', 'atr_max_mult', 'sl_atr_mult', 'tp_atr_mult']:
                if hasattr(temp_strat, k):
                    full_params[k] = getattr(temp_strat, k)
            
            pine_code = temp_strat.export_to_pine(full_params)
            
            # Save Pine Script
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            pine_path = os.path.join(output_dir, f"{self.strategy_name}_Optimized.pine")
            with open(pine_path, "w") as f:
                f.write(pine_code)
            metrics['pine_script_path'] = pine_path
            print(f"      ✅ Saved: {pine_path}")
        except Exception as e:
            print(f"      ⚠️ Pine Gen Failed: {e}")

        # 5. The Skeptic (Denial Tests)
        print("    [E] The Skeptic (Statistical Denial)")
        # We need the vector engine class and strategy class
        # Assuming VectorizedNQORB for now or self.vector_strategy_cls
        skeptic = StrategySkeptic(
            vector_engine_cls=VectorEngine,
            vector_strategy_cls=self.vector_strategy_cls,
            params=best_params,
            initial_capital=self.initial_capital
        )
        # Need dataframe
        if sym in self.data_handler.symbol_data:
             df_skeptic = self.data_handler.symbol_data[sym]
             skeptic_res = skeptic.run_permutation_test(df_skeptic, n_sims=50) # 50 for speed
             metrics['skeptic'] = skeptic_res
             print(f"      🕵️ Verdict: {skeptic_res['verdict']} (p={skeptic_res['p_value']:.2f})")
             
             detrend_res = skeptic.run_detrended_test(df_skeptic)
             metrics['detrended'] = detrend_res
             print(f"      📉 Detrended: {detrend_res['verdict']}")
        
        # 6. Dashboard Generation
        print("    [F] Generating Interactive Dashboard")
        # Need Equity Curve from Portfolio. It is a list of dicts.
        eq_df = pd.DataFrame(portfolio.equity_curve)
        if not eq_df.empty:
            if 'datetime' in eq_df.columns:
                eq_df['datetime'] = pd.to_datetime(eq_df['datetime'])
                eq_df.set_index('datetime', inplace=True)
            
            dash = DashboardGenerator(
                strategy_name=self.strategy_name,
                params=best_params,
                stats={'Total Return': 0.0, 'Sharpe': 0.0}, # Fill properly if avail
                equity_curve=eq_df,
                skeptic_results=metrics.get('skeptic')
            )
            dash.generate(f"{self.strategy_name}_Report.html")

        return metrics

    def execute(self, wfo: bool = False):
        """Main Execution Flow"""
        self.load_data()
        
        best_params = {}
        final_stats = {}
        
        if wfo:
            # WFO Flow
            df_results, stitched_equity = self.run_optimization(wfo=True)
            if df_results.empty:
                print("No WFO results.")
                return
            
            # Aggregate WFO Metrics
            total_ret = 0.0
            if len(stitched_equity) > 0:
                # stitched_equity is list of series of returns.
                # Concatenate and cumprod
                full_curve = pd.concat(stitched_equity)
                # cumprod: (1+r).cumprod()
                # But returns are percent changes.
                combined = (1 + full_curve).prod() - 1.0
                total_ret = combined
            
            print(f"\n[3] WFO Complete. Stitched Return: {total_ret:.2%}")
            
            # Use LATEST params for "Best" (Current state)
            best_row = df_results.iloc[-1] # Last window
            best_params = best_row['params']
            print(f"    Latest Window Params: {best_params}")
            
            # Save WFO specific artifacts
            df_results.to_csv("outputs/wfo_results.csv")
            final_stats['Total Return'] = total_ret
            final_stats['WFO_Details'] = df_results.to_dict()
            
            # Run Check on Latest Params for Pine Gen
            # We skip full event cert for the promptness, or run it just for Robustness checks on *current* regime
            # Let's run robustness checks on the full history using the *latest* params?? 
            # NO. That would reproduce the "static" failure.
            # We simply generate Pine from latest.
            
            # Hack: Create a dummy portfolio to pass to robustness/pine gen
            # Or just call Pine Gen directly.
            
            # Generate Pine Script
            self.run_robustness_checks(Portfolio(self.data_handler, Queue(), 100000), best_params)
            
        else:
            # Standard Grid Search Flow
            df_results = self.run_optimization(wfo=False)
            
            if df_results.empty:
                print("No results found.")
                return

            # Select Best
            best_row = df_results.iloc[0]
            best_params = best_row.to_dict()
            
            # Clean params
            ignore = ['Total Return', 'Final Equity', 'Error', 'train_start', 'train_end', 'test_end', 'test_return', 'train_return']
            clean_params = {k: v for k, v in best_params.items() if k not in ignore and not isinstance(k, int)}
            best_params = clean_params # Ensure consistency
            
            # Certify
            print(f"\n    Best Vector Return: {best_row.get('Total Return', 0):.2%}")
            event_ret, event_eq, portfolio = self.run_event_certification(clean_params)
            final_stats['Total Return'] = event_ret
            final_stats['Ending Equity'] = event_eq
            
            # Robustness
            stats = self.run_robustness_checks(portfolio, clean_params)
            final_stats.update(stats)

        # Generate Insights Report with AI Agents
        print("\n[5] Generating Strategy Insights...")
        try:
            # Find strategy file path for code review
            import inspect
            strategy_file = None
            try:
                strategy_file = inspect.getfile(self.strategy_cls)
            except (TypeError, OSError):
                pass

            insights, ai_decision = self._generate_insights(
                final_stats,
                portfolio if not wfo else None,
                best_params=best_params,
                strategy_file=strategy_file
            )
            final_stats['_ai_decision'] = ai_decision
        except Exception as e:
            print(f"    [Insights] Could not generate: {e}")

        # Archive
        self.registry.save_run(
            self.strategy_name,
            self.symbol_list[0],
            self.interval,
            best_params,
            final_stats,
            (self.start_date, self.end_date),
            "Unified Pipeline Run"
        )
        print("\n[6] Run Complete & Archived.")

    def _generate_insights(self, stats: Dict, portfolio: Portfolio = None,
                           best_params: Dict = None, strategy_file: str = None):
        """Generate comprehensive insights report with AI agent integration."""

        # Prepare attribution data if portfolio available
        attribution = None
        significance = None
        wfo_analytics = None
        ai_decision = None
        code_review = None

        # Statistical significance from stored analysis
        if '_significance_analysis' in stats:
            significance = stats['_significance_analysis']

        # Trade attribution
        if portfolio and portfolio.trade_log:
            sym = self.symbol_list[0]
            if sym in self.data_handler.symbol_data:
                df = self.data_handler.symbol_data[sym]

                # Get regime series
                try:
                    regime_series = MarketRegimeDetector().get_regime_series(df)
                except Exception:
                    regime_series = None

                try:
                    attribution = analyze_trades(
                        portfolio.trade_log,
                        df,
                        regime_series
                    )
                except Exception as e:
                    print(f"    [Attribution] Could not analyze: {e}")

        # WFO analytics from stats
        if 'WFO_Details' in stats:
            try:
                wfo_df = pd.DataFrame(stats['WFO_Details'])
                wfo_analytics = analyze_wfo_results(wfo_df)
            except Exception:
                pass

        # AI Decision Agent
        if self.config.get('decision_agent', {}).get('enabled', False):
            print("    [AI] Running Decision Agent...")
            try:
                decision_agent = ValidationDecisionAgent(self.config)
                context = DecisionContext.from_stats(
                    stats,
                    self.strategy_name,
                    self.symbol_list[0],
                    best_params or {}
                )
                result = decision_agent.evaluate(context)
                ai_decision = result.to_dict()
                print(f"    [AI] Decision: {result.decision.value} (confidence: {result.confidence:.0%})")
                print(f"    [AI] Reasoning: {result.reasoning}")
            except Exception as e:
                print(f"    [AI] Decision Agent failed: {e}")

        # Code Review Agent
        if self.config.get('code_review_agent', {}).get('enabled', False) and strategy_file:
            print("    [AI] Running Code Review...")
            try:
                review_agent = CodeReviewAgent(self.config)
                review_result = review_agent.review_strategy(strategy_file)
                code_review = {
                    'passed': review_result.passed,
                    'issues': [
                        {
                            'severity': i.severity.value,
                            'category': i.category,
                            'message': i.message,
                            'line_number': i.line_number,
                            'suggestion': i.suggestion,
                        }
                        for i in review_result.issues
                    ],
                    'review_time_ms': review_result.review_time_ms,
                }
                status = "PASSED" if review_result.passed else "FAILED"
                print(f"    [AI] Code Review: {status} ({review_result.critical_count} critical, {review_result.warning_count} warnings)")
            except Exception as e:
                print(f"    [AI] Code Review failed: {e}")

        # Generate insights
        regime_stats = stats.get('regime_stats', {})

        output_path = f"outputs/{self.strategy_name}_insights"
        os.makedirs("outputs", exist_ok=True)

        insights = generate_insights(
            stats=stats,
            attribution=attribution,
            wfo_analytics=wfo_analytics,
            stat_significance=significance,
            regime_stats=regime_stats,
            ai_decision=ai_decision,
            code_review=code_review,
            output_path=output_path,
            print_console=True
        )

        print(f"    Insights saved to: {output_path}.html, {output_path}.md")

        return insights, ai_decision
