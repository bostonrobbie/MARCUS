
import sys
import os
import pandas as pd
from datetime import datetime

# Setup Path
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('scripts'))

# We need to mock 'run_strategy' since it's hard to verify optimization loop directly without long run.
# Or we can just import the components and run them manually like in runner.py

from backtesting.validation import ValidationSuite, SensitivityTester
from backtesting.regime import MarketRegimeDetector
from backtesting.data import SmartDataHandler

def mock_runner_func(strategy_cls, symbol_list, data_dirs, **kwargs):
    # Returns dummy stats
    p = kwargs.get('param1', 10)
    ret = 0.1 * (p / 10.0) # Sensitivity: Return changes with param
    return {'Total Return': ret, 'Sharpe Ratio': 1.5, 'Max Drawdown': -0.05}, None

def test_sensitivity():
    print("Testing SensitivityTester...")
    tester = SensitivityTester(
        strategy_cls=str, # Dummy
        symbol_list=['SPY'],
        data_dirs=[],
        base_params={'param1': 10},
        runner_func=mock_runner_func
    )
    tester.run()

def test_regime_analysis():
    print("\nTesting Regime Analysis...")
    # Create Dummy Equity Curve
    dates = pd.date_range('2023-01-01', periods=100)
    equity = [100000 * (1.01 ** i) for i in range(100)]
    df = pd.DataFrame({'equity': equity}, index=dates)
    
    # Create Dummy Regimes
    from backtesting.regime import Regime
    # 50 days Bull, 50 days Chop
    regimes = [Regime.BULL_TREND]*50 + [Regime.CHOPPY_QUIET]*50
    regime_series = pd.Series(regimes, index=dates)
    
    suite = ValidationSuite({}, [])
    suite.analyze_regimes(df, regime_series)

if __name__ == "__main__":
    test_sensitivity()
    test_regime_analysis()
