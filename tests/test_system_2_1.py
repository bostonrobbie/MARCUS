
import sys
import os
import unittest
# Setup Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from backtesting.pipeline.runner import PipelineRunner
from backtesting.strategy import Strategy
from backtesting.schema import Bar
from backtesting.pipeline.config import config
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Mock Strategy
class MockStrategy(Strategy):
    """Buys on every bar."""
    def calculate_signals(self, event):
        if isinstance(event, Bar):
            self.buy('TEST', 1)

class TestSystem2_1(unittest.TestCase):
    def setUp(self):
        self.runner = PipelineRunner()
        self.test_dir = Path(PROJECT_ROOT) / 'data' / 'test_data_v2'
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Dummy Data
        dates = pd.date_range('2023-01-01', periods=100)
        df = pd.DataFrame({
            'open': np.linspace(100, 110, 100),
            'high': np.linspace(101, 111, 100),
            'low': np.linspace(99, 109, 100),
            'close': np.linspace(100, 110, 100), # Clean uptrend
            'volume': 1000
        }, index=dates)
        
        self.symbol = 'TEST'
        self.csv_path = self.test_dir / f'{self.symbol}.csv'
        df.to_csv(self.csv_path)
        
        # Config Patch
        config.raw_config['data'] = {'search_dirs': [str(self.test_dir)]}
        config.raw_config['optimization'] = {'gpu': False} # Force CPU for tests
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_verify_command(self):
        print("\n[TEST] Testing PipelineRunner.verify() (System 2.1 Integration)...")
        
        # Should run Cost Test and Noise Test
        result = self.runner.verify(
            strategy_cls=MockStrategy,
            symbol=self.symbol,
            params={}
        )
        
        self.assertTrue(result, "Verify method returned False")
        print("[PASS] System 2.1 Verify ran successfully.")

if __name__ == '__main__':
    unittest.main()
