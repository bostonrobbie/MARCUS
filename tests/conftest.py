"""
Shared pytest fixtures for NHHF test suite.

This module provides common fixtures used across all test modules including:
- Sample OHLCV data generation
- Schema object fixtures (Bar, Signal, Order, Fill events)
- Component fixtures (DataHandler, Portfolio, etc.)
- Mock fixtures (Ollama, GPU)
- Temporary file fixtures
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from queue import Queue
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import json

# Add project paths for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "StrategyPipeline", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Fund_Manager"))


# ============================================
# Sample Data Fixtures
# ============================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing (100 bars, 5-min intervals)."""
    dates = pd.date_range(
        start='2023-01-03 09:30:00',
        periods=100,
        freq='5min'
    )
    np.random.seed(42)

    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.abs(np.random.randn(100) * 0.3)
    low = close - np.abs(np.random.randn(100) * 0.3)
    open_price = close + np.random.randn(100) * 0.2
    volume = np.random.randint(1000, 10000, 100)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def sample_ohlcv_dict(sample_ohlcv_data):
    """Sample data as dict for MemoryDataHandler."""
    return {'NQ': sample_ohlcv_data}


@pytest.fixture
def multi_day_ohlcv_data():
    """Generate multi-day OHLCV data for ORB testing (5 trading days)."""
    all_data = []

    for day_offset in range(5):
        base_date = datetime(2023, 1, 3 + day_offset)

        # Skip weekends
        if base_date.weekday() >= 5:
            continue

        # Generate intraday data (9:30 AM to 4:00 PM, 15-min bars)
        times = pd.date_range(
            start=base_date.replace(hour=9, minute=30),
            end=base_date.replace(hour=16, minute=0),
            freq='15min'
        )

        np.random.seed(42 + day_offset)
        n = len(times)
        base_price = 15000 + day_offset * 50
        close = base_price + np.cumsum(np.random.randn(n) * 10)

        day_data = pd.DataFrame({
            'Open': close + np.random.randn(n) * 5,
            'High': close + np.abs(np.random.randn(n) * 8),
            'Low': close - np.abs(np.random.randn(n) * 8),
            'Close': close,
            'Volume': np.random.randint(1000, 5000, n)
        }, index=times)

        all_data.append(day_data)

    return pd.concat(all_data)


@pytest.fixture
def empty_ohlcv_data():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


# ============================================
# Type Utils Fixtures
# ============================================

@pytest.fixture
def numpy_array_equity():
    """Sample equity curve as numpy array."""
    return np.array([100000, 100500, 101000, 100800, 101500])


@pytest.fixture
def pandas_series_equity():
    """Sample equity curve as pandas Series."""
    return pd.Series([100000, 100500, 101000, 100800, 101500], name='equity')


@pytest.fixture
def mock_cudf_series():
    """Mock cuDF Series with to_pandas() method."""
    class MockCudfSeries:
        def __init__(self, data):
            self._data = np.array(data)

        def to_pandas(self):
            return pd.Series(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def __len__(self):
            return len(self._data)

    return MockCudfSeries([100000, 100500, 101000, 100800, 101500])


# ============================================
# Component Fixtures
# ============================================

@pytest.fixture
def events_queue():
    """Create a fresh events queue."""
    return Queue()


@pytest.fixture
def mock_data_handler(sample_ohlcv_dict):
    """Create a MemoryDataHandler with sample data."""
    from backtesting.data import MemoryDataHandler
    return MemoryDataHandler(sample_ohlcv_dict)


@pytest.fixture
def instrument_specs():
    """Standard instrument specifications for NQ futures."""
    return {
        'NQ': {
            'multiplier': 20.0,
            'tick_size': 0.25,
            'commission': 2.05
        }
    }


# ============================================
# Mock Ollama Fixture
# ============================================

@pytest.fixture
def mock_ollama():
    """Mock Ollama API responses for agent testing."""
    with patch('ollama.chat') as mock_chat, \
         patch('ollama.embeddings') as mock_embed:

        # Default approval response
        mock_chat.return_value = {
            'message': {'content': 'APPROVED: Strategy passes all risk checks.'}
        }

        # Mock embeddings
        mock_embed.return_value = {
            'embedding': [0.1] * 1024
        }

        yield {'chat': mock_chat, 'embeddings': mock_embed}


@pytest.fixture
def mock_ollama_veto():
    """Mock Ollama to return VETO response."""
    with patch('ollama.chat') as mock_chat:
        mock_chat.return_value = {
            'message': {'content': 'VETO: Max drawdown exceeds 25% threshold.'}
        }
        yield mock_chat


# ============================================
# Temporary File Fixtures
# ============================================

@pytest.fixture
def temp_data_dir(sample_ohlcv_data):
    """Create temporary directory with sample CSV data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save sample data as CSV
        csv_path = os.path.join(tmpdir, 'NQ.csv')
        df = sample_ohlcv_data.reset_index()
        df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df.to_csv(csv_path, index=False)
        yield tmpdir


@pytest.fixture
def temp_portfolio_file():
    """Create temporary portfolio.json file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "strategies": [],
            "total_capital": 100000.0,
            "last_updated": None,
            "portfolio_metrics": {
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "num_strategies": 0
            }
        }, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_sandbox_dir():
    """Create temporary sandbox directory for generated strategies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================
# GPU Mock Fixtures
# ============================================

@pytest.fixture
def mock_gpu_unavailable():
    """Mock GPU as unavailable for consistent CPU-only testing."""
    with patch.dict('sys.modules', {'cudf': None, 'cupy': None}):
        try:
            from backtesting import accelerate
            with patch.object(accelerate, 'GPU_AVAILABLE', False):
                yield
        except ImportError:
            yield


@pytest.fixture
def mock_gpu_available():
    """Mock GPU as available (for GPU-specific tests)."""
    try:
        from backtesting import accelerate
        with patch.object(accelerate, 'GPU_AVAILABLE', True):
            yield
    except ImportError:
        yield


# ============================================
# Code Validation Fixtures
# ============================================

@pytest.fixture
def valid_strategy_code():
    """Sample valid strategy code."""
    return '''
import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue

class TestStrategy:
    def __init__(self, data_handler, events, **params):
        self.bars = data_handler
        self.events = events

    def calculate_signals(self, event):
        return 0
'''


@pytest.fixture
def invalid_syntax_code():
    """Code with syntax errors."""
    return '''
import pandas as pd

def broken_function(
    return 42
'''


@pytest.fixture
def missing_imports_code():
    """Code that uses modules without importing them."""
    return '''
class TestStrategy:
    def run(self):
        df = pd.DataFrame()
        arr = np.array([1, 2, 3])
        return df, arr
'''


@pytest.fixture
def wrong_time_import_code():
    """Code with incorrect time import."""
    return '''
import time

start_time = time(9, 30)
end_time = time(15, 45)
'''


# ============================================
# Portfolio Test Fixtures
# ============================================

@pytest.fixture
def sample_portfolio_summary():
    """Sample portfolio summary dict."""
    return {
        'total_strategies': 3,
        'active_count': 2,
        'paused_count': 1,
        'deprecated_count': 0,
        'total_capital': 100000.0,
        'portfolio_metrics': {
            'sharpe': 1.5,
            'total_return': 15.5,
            'max_drawdown': -10.0,
            'num_strategies': 2
        },
        'last_updated': '2023-01-03T10:00:00',
        'strategies': [
            {'name': 'Strategy1', 'weight': 50.0, 'sharpe': 1.8, 'return_pct': 20.0, 'status': 'active'},
            {'name': 'Strategy2', 'weight': 30.0, 'sharpe': 1.2, 'return_pct': 10.0, 'status': 'active'},
            {'name': 'Strategy3', 'weight': 20.0, 'sharpe': 0.5, 'return_pct': -5.0, 'status': 'paused'},
        ]
    }


@pytest.fixture
def empty_portfolio_summary():
    """Portfolio summary with no strategies (edge case)."""
    return {
        'total_strategies': 0,
        'active_count': 0,
        'paused_count': 0,
        'deprecated_count': 0,
        'total_capital': 100000.0,
        'portfolio_metrics': {},
        'last_updated': None,
        'strategies': []
    }
