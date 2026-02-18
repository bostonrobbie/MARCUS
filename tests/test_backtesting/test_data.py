"""
Tests for data handlers (DataHandler, MemoryDataHandler, SmartDataHandler).

These tests verify data loading, bar iteration, and data access methods.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))


class TestMemoryDataHandler:
    """Tests for MemoryDataHandler class."""

    def test_initialization(self, sample_ohlcv_dict):
        """Should initialize with symbol data."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)

        assert 'NQ' in handler.symbol_data
        assert 'ES' in handler.symbol_data
        assert handler.continue_backtest is True

    def test_update_bars_returns_true_when_data_available(self, sample_ohlcv_dict):
        """Should return True when bars are available."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)

        result = handler.update_bars()

        assert result is True

    def test_update_bars_returns_false_when_exhausted(self):
        """Should return False when no more data."""
        from backtesting.data import MemoryDataHandler

        # Create single-row data
        single_bar = pd.DataFrame({
            'Open': [100.0],
            'High': [101.0],
            'Low': [99.0],
            'Close': [100.5],
            'Volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='1min'))

        handler = MemoryDataHandler({'NQ': single_bar})

        # First update consumes the only bar
        handler.update_bars()

        # Second update should return False
        result = handler.update_bars()

        assert result is False
        assert handler.continue_backtest is False

    def test_get_latest_bar(self, sample_ohlcv_dict):
        """Should return most recent bar."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)
        handler.update_bars()

        bar = handler.get_latest_bar('NQ')

        assert bar is not None
        assert bar.symbol == 'NQ'
        assert hasattr(bar, 'open')
        assert hasattr(bar, 'close')

    def test_get_latest_bar_returns_none_before_update(self, sample_ohlcv_dict):
        """Should return None if no bars loaded yet."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)

        bar = handler.get_latest_bar('NQ')

        assert bar is None

    def test_get_latest_bar_returns_none_for_unknown_symbol(self, sample_ohlcv_dict):
        """Should return None for unknown symbol."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)
        handler.update_bars()

        bar = handler.get_latest_bar('UNKNOWN')

        assert bar is None

    def test_get_latest_bars_single(self, sample_ohlcv_dict):
        """Should return list with single bar."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)
        handler.update_bars()

        bars = handler.get_latest_bars('NQ', N=1)

        assert len(bars) == 1
        assert bars[0].symbol == 'NQ'

    def test_get_latest_bars_multiple(self, sample_ohlcv_dict):
        """Should return multiple bars."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)

        # Update multiple times
        for _ in range(5):
            handler.update_bars()

        bars = handler.get_latest_bars('NQ', N=3)

        assert len(bars) == 3

    def test_get_latest_bars_empty_for_unknown_symbol(self, sample_ohlcv_dict):
        """Should return empty list for unknown symbol."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)
        handler.update_bars()

        bars = handler.get_latest_bars('UNKNOWN', N=5)

        assert bars == []

    def test_bar_fields_correct(self, sample_ohlcv_dict):
        """Should create bars with correct field values."""
        from backtesting.data import MemoryDataHandler

        handler = MemoryDataHandler(sample_ohlcv_dict)
        handler.update_bars()

        bar = handler.get_latest_bar('NQ')
        first_row = sample_ohlcv_dict['NQ'].iloc[0]

        assert bar.open == first_row['Open']
        assert bar.high == first_row['High']
        assert bar.low == first_row['Low']
        assert bar.close == first_row['Close']
        assert bar.volume == first_row['Volume']

    def test_iterates_through_all_bars(self):
        """Should iterate through all bars in order."""
        from backtesting.data import MemoryDataHandler

        data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [100, 200, 300]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))

        handler = MemoryDataHandler({'TEST': data})

        closes = []
        while handler.continue_backtest:
            if handler.update_bars():
                bar = handler.get_latest_bar('TEST')
                closes.append(bar.close)

        assert closes == [100.5, 101.5, 102.5]


class TestSmartDataHandler:
    """Tests for SmartDataHandler class."""

    @patch('backtesting.data.yf', None)
    def test_raises_error_when_data_not_found(self, tmp_path):
        """Should raise ValueError when no data source available."""
        from backtesting.data import SmartDataHandler

        with pytest.raises(ValueError, match="Could not find"):
            SmartDataHandler(
                symbol_list=['NONEXISTENT'],
                search_dirs=[str(tmp_path)]
            )

    def test_loads_from_csv(self, tmp_path):
        """Should load data from CSV file."""
        from backtesting.data import SmartDataHandler

        # Create test CSV
        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10, freq='1D'),
            'Open': np.random.uniform(100, 110, 10),
            'High': np.random.uniform(110, 120, 10),
            'Low': np.random.uniform(90, 100, 10),
            'Close': np.random.uniform(100, 110, 10),
            'Volume': np.random.randint(1000, 10000, 10)
        })
        csv_path = tmp_path / 'TEST.csv'
        data.to_csv(csv_path, index=False)

        handler = SmartDataHandler(
            symbol_list=['TEST'],
            search_dirs=[str(tmp_path)]
        )

        assert 'TEST' in handler.symbol_data
        assert len(handler.symbol_data['TEST']) == 10

    def test_date_filtering(self, tmp_path):
        """Should filter data by date range."""
        from backtesting.data import SmartDataHandler

        # Create test CSV with 30 days
        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='1D'),
            'Open': np.random.uniform(100, 110, 30),
            'High': np.random.uniform(110, 120, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(100, 110, 30),
            'Volume': np.random.randint(1000, 10000, 30)
        })
        csv_path = tmp_path / 'TEST.csv'
        data.to_csv(csv_path, index=False)

        handler = SmartDataHandler(
            symbol_list=['TEST'],
            search_dirs=[str(tmp_path)],
            start_date=datetime(2024, 1, 10),
            end_date=datetime(2024, 1, 20)
        )

        # Should only have 11 days (10th to 20th inclusive)
        assert len(handler.symbol_data['TEST']) == 11


class TestDataHandlerAbstract:
    """Tests for DataHandler abstract base class."""

    def test_abstract_methods(self):
        """Should define required abstract methods."""
        from backtesting.data import DataHandler

        assert hasattr(DataHandler, 'get_latest_bar')
        assert hasattr(DataHandler, 'get_latest_bars')
        assert hasattr(DataHandler, 'update_bars')

    def test_cannot_instantiate_directly(self):
        """Should not be able to instantiate abstract class."""
        from backtesting.data import DataHandler

        with pytest.raises(TypeError):
            DataHandler()
