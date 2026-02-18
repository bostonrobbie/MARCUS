"""
Tests for type normalization utilities (Bug 1 fix).

These tests verify that the type_utils module correctly handles
mixed pandas/cuDF/numpy types in the backtesting engine.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'StrategyPipeline', 'src'))

from backtesting.type_utils import (
    ensure_pandas_series,
    ensure_pandas_dataframe,
    safe_iloc,
    safe_values,
    is_pandas_like,
    normalize_returns
)


class TestEnsurePandasSeries:
    """Tests for ensure_pandas_series function."""

    def test_pandas_series_passthrough(self, pandas_series_equity):
        """Pandas Series should pass through unchanged."""
        result = ensure_pandas_series(pandas_series_equity)

        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, pandas_series_equity)

    def test_numpy_array_conversion(self, numpy_array_equity):
        """Numpy array should convert to pandas Series."""
        result = ensure_pandas_series(numpy_array_equity, name='equity')

        assert isinstance(result, pd.Series)
        assert result.name == 'equity'
        np.testing.assert_array_equal(result.values, numpy_array_equity)

    def test_list_conversion(self):
        """List should convert to pandas Series."""
        data = [1, 2, 3, 4, 5]
        result = ensure_pandas_series(data)

        assert isinstance(result, pd.Series)
        assert list(result) == data

    def test_cudf_series_conversion(self, mock_cudf_series):
        """cuDF Series should convert via to_pandas()."""
        result = ensure_pandas_series(mock_cudf_series)

        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_scalar_value_conversion(self):
        """Scalar value should wrap in single-element Series."""
        result = ensure_pandas_series(42.0, name='value')

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] == 42.0

    def test_with_custom_index(self, numpy_array_equity):
        """Should apply custom index if provided."""
        custom_index = pd.date_range('2023-01-01', periods=5)
        result = ensure_pandas_series(numpy_array_equity, index=custom_index)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result.index) == 5


class TestEnsurePandasDataFrame:
    """Tests for ensure_pandas_dataframe function."""

    def test_pandas_dataframe_passthrough(self, sample_ohlcv_data):
        """Pandas DataFrame should pass through unchanged."""
        result = ensure_pandas_dataframe(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)

    def test_dict_conversion(self):
        """Dict should convert to DataFrame."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        result = ensure_pandas_dataframe(data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['a', 'b']


class TestSafeIloc:
    """Tests for safe_iloc function."""

    def test_pandas_series_iloc(self, pandas_series_equity):
        """Should use .iloc for pandas Series."""
        assert safe_iloc(pandas_series_equity, 0) == 100000
        assert safe_iloc(pandas_series_equity, -1) == 101500
        assert safe_iloc(pandas_series_equity, 2) == 101000

    def test_numpy_array_indexing(self, numpy_array_equity):
        """Should use [] for numpy arrays."""
        assert safe_iloc(numpy_array_equity, 0) == 100000
        assert safe_iloc(numpy_array_equity, -1) == 101500

    def test_list_indexing(self):
        """Should use [] for lists."""
        data = [10, 20, 30, 40, 50]

        assert safe_iloc(data, 0) == 10
        assert safe_iloc(data, -1) == 50

    def test_cudf_series_indexing(self, mock_cudf_series):
        """Should handle cuDF Series (converts to pandas first)."""
        result = safe_iloc(mock_cudf_series, -1)
        assert result == 101500

    def test_invalid_type_raises_error(self):
        """Should raise TypeError for non-indexable types."""
        with pytest.raises(TypeError):
            safe_iloc(42, 0)


class TestSafeValues:
    """Tests for safe_values function."""

    def test_numpy_array_passthrough(self, numpy_array_equity):
        """Numpy array should pass through."""
        result = safe_values(numpy_array_equity)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, numpy_array_equity)

    def test_pandas_series_extraction(self, pandas_series_equity):
        """Should extract .values from pandas Series."""
        result = safe_values(pandas_series_equity)

        assert isinstance(result, np.ndarray)

    def test_list_conversion(self):
        """Should convert list to numpy array."""
        data = [1, 2, 3]
        result = safe_values(data)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(data))


class TestIsPandasLike:
    """Tests for is_pandas_like function."""

    def test_pandas_series(self, pandas_series_equity):
        """Pandas Series should return True."""
        assert is_pandas_like(pandas_series_equity) is True

    def test_pandas_dataframe(self, sample_ohlcv_data):
        """Pandas DataFrame should return True."""
        assert is_pandas_like(sample_ohlcv_data) is True

    def test_numpy_array(self, numpy_array_equity):
        """Numpy array should return False."""
        assert is_pandas_like(numpy_array_equity) is False

    def test_cudf_like(self, mock_cudf_series):
        """Object with to_pandas() should return True."""
        assert is_pandas_like(mock_cudf_series) is True

    def test_plain_list(self):
        """Plain list should return False."""
        assert is_pandas_like([1, 2, 3]) is False


class TestNormalizeReturns:
    """Tests for normalize_returns function."""

    def test_normalizes_arrays_to_series(self, numpy_array_equity):
        """Should convert array values to pandas Series."""
        result_dict = {
            'equity_curve': numpy_array_equity,
            'signals': np.array([0, 1, 1, -1, 0]),
            'returns': np.array([0.0, 0.005, 0.005, -0.002, 0.007])
        }

        normalized = normalize_returns(result_dict)

        assert isinstance(normalized['equity_curve'], pd.Series)
        assert isinstance(normalized['signals'], pd.Series)
        assert isinstance(normalized['returns'], pd.Series)

    def test_preserves_scalars(self):
        """Should preserve scalar values."""
        result_dict = {
            'total_return': 0.15,
            'trade_count': 42,
            'name': 'test'
        }

        normalized = normalize_returns(result_dict)

        assert normalized['total_return'] == 0.15
        assert normalized['trade_count'] == 42
        assert normalized['name'] == 'test'

    def test_handles_none_values(self):
        """Should preserve None values."""
        result_dict = {
            'equity_curve': None,
            'signals': np.array([0, 1, 0])
        }

        normalized = normalize_returns(result_dict)

        assert normalized['equity_curve'] is None
        assert isinstance(normalized['signals'], pd.Series)

    def test_applies_custom_index(self, numpy_array_equity):
        """Should apply index to all Series."""
        custom_index = pd.date_range('2023-01-01', periods=5)
        result_dict = {'equity': numpy_array_equity}

        normalized = normalize_returns(result_dict, index=custom_index)

        assert isinstance(normalized['equity'].index, pd.DatetimeIndex)
