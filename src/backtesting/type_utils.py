"""
Type normalization utilities for mixed pandas/cuDF/numpy environments.

This module provides helper functions to ensure consistent data types
across GPU (cuDF) and CPU (pandas/numpy) backends in the backtesting engine.
"""
import pandas as pd
import numpy as np
from typing import Any, Optional, Union


def ensure_pandas_series(
    data: Any,
    index: Optional[Any] = None,
    name: Optional[str] = None
) -> pd.Series:
    """
    Convert any array-like to pandas Series.

    Handles pandas Series, cuDF Series, numpy arrays, and lists.
    This ensures consistent return types regardless of GPU acceleration status.

    Args:
        data: Input data - can be pandas Series, cuDF Series, numpy ndarray, or list
        index: Optional index for the resulting Series
        name: Optional name for the resulting Series

    Returns:
        pandas.Series with the same data

    Examples:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> s = ensure_pandas_series(arr, name='equity')
        >>> isinstance(s, pd.Series)
        True
    """
    # Already a pandas Series - return as-is or with updated index/name
    if isinstance(data, pd.Series):
        if index is not None or name is not None:
            return pd.Series(data.values, index=index if index is not None else data.index, name=name or data.name)
        return data

    # cuDF Series - convert via to_pandas() method
    # Check by method existence to avoid importing cuDF
    if hasattr(data, 'to_pandas') and callable(getattr(data, 'to_pandas')):
        pandas_data = data.to_pandas()
        if index is not None or name is not None:
            return pd.Series(pandas_data.values, index=index, name=name)
        return pandas_data

    # numpy ndarray or list - construct new Series
    if isinstance(data, (np.ndarray, list)):
        return pd.Series(data, index=index, name=name)

    # Scalar value - wrap in single-element Series
    if np.isscalar(data):
        return pd.Series([data], index=[0] if index is None else index[:1], name=name)

    # Fallback: try to convert to numpy first, then to Series
    try:
        arr = np.asarray(data)
        return pd.Series(arr, index=index, name=name)
    except Exception:
        raise TypeError(f"Cannot convert type {type(data).__name__} to pandas Series")


def ensure_pandas_dataframe(
    data: Any,
    index: Optional[Any] = None,
    columns: Optional[Any] = None
) -> pd.DataFrame:
    """
    Convert any DataFrame-like to pandas DataFrame.

    Args:
        data: Input data - pandas DataFrame, cuDF DataFrame, dict, or 2D array
        index: Optional index for the resulting DataFrame
        columns: Optional column names

    Returns:
        pandas.DataFrame with the same data
    """
    # Already a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        if index is not None or columns is not None:
            return pd.DataFrame(
                data.values,
                index=index if index is not None else data.index,
                columns=columns if columns is not None else data.columns
            )
        return data

    # cuDF DataFrame - convert via to_pandas()
    if hasattr(data, 'to_pandas') and callable(getattr(data, 'to_pandas')):
        pandas_data = data.to_pandas()
        if index is not None or columns is not None:
            return pd.DataFrame(
                pandas_data.values,
                index=index if index is not None else pandas_data.index,
                columns=columns if columns is not None else pandas_data.columns
            )
        return pandas_data

    # Dict, numpy array, or list - construct new DataFrame
    return pd.DataFrame(data, index=index, columns=columns)


def safe_iloc(data: Any, idx: int) -> Any:
    """
    Safely access element by integer location, handling all array types.

    This function provides a unified interface for accessing elements by
    integer position across pandas Series, numpy arrays, cuDF Series, and lists.

    Args:
        data: Series-like or array-like object
        idx: Integer index (supports negative indexing)

    Returns:
        Scalar value at the specified index

    Raises:
        TypeError: If the data type doesn't support indexing
        IndexError: If the index is out of bounds

    Examples:
        >>> safe_iloc(pd.Series([10, 20, 30]), -1)
        30
        >>> safe_iloc(np.array([10, 20, 30]), 0)
        10
    """
    # pandas Series/DataFrame - use iloc
    if hasattr(data, 'iloc'):
        return data.iloc[idx]

    # cuDF Series - may have different accessor
    if hasattr(data, 'to_pandas'):
        return data.to_pandas().iloc[idx]

    # numpy array, list, or other sequence - use direct indexing
    if hasattr(data, '__getitem__'):
        return data[idx]

    raise TypeError(f"Cannot index into type {type(data).__name__}")


def safe_values(data: Any) -> np.ndarray:
    """
    Safely extract numpy array from any Series-like object.

    Args:
        data: Series-like object (pandas, cuDF, or numpy array)

    Returns:
        numpy.ndarray with the values
    """
    # Already numpy array
    if isinstance(data, np.ndarray):
        return data

    # pandas Series/DataFrame - use .values
    if hasattr(data, 'values'):
        values = data.values
        # cuDF returns cupy array, convert to numpy
        if hasattr(values, 'get'):
            return values.get()
        return values

    # cuDF with to_pandas
    if hasattr(data, 'to_pandas'):
        return data.to_pandas().values

    # List or other
    return np.asarray(data)


def is_pandas_like(data: Any) -> bool:
    """
    Check if data is a pandas-like object (Series or DataFrame).

    Args:
        data: Any object

    Returns:
        True if the object is a pandas Series/DataFrame or cuDF equivalent
    """
    return (
        isinstance(data, (pd.Series, pd.DataFrame)) or
        hasattr(data, 'to_pandas')
    )


def normalize_returns(result_dict: dict, index: Optional[Any] = None) -> dict:
    """
    Normalize all array-like values in a result dictionary to pandas types.

    This is designed to be used at the end of VectorEngine.run() to ensure
    consistent return types.

    Args:
        result_dict: Dictionary with string keys and array-like values
        index: Optional index to apply to all Series

    Returns:
        Dictionary with all values converted to pandas Series
    """
    normalized = {}
    for key, value in result_dict.items():
        if value is None:
            normalized[key] = None
        elif isinstance(value, (dict, str, int, float, bool)):
            normalized[key] = value
        else:
            try:
                normalized[key] = ensure_pandas_series(value, index=index, name=key)
            except TypeError:
                normalized[key] = value
    return normalized
