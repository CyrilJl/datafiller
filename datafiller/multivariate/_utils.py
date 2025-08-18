"""Utility functions for the multivariate imputer."""

from typing import Iterable

import numpy as np


def _process_to_impute(size: int, to_impute: None | int | Iterable[int]) -> np.ndarray:
    """Processes the `to_impute` argument into a numpy array of indices.

    Args:
        size: The total number of items (e.g., rows or columns).
        to_impute: The user-provided argument.

    Returns:
        An array of indices to impute.
    """
    if to_impute is None:
        return np.arange(size)
    if isinstance(to_impute, int):
        return np.array([to_impute])
    else:
        return np.array(to_impute)


def _dataframe_rows_to_impute_to_indices(rows_to_impute, index):
    """Converts row labels to integer indices for a DataFrame."""
    if rows_to_impute is None:
        return np.arange(len(index))

    to_impute_list = (
        [rows_to_impute]
        if not isinstance(rows_to_impute, Iterable) or isinstance(rows_to_impute, str)
        else list(rows_to_impute)
    )

    indexer = index.get_indexer(to_impute_list)
    if np.any(indexer == -1):
        missing = [l for l, i in zip(to_impute_list, indexer) if i == -1]
        raise ValueError(f"Row labels not found in index: {missing}")
    return indexer


def _dataframe_cols_to_impute_to_indices(cols_to_impute, columns):
    """Converts column labels to integer indices for a DataFrame."""
    if cols_to_impute is None:
        return np.arange(len(columns))

    to_impute_list = (
        [cols_to_impute]
        if not isinstance(cols_to_impute, Iterable) or isinstance(cols_to_impute, str)
        else list(cols_to_impute)
    )

    indexer = columns.get_indexer(to_impute_list)
    if np.any(indexer == -1):
        missing = [l for l, i in zip(to_impute_list, indexer) if i == -1]
        raise ValueError(f"Column labels not found in columns: {missing}")
    return indexer


def _validate_input(
    x: np.ndarray,
    rows_to_impute: None | int | Iterable[int],
    cols_to_impute: None | int | Iterable[int],
    n_nearest_features: None | float | int,
) -> int:
    """Validates the inputs to the `__call__` method.

    Args:
        x: The input data matrix.
        rows_to_impute: Rows to impute.
        cols_to_impute: Columns to impute.
        n_nearest_features: Number of features to use for imputation.

    Returns:
        The validated and processed number of nearest features.

    Raises:
        ValueError: If any of the inputs are invalid.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy array.")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D array, but got {x.ndim} dimensions.")
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"x must have a numeric dtype, but got {x.dtype}.")
    if np.isinf(x).any():
        raise ValueError("x cannot contain infinity.")

    m, n = x.shape

    if rows_to_impute is not None:
        if isinstance(rows_to_impute, int):
            rows_to_impute = [rows_to_impute]
        if isinstance(rows_to_impute, np.ndarray):
            if not np.issubdtype(rows_to_impute.dtype, np.integer):
                raise ValueError(f"rows_to_impute must have an integer dtype, but got {rows_to_impute.dtype}.")
            if not (np.all(rows_to_impute >= 0) and np.all(rows_to_impute < m)):
                raise ValueError(f"rows_to_impute must be a list of integers between 0 and {m - 1}.")
        elif not all(isinstance(i, int) for i in rows_to_impute) or not all(0 <= i < m for i in rows_to_impute):
            raise ValueError(f"rows_to_impute must be a list of integers between 0 and {m - 1}.")

    if cols_to_impute is not None:
        if isinstance(cols_to_impute, int):
            cols_to_impute = [cols_to_impute]
        if not all(isinstance(i, (int, np.integer)) for i in cols_to_impute) or not all(
            0 <= i < n for i in cols_to_impute
        ):
            raise ValueError(f"cols_to_impute must be a list of integers between 0 and {n - 1}.")

    if n_nearest_features is not None:
        if isinstance(n_nearest_features, float):
            if not (0 < n_nearest_features <= 1.0):
                raise ValueError("If n_nearest_features is a float, it must be in (0, 1].")
            n_nearest_features = int(n_nearest_features * n)
            if n_nearest_features == 0:
                raise ValueError("n_nearest_features resulted in 0 features to select.")
        if not isinstance(n_nearest_features, int):
            raise ValueError("n_nearest_features must be an int or float.")
        if not (0 < n_nearest_features <= n):
            raise ValueError(f"n_nearest_features must be between 1 and {n}.")

    return n_nearest_features
