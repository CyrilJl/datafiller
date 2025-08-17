"""Numba-jitted utility functions for the multivariate imputer."""

from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(boundscheck=True, cache=True)
def nan_positions(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the positions of NaNs in a 2D array.

    Args:
        x: The input array.

    Returns:
        A tuple containing:
            - mask_nan (np.ndarray): A boolean mask of the same shape as x,
              True where NaNs are.
            - iy (np.ndarray): The row indices of NaNs.
            - ix (np.ndarray): The column indices of NaNs.
    """
    m, n = x.shape
    mask_nan = np.zeros((m, n), dtype=np.bool_)
    iy, ix = np.empty(m * n, dtype=np.uint32), np.empty(m * n, dtype=np.uint32)
    cnt = 0
    for i in range(m):
        for j in range(n):
            if np.isnan(x[i, j]):
                mask_nan[i, j] = True
                iy[cnt] = i
                ix[cnt] = j
                cnt += 1

    return mask_nan, iy[:cnt], ix[:cnt]


@njit(boundscheck=True, cache=True)
def nan_positions_subset(
    iy: np.ndarray,
    ix: np.ndarray,
    mask_subset_rows: np.ndarray,
    mask_subset_cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds NaN positions within a subset of rows and columns.

    Args:
        iy: Row indices of all NaNs in the original matrix.
        ix: Column indices of all NaNs in the original matrix.
        mask_subset_rows: A boolean mask for rows to consider.
        mask_subset_cols: A boolean mask for columns to consider.

    Returns:
        A tuple containing NaN positions within the subset.
    """
    n_nan = len(ix)
    size = min(n_nan, mask_subset_rows.sum() * mask_subset_cols.sum())
    sub_iy, sub_ix = np.empty(size, np.uint32), np.empty(size, np.uint32)
    cnt = 0
    for k in range(n_nan):
        row, col = iy[k], ix[k]
        if mask_subset_cols[col] and mask_subset_rows[row]:
            sub_iy[cnt] = row
            sub_ix[cnt] = col
            cnt += 1

    return sub_iy[:cnt], sub_ix[:cnt]


@njit(parallel=True, boundscheck=True, cache=True)
def _subset(X: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Extracts a subset of a matrix based on row and column indices.

    Args:
        X: The matrix to extract from.
        rows: The indices of rows to extract.
        columns: The indices of columns to extract.

    Returns:
        The extracted sub-matrix.
    """
    Xs = np.empty((len(rows), len(columns)), dtype=X.dtype)
    for i in prange(len(rows)):
        for j in range(len(columns)):
            Xs[i, j] = X[rows[i], columns[j]]
    return Xs


@njit(boundscheck=True, cache=True)
def _subset_one_column(X: np.ndarray, rows: np.ndarray, col: int) -> np.ndarray:
    Xs = np.empty(len(rows), dtype=X.dtype)
    for i in range(len(rows)):
        Xs[i] = X[rows[i], col]
    return Xs


@njit(boundscheck=True, cache=True)
def _imputable_rows(mask_nan: np.ndarray, col: int, mask_rows_to_impute: np.ndarray) -> np.ndarray:
    """Finds rows that have a NaN in a specific column and are marked for imputation.

    Args:
        mask_nan: The boolean mask of NaNs for the entire matrix.
        col: The column index to check.
        mask_rows_to_impute: A boolean mask of rows to be imputed.

    Returns:
        An array of row indices that can be imputed for the given column.
    """
    m = len(mask_nan)
    ret = np.empty(m, dtype=np.uint32)
    cnt = 0
    for k in range(m):
        if mask_nan[k, col] and mask_rows_to_impute[k]:
            ret[cnt] = k
            cnt += 1
    return ret[:cnt]


@njit(boundscheck=True, cache=True)
def _trainable_rows(mask_nan: np.ndarray, col: int) -> np.ndarray:
    """Finds rows that do not have a NaN in a specific column.

    These rows can be used for training a model to impute that column.

    Args:
        mask_nan: The boolean mask of NaNs for the entire matrix.
        col: The column index to check.

    Returns:
        An array of row indices that can be used for training.
    """
    m = len(mask_nan)
    ret = np.empty(m, dtype=np.uint32)
    cnt = 0
    for k in range(m):
        if not mask_nan[k, col]:
            ret[cnt] = k
            cnt += 1
    return ret[:cnt]


@njit(boundscheck=True, parallel=True)
def _mask_index_to_impute(size: int, to_impute: np.ndarray) -> np.ndarray:
    """Converts a list of indices to a boolean mask.

    Args:
        size: The size of the mask to create.
        to_impute: An array of indices.

    Returns:
        A boolean mask of length `size`.
    """
    ret = np.zeros(size, dtype=np.bool_)
    for i in prange(len(to_impute)):
        ret[to_impute[i]] = True
    return ret


def unique2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-compatible equivalent of `np.unique(x, return_inverse=True, axis=0)`."""
    x_struct = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx, inv = np.unique(x_struct, return_index=True, return_inverse=True)
    return x[idx], inv.ravel()


@njit(boundscheck=True, parallel=True)
def _index_to_mask(x: np.ndarray, n: int) -> np.ndarray:
    """Converts an array of indices to a boolean mask.

    Args:
        x: The indices to include in the mask.
        n: The size of the mask.

    Returns:
        A boolean mask of size `n`.
    """
    ret = np.zeros(n, dtype=np.bool_)
    for k in prange(len(x)):
        ret[x[k]] = True
    return ret
