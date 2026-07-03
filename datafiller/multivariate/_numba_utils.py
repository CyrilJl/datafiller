"""Numba-jitted utility functions for the multivariate imputer."""

from typing import Tuple

import numpy as np
from numba import njit


@njit(boundscheck=False, cache=True)
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


@njit(boundscheck=False, cache=True)
def nan_cols_csc(iy: np.ndarray, ix: np.ndarray, n_cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Group NaN positions by column (CSC-like layout).

    Args:
        iy: Row indices of NaNs.
        ix: Column indices of NaNs.
        n_cols: Number of columns of the matrix.

    Returns:
        A tuple `(col_ptr, col_rows)` where the NaN rows of column `c` are
        `col_rows[col_ptr[c]:col_ptr[c + 1]]`.
    """
    n_nan = len(ix)
    col_ptr = np.zeros(n_cols + 1, dtype=np.int64)
    for k in range(n_nan):
        col_ptr[ix[k] + 1] += 1
    for j in range(n_cols):
        col_ptr[j + 1] += col_ptr[j]
    fill = col_ptr[:n_cols].copy()
    col_rows = np.empty(n_nan, dtype=np.uint32)
    for k in range(n_nan):
        c = ix[k]
        col_rows[fill[c]] = iy[k]
        fill[c] += 1
    return col_ptr, col_rows


@njit(boundscheck=False, cache=True)
def _mark_rows_with_nan_in_excluded(
    col_ptr: np.ndarray,
    col_rows: np.ndarray,
    excluded_cols: np.ndarray,
    hits: np.ndarray,
    stamp: np.ndarray,
    epoch: np.int64,
) -> None:
    """Count, per row, how many of its NaNs fall inside `excluded_cols`."""
    for j in range(len(excluded_cols)):
        c = excluded_cols[j]
        for k in range(col_ptr[c], col_ptr[c + 1]):
            r = col_rows[k]
            if stamp[r] != epoch:
                stamp[r] = epoch
                hits[r] = 1
            else:
                hits[r] += 1


@njit(boundscheck=False, cache=True)
def complete_rows_excluding(
    row_nan_count: np.ndarray,
    col_ptr: np.ndarray,
    col_rows: np.ndarray,
    excluded_cols: np.ndarray,
    hits: np.ndarray,
    stamp: np.ndarray,
    epoch: np.int64,
) -> np.ndarray:
    """Rows whose NaNs (if any) all fall inside `excluded_cols`.

    These are exactly the rows that are complete on the complement of
    `excluded_cols`, but the cost scales with the number of NaNs in the
    excluded columns instead of with `n_rows * n_usable_cols`.

    `hits` and `stamp` are scratch buffers of length `n_rows`; `epoch` must be
    a fresh value for each call so the buffers never need clearing.
    """
    _mark_rows_with_nan_in_excluded(col_ptr, col_rows, excluded_cols, hits, stamp, epoch)
    m = len(row_nan_count)
    cnt = 0
    for r in range(m):
        k = row_nan_count[r]
        if k == 0 or (stamp[r] == epoch and hits[r] == k):
            cnt += 1
    out = np.empty(cnt, dtype=np.uint32)
    p = 0
    for r in range(m):
        k = row_nan_count[r]
        if k == 0 or (stamp[r] == epoch and hits[r] == k):
            out[p] = r
            p += 1
    return out


@njit(boundscheck=False, cache=True)
def extra_rows_excluding(
    row_nan_count: np.ndarray,
    col_ptr: np.ndarray,
    col_rows: np.ndarray,
    excluded_cols: np.ndarray,
    hits: np.ndarray,
    stamp: np.ndarray,
    epoch: np.int64,
) -> Tuple[np.ndarray, np.int64]:
    """Like :func:`complete_rows_excluding` but only lists rows that have NaNs.

    Returns the rows whose NaNs all fall inside `excluded_cols` while having
    at least one NaN (the "extra" rows relative to the globally complete
    rows), together with the total count of complete rows.
    """
    _mark_rows_with_nan_in_excluded(col_ptr, col_rows, excluded_cols, hits, stamp, epoch)
    m = len(row_nan_count)
    n_extra = 0
    n_complete = 0
    for r in range(m):
        k = row_nan_count[r]
        if k == 0:
            n_complete += 1
        elif stamp[r] == epoch and hits[r] == k:
            n_extra += 1
    out = np.empty(n_extra, dtype=np.uint32)
    p = 0
    for r in range(m):
        k = row_nan_count[r]
        if k > 0 and stamp[r] == epoch and hits[r] == k:
            out[p] = r
            p += 1
    return out, np.int64(n_complete + n_extra)


@njit(boundscheck=False, cache=True)
def nan_positions_subset_cols(
    iy: np.ndarray,
    ix: np.ndarray,
    mask_subset_cols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find NaN positions for a prefiltered row set and a subset of columns."""
    n_nan = len(ix)
    sub_iy, sub_ix = np.empty(n_nan, np.uint32), np.empty(n_nan, np.uint32)
    cnt = 0
    for k in range(n_nan):
        col = ix[k]
        if mask_subset_cols[col]:
            sub_iy[cnt] = iy[k]
            sub_ix[cnt] = col
            cnt += 1

    return sub_iy[:cnt], sub_ix[:cnt]


@njit(boundscheck=False, cache=True)
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
    for i in range(len(rows)):
        for j in range(len(columns)):
            Xs[i, j] = X[rows[i], columns[j]]
    return Xs


@njit(boundscheck=False, cache=True)
def _subset_one_column(X: np.ndarray, rows: np.ndarray, col: int) -> np.ndarray:
    Xs = np.empty(len(rows), dtype=X.dtype)
    for i in range(len(rows)):
        Xs[i] = X[rows[i], col]
    return Xs


@njit(boundscheck=False, cache=True)
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


@njit(boundscheck=False, cache=True)
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


@njit(boundscheck=False)
def _mask_index_to_impute(size: int, to_impute: np.ndarray) -> np.ndarray:
    """Converts a list of indices to a boolean mask.

    Args:
        size: The size of the mask to create.
        to_impute: An array of indices.

    Returns:
        A boolean mask of length `size`.
    """
    ret = np.zeros(size, dtype=np.bool_)
    for i in range(len(to_impute)):
        ret[to_impute[i]] = True
    return ret


def unique2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-compatible equivalent of `np.unique(x, return_inverse=True, axis=0)`."""
    x_struct = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx, inv = np.unique(x_struct, return_index=True, return_inverse=True)
    return x[idx], inv.ravel()


@njit(boundscheck=False)
def _index_to_mask(x: np.ndarray, n: int) -> np.ndarray:
    """Converts an array of indices to a boolean mask.

    Args:
        x: The indices to include in the mask.
        n: The size of the mask.

    Returns:
        A boolean mask of size `n`.
    """
    ret = np.zeros(n, dtype=np.bool_)
    for k in range(len(x)):
        ret[x[k]] = True
    return ret
