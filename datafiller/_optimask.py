"""Finds optimal rectangular subsets of a matrix.

This module provides the `optimask` function, a low-level utility for finding
an optimal rectangular subset of a matrix that contains the fewest missing
values. This is used to select the best rows and columns for training an
imputation model.
"""

import numpy as np
from numba import bool_, njit, prange, uint32
from numba.types import UniTuple


@njit(bool_(uint32[:]), boundscheck=True, cache=True)
def is_decreasing(h: np.ndarray) -> bool:
    """Numba-jitted check if a 1D array is decreasing."""
    for i in range(len(h) - 1):
        if h[i] < h[i + 1]:
            return False
    return True


@njit(uint32[:](uint32[:], uint32[:], uint32), boundscheck=True, cache=True)
def groupby_max(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Numba-jitted equivalent of `np.maximum.at` for a groupby-max operation."""
    size_a = len(a)
    ret = np.zeros(n, dtype=np.uint32)
    for k in range(size_a):
        ak = a[k]
        ret[ak] = max(ret[ak], b[k] + 1)
    return ret


@njit(uint32[:](uint32[:], uint32[:], uint32[:], uint32, uint32), boundscheck=True, cache=True, parallel=True)
def diff1d(index, index_with_nan, permutation, index_split, max_val):
    """
    equivalent to np.setdiff1d(rows, rows_with_nan[p_rows][:j0])
    """
    to_exclude = np.zeros(max_val, dtype=np.bool_)

    for i in prange(index_split):
        val = index_with_nan[permutation[i]]
        to_exclude[val] = True

    count = 0
    for val in index:
        if val <= max_val and not to_exclude[val]:
            count += 1

    result = np.empty(count, dtype=index.dtype)
    pnt = 0
    for val in index:
        if val <= max_val and not to_exclude[val]:
            result[pnt] = val
            pnt += 1

    return result


@njit(UniTuple(uint32[:], 2)(uint32[:], uint32[:], uint32[:]), parallel=True, boundscheck=True, cache=True)
def apply_p_step(p_step, a, b):
    """Applies a permutation to two arrays."""
    ret_a = np.empty(a.size, dtype=np.uint32)
    ret_b = np.empty(b.size, dtype=np.uint32)
    for k in prange(a.size):
        pk = p_step[k]
        ret_a[k] = a[pk]
        ret_b[k] = b[pk]
    return ret_a, ret_b


@njit(uint32[:](uint32[:], uint32[:]), parallel=True, boundscheck=True, cache=True)
def numba_apply_permutation(p, x):
    """
    numba equivalent to:
        rank = np.empty_like(p)
        rank[p] = np.arange(len(p))
        # Use the rank array to permute x
        return rank[x]
    """
    n = p.size
    m = x.size
    rank = np.empty(n, dtype=np.uint32)
    result = np.empty(m, dtype=np.uint32)

    for i in prange(n):
        rank[p[i]] = i

    for i in prange(m):
        result[i] = rank[x[i]]
    return result


@njit((uint32[:], uint32[:]), parallel=True, boundscheck=True, cache=True)
def numba_apply_permutation_inplace(p: np.ndarray, x: np.ndarray):
    """Applies a permutation to an array in-place (Numba-jitted).

    Args:
        p: The permutation array.
        x: The array to be permuted.
    """
    n = p.size
    rank = np.empty(n, dtype=np.uint32)

    for i in prange(n):
        rank[p[i]] = i

    for i in prange(x.size):
        x[i] = rank[x[i]]


def apply_permutation(p: np.ndarray, x: np.ndarray, inplace: bool) -> np.ndarray | None:
    """Applies a permutation to an array.

    Args:
        p: The permutation array.
        x: The array to be permuted.
        inplace: If True, applies the permutation in place; otherwise,
            returns a new permuted array.

    Returns:
        The permuted array if `inplace` is False; otherwise, None.
    """
    if inplace:
        numba_apply_permutation_inplace(p, x)
    else:
        return numba_apply_permutation(p, x)


@njit(boundscheck=True, fastmath=True, nogil=True)
def _process_index(index: np.ndarray, num: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Compresses an array of indices into a dense, zero-based array.

    This is useful for creating a mapping from original indices to a smaller,
    contiguous set of indices, for example, when dealing with a subset of
    rows or columns.

    Args:
        index: The array of indices to process.
        num: The maximum value in the index array (e.g., total number of
            rows).

    Returns:
        A tuple containing:
            - ret (np.ndarray): The compressed index array.
            - table_inv (np.ndarray): The inverse mapping to get original
              indices back.
            - cnt (int): The number of unique indices.
    """
    size = len(index)
    table = np.zeros(num, dtype=np.uint32)
    table_inv = np.empty(num, dtype=np.uint32)
    ret = np.empty(size, dtype=np.uint32)
    cnt = np.uint32(0)

    for k in range(size):
        elem = index[k]
        if table[elem] == 0:
            cnt += 1
            table[elem] = cnt
            table_inv[cnt - 1] = elem
        ret[k] = table[elem] - 1

    return ret, table_inv[:cnt], cnt


def _get_largest_rectangle(heights: np.ndarray, m: int, n: int) -> tuple[int, int, int]:
    """Finds the largest rectangle under a histogram.

    This is used to find the largest area of non-missing values.

    Args:
        heights: The histogram of heights.
        m: The total number of rows.
        n: The total number of columns.

    Returns:
        A tuple containing the top-left corner and the area of the
        largest rectangle.
    """
    if n > len(heights):
        heights = np.concatenate((heights, np.array([0])))
    areas = (m - heights) * (n - np.arange(len(heights)))
    i0 = np.argmax(areas)
    return i0, heights[i0], areas[i0]


def optimask(
    iy: np.ndarray,
    ix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    global_matrix_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the largest rectangular area of a matrix for training.

    This is the main function of this module. It uses a pareto-optimal
    sorting strategy to find the largest rectangle of non-NaN values, which
    can then be used to train a model for imputation.

    Args:
        iy: Row indices of NaNs.
        ix: Column indices of NaNs.
        rows: The rows to consider for the mask.
        cols: The columns to consider for the mask.
        global_matrix_size: The shape of the original matrix (m, n).

    Returns:
        A tuple containing the rows and columns to keep for training.
    """
    m, n = global_matrix_size

    # Process row and column indices of NaNs
    iyp, rows_with_nan, m_nan = _process_index(index=iy, num=m)
    ixp, cols_with_nan, n_nan = _process_index(index=ix, num=n)

    # For each row with NaNs, find the maximum index of a column with a NaN
    hy = groupby_max(iyp, ixp, m_nan)
    # For each col with NaNs, find the maximum index of a row with a NaN
    hx = groupby_max(ixp, iyp, n_nan)

    p_rows = np.arange(m_nan, dtype=np.uint32)
    p_cols = np.arange(n_nan, dtype=np.uint32)
    is_pareto_ordered = False

    # Iteratively sort rows and columns to find a pareto-optimal ordering
    step = 0
    while not is_pareto_ordered and step < 16:
        kind = "stable" if step else "quicksort"
        axis = step % 2
        step += 1
        if axis == 0:  # Sort by rows
            p_step = (-hy).argsort(kind=kind).astype(np.uint32)
            apply_permutation(p_step, iyp, inplace=True)
            p_rows, hy = apply_p_step(p_step, p_rows, hy)
            hx = groupby_max(ixp, iyp, n_nan)
            is_pareto_ordered = is_decreasing(hx)
        else:  # Sort by columns
            p_step = (-hx).argsort(kind=kind).astype(np.uint32)
            apply_permutation(p_step, ixp, inplace=True)
            hy = groupby_max(iyp, ixp, m_nan)
            p_cols, hx = apply_p_step(p_step, p_cols, hx)
            is_pareto_ordered = is_decreasing(hy)

    if not is_pareto_ordered:
        raise ValueError(f"Pareto optimization did not converge after {step} steps.")

    # Find the largest rectangle in the pareto-optimal ordering
    i0, j0, area = _get_largest_rectangle(hx, len(rows), len(cols))

    if area == 0:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)

    # Determine which columns and rows to keep for imputation
    cols_to_keep = diff1d(cols, cols_with_nan, p_cols, i0, n)
    rows_to_keep = diff1d(rows, rows_with_nan, p_rows, j0, m)

    return rows_to_keep, cols_to_keep
