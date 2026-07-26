"""Numba-jitted utility functions for the multivariate imputer."""

import numpy as np
from numba import get_num_threads, njit, prange

#: Row-block sizing for the fused full-matrix kernels. Each parallel block owns
#: private accumulators, so the block count trades parallelism against scratch
#: memory instead of relying on atomics.
_MIN_BLOCK_ROWS = 1024
_SCRATCH_BUDGET_BYTES = 16_000_000


def row_blocks(n_rows: int, scratch_bytes_per_block: int) -> int:
    """Number of row blocks to split a full-matrix pass into.

    Enough blocks to keep every core busy, but never so many that the private
    per-block accumulators dominate memory.
    """
    by_rows = max(1, (n_rows + _MIN_BLOCK_ROWS - 1) // _MIN_BLOCK_ROWS)
    by_memory = max(1, _SCRATCH_BUDGET_BYTES // max(scratch_bytes_per_block, 1))
    return int(min(by_rows, 4 * get_num_threads(), by_memory))


@njit(boundscheck=False, cache=True, parallel=True)
def nan_mask_count_sum(x: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.bool_]:
    """NaN mask, observed count/sum per column and an infinity flag, in one pass.

    Fuses what used to be a `np.isnan` scan, an `np.isinf(x).any()` scan and the
    first pass of the column statistics into a single traversal of `x`.

    Args:
        x: The input matrix (floating point).
        n_blocks: Number of row blocks processed in parallel.

    Returns:
        A tuple `(mask_nan, counts, sums, has_inf)` where `counts` and `sums`
        cover the observed (non-NaN) cells of each column, accumulated in
        float64 regardless of the input precision.
    """
    m, n = x.shape
    mask = np.empty((m, n), dtype=np.bool_)
    block = (m + n_blocks - 1) // n_blocks
    counts_p = np.zeros((n_blocks, n), dtype=np.int64)
    sums_p = np.zeros((n_blocks, n), dtype=np.float64)
    inf_p = np.zeros(n_blocks, dtype=np.bool_)
    for b in prange(n_blocks):  # ty: ignore[not-iterable]
        i0 = b * block
        i1 = min(i0 + block, m)
        counts = counts_p[b]
        sums = sums_p[b]
        has_inf = False
        for i in range(i0, i1):
            row = x[i]
            mask_row = mask[i]
            for j in range(n):
                v = row[j]
                if np.isnan(v):
                    mask_row[j] = True
                else:
                    mask_row[j] = False
                    counts[j] += 1
                    sums[j] += v
                    if np.isinf(v):
                        has_inf = True
        inf_p[b] = has_inf
    return mask, counts_p.sum(axis=0), sums_p.sum(axis=0), inf_p.any()


@njit(boundscheck=False, cache=True, parallel=True)
def centered_sumsq(x: np.ndarray, means: np.ndarray, n_blocks: int) -> np.ndarray:
    """Per-column sum of squared deviations from `means` over observed cells."""
    m, n = x.shape
    block = (m + n_blocks - 1) // n_blocks
    ss_p = np.zeros((n_blocks, n), dtype=np.float64)
    for b in prange(n_blocks):  # ty: ignore[not-iterable]
        i0 = b * block
        i1 = min(i0 + block, m)
        ss = ss_p[b]
        for i in range(i0, i1):
            row = x[i]
            for j in range(n):
                v = row[j]
                if not np.isnan(v):
                    d = v - means[j]
                    ss[j] += d * d
    return ss_p.sum(axis=0)


@njit(boundscheck=False, cache=True, parallel=True)
def normalize_with_copy(x: np.ndarray, means: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `((x - means) / scales, x.copy())` in a single pass over `x`.

    The imputer needs the standardized matrix to fit models on and an
    original-scale copy to write imputed values into; producing both from one
    read replaces two full copies plus two in-place rescaling passes.
    """
    m, n = x.shape
    normalized = np.empty((m, n), dtype=x.dtype)
    original = np.empty((m, n), dtype=x.dtype)
    for i in prange(m):  # ty: ignore[not-iterable]
        row = x[i]
        row_normalized = normalized[i]
        row_original = original[i]
        for j in range(n):
            v = row[j]
            row_original[j] = v
            row_normalized[j] = (v - means[j]) / scales[j]
    return normalized, original


@njit(boundscheck=False, cache=True, parallel=True)
def normalize_in_place_with_copy(x: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Standardize `x` in place and return the original-scale copy.

    Same single read of `x` and same two writes as :func:`normalize_with_copy`,
    but only one new full-size array: for callers that own their input buffer
    this removes a whole copy of the matrix from the peak footprint.
    """
    m, n = x.shape
    original = np.empty((m, n), dtype=x.dtype)
    for i in prange(m):  # ty: ignore[not-iterable]
        row = x[i]
        row_original = original[i]
        for j in range(n):
            v = row[j]
            row_original[j] = v
            row[j] = (v - means[j]) / scales[j]
    return original


@njit(boundscheck=False, cache=True, parallel=True)
def copy_matrix(x: np.ndarray) -> np.ndarray:
    """Parallel equivalent of `x.copy()` for large matrices."""
    m, n = x.shape
    out = np.empty((m, n), dtype=x.dtype)
    for i in prange(m):  # ty: ignore[not-iterable]
        row = x[i]
        out_row = out[i]
        for j in range(n):
            out_row[j] = row[j]
    return out


@njit(boundscheck=False, cache=True, parallel=True)
def gather_columns_transposed(x: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pick columns out of `x` into a `(len(cols), n_rows)` array.

    Cache-blocked, so the strided reads stay local. The transpose of the result
    is Fortran-contiguous, which is the layout a pandas DataFrame stores
    internally: wrapping it costs nothing, where handing pandas a row-major
    slice makes it transpose the whole thing itself.
    """
    n_rows = x.shape[0]
    n_cols = len(cols)
    out = np.empty((n_cols, n_rows), dtype=x.dtype)
    block = 64
    n_row_blocks = (n_rows + block - 1) // block
    for bi in prange(n_row_blocks):  # ty: ignore[not-iterable]
        row_start = bi * block
        row_end = min(row_start + block, n_rows)
        for col_start in range(0, n_cols, block):
            col_end = min(col_start + block, n_cols)
            for i in range(row_start, row_end):
                row = x[i]
                for j in range(col_start, col_end):
                    out[j, i] = row[cols[j]]
    return out


@njit(boundscheck=False, cache=True)
def all_nan_columns(x: np.ndarray) -> np.ndarray:
    """Mask of columns that hold no observed value.

    Scans rows and stops as soon as every column has been seen observed, which
    is the overwhelmingly common case and makes this far cheaper than
    materializing `np.isnan(x)` to reduce it.
    """
    m, n = x.shape
    seen = np.zeros(n, dtype=np.bool_)
    remaining = n
    for i in range(m):
        row = x[i]
        for j in range(n):
            if not seen[j] and not np.isnan(row[j]):
                seen[j] = True
                remaining -= 1
        if remaining == 0:
            break
    return ~seen


@njit(boundscheck=False, cache=True, parallel=True)
def masked_cross_moments(
    x: np.ndarray,
    means: np.ndarray,
    cols: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise co-observation counts and centered cross-products for `cols`.

    For every target column `c` in `cols` and every column `j`, accumulates the
    number of rows where both are observed and the sum of the products of their
    deviations from `means` over those rows. Also returns each column's observed
    count and centered sum of squares, so a caller can derive correlations
    without a second traversal.
    """
    m, n = x.shape
    n_cols = len(cols)
    block = (m + n_blocks - 1) // n_blocks
    shared_p = np.zeros((n_blocks, n_cols, n), dtype=np.float64)
    cross_p = np.zeros((n_blocks, n_cols, n), dtype=np.float64)
    sumsq_p = np.zeros((n_blocks, n), dtype=np.float64)
    counts_p = np.zeros((n_blocks, n), dtype=np.int64)
    for b in prange(n_blocks):  # ty: ignore[not-iterable]
        i0 = b * block
        i1 = min(i0 + block, m)
        shared = shared_p[b]
        cross = cross_p[b]
        sumsq = sumsq_p[b]
        counts = counts_p[b]
        centered = np.empty(n, dtype=np.float64)
        observed = np.empty(n, dtype=np.float64)
        for i in range(i0, i1):
            row = x[i]
            for j in range(n):
                v = row[j]
                if np.isnan(v):
                    centered[j] = 0.0
                    observed[j] = 0.0
                else:
                    centered[j] = v - means[j]
                    observed[j] = 1.0
                    counts[j] += 1
            for j in range(n):
                sumsq[j] += centered[j] * centered[j]
            for t in range(n_cols):
                c = cols[t]
                if observed[c] != 0.0:
                    value = centered[c]
                    cross_t = cross[t]
                    shared_t = shared[t]
                    for j in range(n):
                        cross_t[j] += value * centered[j]
                        shared_t[j] += observed[j]
    return (
        shared_p.sum(axis=0),
        cross_p.sum(axis=0),
        sumsq_p.sum(axis=0),
        counts_p.sum(axis=0),
    )


@njit(boundscheck=False, cache=True)
def nan_positions(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def nan_cols_csc(iy: np.ndarray, ix: np.ndarray, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
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
def nan_positions_subset_cols(
    iy: np.ndarray,
    ix: np.ndarray,
    mask_subset_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def unique2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
