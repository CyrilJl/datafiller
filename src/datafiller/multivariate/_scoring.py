import numpy as np

from ._numba_utils import masked_cross_moments, nan_mask_count_sum, row_blocks

#: Above this many target columns the per-block accumulators of the fused kernel
#: stop paying for themselves and the chunked BLAS formulation wins.
_MAX_FUSED_TARGETS = 64

#: Row chunk of the BLAS fallback. Sized so the two working buffers stay in
#: cache instead of materializing full-size temporaries.
_CHUNK_ROWS = 2048


@np.errstate(all="ignore")
def scoring(
    x: np.ndarray,
    cols_to_impute: np.ndarray,
    mask_nan: np.ndarray | None = None,
    column_means: np.ndarray | None = None,
) -> np.ndarray:
    """Calculates a score for each feature pair to guide feature selection.

    The score is based on the correlation and the proportion of shared
    non-NaN values. Mathematically this matches correlating the mean
    pre-imputed matrix, but it is computed from masked moments so no
    full-size temporary is ever materialized.

    Args:
        x: The input data matrix.
        cols_to_impute: The columns that are candidates for imputation.
        mask_nan: Unused; accepted for backwards compatibility with callers
            that already hold `np.isnan(x)`.
        column_means: Optional per-column means over observed cells. When the
            caller already knows them (e.g. because it just standardized `x`),
            passing them removes one traversal of the matrix.

    Returns:
        A score matrix of shape `(len(cols_to_impute), x.shape[1])`.
    """
    del mask_nan  # the fused kernels read `x` directly
    m, n = x.shape
    cols = np.asarray(cols_to_impute, dtype=np.int64).ravel()

    if not _can_fuse(x, cols):
        return _scoring_chunked(x, cols, column_means)

    if column_means is None:
        _, counts, sums, _ = nan_mask_count_sum(x, row_blocks(m, 8 * n))
        column_means = np.where(counts == 0, 0.0, sums / counts)

    n_blocks = row_blocks(m, 16 * len(cols) * n)
    shared, cross, sumsq, counts = masked_cross_moments(
        x, np.ascontiguousarray(column_means, dtype=np.float64), cols, n_blocks
    )
    return _combine(shared, cross, sumsq, counts, cols, m)


def _can_fuse(x: np.ndarray, cols: np.ndarray) -> bool:
    return x.dtype in (np.float32, np.float64) and len(cols) <= _MAX_FUSED_TARGETS


def _combine(
    shared: np.ndarray,
    cross: np.ndarray,
    sumsq: np.ndarray,
    counts: np.ndarray,
    cols: np.ndarray,
    m: int,
) -> np.ndarray:
    """Assemble `in_common * |corr|` from the accumulated masked moments."""
    # sum(z**2) / m is the variance of the mean pre-imputed column, so this
    # reproduces its correlation matrix. All-NaN columns get a NaN scale, which
    # propagates to a NaN score exactly like the pre-imputed formulation.
    std = np.sqrt(sumsq / m)
    std = np.where(counts == 0, np.nan, std)
    corr = (cross / m) / np.outer(std[cols], std)
    return (shared / m) * np.abs(corr)


def _scoring_chunked(x: np.ndarray, cols: np.ndarray, column_means: np.ndarray | None) -> np.ndarray:
    """Row-chunked fallback for dtypes or target counts the kernel does not cover."""
    m, n = x.shape
    work_dtype = x.dtype if x.dtype == np.float32 else np.float64

    if column_means is None:
        counts = np.zeros(n, dtype=np.int64)
        sums = np.zeros(n, dtype=np.float64)
        for start in range(0, m, _CHUNK_ROWS):
            chunk = x[start : start + _CHUNK_ROWS]
            observed = ~np.isnan(chunk)
            counts += np.count_nonzero(observed, axis=0)
            sums += np.where(observed, chunk, 0).sum(axis=0, dtype=np.float64)
        column_means = np.where(counts == 0, 0.0, sums / counts)
    else:
        counts = np.zeros(n, dtype=np.int64)
        for start in range(0, m, _CHUNK_ROWS):
            counts += np.count_nonzero(~np.isnan(x[start : start + _CHUNK_ROWS]), axis=0)

    means = np.asarray(column_means, dtype=work_dtype)
    shared = np.zeros((len(cols), n), dtype=np.float64)
    cross = np.zeros((len(cols), n), dtype=np.float64)
    sumsq = np.zeros(n, dtype=np.float64)
    for start in range(0, m, _CHUNK_ROWS):
        chunk = x[start : start + _CHUNK_ROWS]
        observed = (~np.isnan(chunk)).astype(work_dtype)
        centered = np.where(observed != 0, chunk - means, 0).astype(work_dtype, copy=False)
        shared += observed[:, cols].T @ observed
        cross += centered[:, cols].T @ centered
        sumsq += np.einsum("ij,ij->j", centered, centered)
    return _combine(shared, cross, sumsq, counts, cols, m)
