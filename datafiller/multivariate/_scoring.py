import numpy as np


@np.errstate(all="ignore")
def preimpute(x: np.ndarray) -> np.ndarray:
    """Performs a simple pre-imputation by filling NaNs with column means.

    Args:
        x: The array to pre-impute.

    Returns:
        The array with NaNs filled by column means.

    """
    xp = x.copy()
    col_means = np.nanmean(x, axis=0)
    nan_mask = np.isnan(x)
    xp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return xp


@np.errstate(all="ignore")
def scoring(x: np.ndarray, cols_to_impute: np.ndarray, mask_nan: np.ndarray | None = None) -> np.ndarray:
    """Calculates a score for each feature pair to guide feature selection.

    The score is based on the correlation and the proportion of shared
    non-NaN values. Mathematically this matches correlating the mean
    pre-imputed matrix, but it is computed from masked sums to avoid
    materializing several full-size temporaries.

    Args:
        x: The input data matrix.
        cols_to_impute: The columns that are candidates for imputation.
        mask_nan: Optional precomputed `np.isnan(x)` mask, to avoid a
            redundant scan when the caller already has it.

    Returns:
        A score matrix.
    """
    m = len(x)
    if mask_nan is None:
        mask_nan = np.isnan(x)

    work_dtype = x.dtype if x.dtype == np.float32 else np.float64
    valid = (~mask_nan).astype(work_dtype, copy=False)
    counts = valid.sum(axis=0)
    in_common = np.dot(valid[:, cols_to_impute].T, valid) / m

    z = np.where(mask_nan, 0, x).astype(work_dtype, copy=False)
    nan_means = z.sum(axis=0) / counts
    # Center observed entries in place; NaN slots must stay exactly 0, and
    # all-NaN columns propagate NaN just like the pre-imputed formulation.
    z -= nan_means
    z *= valid

    # After centering, sum(z**2) / m is the variance of the mean pre-imputed
    # column, so this reproduces `corr(preimpute(x))` without the copies.
    std = np.sqrt(np.einsum("ij,ij->j", z, z) / m)
    corr = np.dot(z[:, cols_to_impute].T, z) / m
    corr /= np.outer(std[cols_to_impute], std)

    return in_common * np.abs(corr)
