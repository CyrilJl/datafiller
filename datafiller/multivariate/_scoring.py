import numpy as np


@np.errstate(all="ignore")
def preimpute(x: np.ndarray) -> np.ndarray:
    """Performs a simple pre-imputation by filling NaNs with column means.

    Args:
        x: The array to pre-impute.

    Returns:
        The array with NaNs filled by column means.

    Raises:
        ValueError: If any column is entirely NaN.
    """
    xp = x.copy()
    col_means = np.nanmean(x, axis=0)
    nan_mask = np.isnan(x)
    xp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return xp


@np.errstate(all="ignore")
def scoring(x: np.ndarray, cols_to_impute: np.ndarray) -> np.ndarray:
    """Calculates a score for each feature pair to guide feature selection.

    The score is based on the correlation and the proportion of shared
    non-NaN values.

    Args:
        x: The input data matrix.
        cols_to_impute: The columns that are candidates for imputation.

    Returns:
        A score matrix.
    """
    n = len(x)

    # Optimized isfinite calculation
    isfinite = np.isfinite(x).astype("float32", copy=False)

    # Optimized in_common calculation
    isfinite_cols = isfinite[:, cols_to_impute]
    in_common = np.dot(isfinite_cols.T, isfinite) / n

    # Pre-impute and standardize
    xp = preimpute(x)
    mx = np.mean(xp, axis=0)
    sx = np.std(xp, axis=0)
    xp_standard = (xp - mx) / sx

    # Optimized correlation calculation
    yp_standard = xp_standard[:, cols_to_impute]
    corr = np.dot(yp_standard.T, xp_standard) / n

    return in_common * np.abs(corr)
