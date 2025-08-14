from typing import Iterable

import numpy as np
from numba import njit, prange
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

from ._optimask import optimask


@njit(boundscheck=True, cache=True)
def _flatnonzero(x):
    n = len(x)
    ret = np.empty(n, dtype=np.uint32)
    cnt = 0
    for k in range(n):
        if x[k]:
            ret[cnt] = k
            cnt += 1
    return ret[:cnt]


@njit(boundscheck=True, cache=True)
def nan_positions(x):
    m, n = x.shape
    mask_nan = np.zeros((m, n), dtype=np.bool_)
    iy, ix = np.empty(m * n, dtype=np.uint32), np.empty(m * n, dtype=np.uint32)
    rows_with_nan, cols_with_nan = np.zeros(m, dtype=np.bool_), np.zeros(n, dtype=np.bool_)
    cnt = 0
    for i in range(m):
        for j in range(n):
            if np.isnan(x[i, j]):
                mask_nan[i, j] = True
                iy[cnt] = i
                ix[cnt] = j
                rows_with_nan[i] = True
                cols_with_nan[j] = True
                cnt += 1

    return (
        mask_nan,
        iy[:cnt],
        ix[:cnt],
        _flatnonzero(rows_with_nan),
        _flatnonzero(cols_with_nan),
    )


@njit(boundscheck=True, cache=True)
def nan_positions_subset(iy, ix, mask_subset_rows, mask_subset_cols):
    n_nan = len(ix)
    m = len(mask_subset_rows)
    n = len(mask_subset_cols)
    rows_with_nan, cols_with_nan = np.zeros(m, dtype=np.bool_), np.zeros(n, dtype=np.bool_)
    size = min(n_nan, mask_subset_rows.sum() * mask_subset_cols.sum())
    sub_iy, sub_ix = np.empty(size, np.uint32), np.empty(size, np.uint32)
    cnt = 0
    for k in range(n_nan):
        row, col = iy[k], ix[k]
        if mask_subset_cols[col] and mask_subset_rows[row]:
            rows_with_nan[row] = True
            cols_with_nan[col] = True
            sub_iy[cnt] = row
            sub_ix[cnt] = col
            cnt += 1

    return (
        sub_iy[:cnt],
        sub_ix[:cnt],
        _flatnonzero(rows_with_nan),
        _flatnonzero(cols_with_nan),
    )


@njit(parallel=True, boundscheck=True, cache=True)
def _subset(X, rows, columns):
    Xs = np.empty((len(rows), len(columns)), dtype=X.dtype)
    for i in prange(len(rows)):
        for j in range(len(columns)):
            Xs[i, j] = X[rows[i], columns[j]]
    return Xs


@njit(boundscheck=True, cache=True)
def _imputable_rows(mask_nan, col, mask_rows_to_impute):
    m = len(mask_nan)
    ret = np.empty(m, dtype=np.uint32)
    cnt = 0
    for k in range(m):
        if mask_nan[k, col] and mask_rows_to_impute[k]:
            ret[cnt] = k
            cnt += 1
    return ret[:cnt]


@njit(boundscheck=True, cache=True)
def _trainable_rows(mask_nan, col):
    m = len(mask_nan)
    ret = np.empty(m, dtype=np.uint32)
    cnt = 0
    for k in range(m):
        if not mask_nan[k, col]:
            ret[cnt] = k
            cnt += 1
    return ret[:cnt]


def _process_to_impute(size, to_impute):
    if to_impute is None:
        return np.arange(size)
    if isinstance(to_impute, int):
        return np.array([to_impute])
    else:
        return np.array(to_impute)


@njit(boundscheck=True, parallel=True)
def _mask_index_to_impute(size, to_impute):
    ret = np.zeros(size, dtype=np.bool_)
    set_to_impute = set(to_impute)
    for k in prange(size):
        if k in set_to_impute:
            ret[k] = True
    return ret


def unique2d(x):
    """
    equivalent to `np.unique(x, return_inverse=True, axis=0)`
    """
    x_struct = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx, inv = np.unique(x_struct, return_index=True, return_inverse=True)
    return x[idx], inv.ravel()


def preimpute(x):
    xp = x.copy()
    col_means = np.nanmean(x, axis=0)
    if np.isnan(col_means).any():
        raise ValueError("One or more columns are all NaNs, which is not supported.")
    nan_mask = np.isnan(x)
    xp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return xp


def scoring(x, cols_to_impute):
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


@njit(boundscheck=True, parallel=True)
def _index_to_mask(x, n):
    ret = np.zeros(n, dtype=np.bool_)
    for k in prange(len(x)):
        ret[x[k]] = True
    return ret


class DataFiller:
    def __init__(self, estimator=LinearRegression(), verbose=0):
        self.estimator = estimator
        self.verbose = int(verbose)

    def _validate_input(self, x, rows_to_impute, cols_to_impute, n_nearest_features):
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
            if not all(isinstance(i, int) for i in rows_to_impute) or not all(0 <= i < m for i in rows_to_impute):
                raise ValueError(f"rows_to_impute must be a list of integers between 0 and {m-1}.")

        if cols_to_impute is not None:
            if isinstance(cols_to_impute, int):
                cols_to_impute = [cols_to_impute]
            if not all(isinstance(i, int) for i in cols_to_impute) or not all(0 <= i < n for i in cols_to_impute):
                raise ValueError(f"cols_to_impute must be a list of integers between 0 and {n-1}.")

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

    def _get_sampled_cols(self, n_features, n_nearest_features, scores, i):
        """Selects the columns to use for imputation."""
        if n_nearest_features is not None:
            p = scores[i] / scores[i].sum()
            if np.isnan(p).any():
                p = None
            sampled_cols = np.random.default_rng().choice(
                a=np.arange(n_features),
                size=n_nearest_features,
                replace=False,
                p=p,
            )
            return np.sort(sampled_cols)
        return np.arange(n_features)

    def _impute_col(self, x, x_imputed, col_to_impute, mask_nan, mask_rows_to_impute, iy, ix, n_nearest_features, scores):
        """Imputes a single column."""
        m, n = x.shape
        i = col_to_impute

        sampled_cols = self._get_sampled_cols(n, n_nearest_features, scores, i)

        imputable_rows = _imputable_rows(mask_nan=mask_nan, col=i, mask_rows_to_impute=mask_rows_to_impute)
        if not len(imputable_rows):
            return

        trainable_rows = _trainable_rows(mask_nan=mask_nan, col=i)
        if not len(trainable_rows):
            return # Cannot impute if no training data is available for this column

        mask_trainable_rows = _index_to_mask(trainable_rows, m)
        mask_valid = ~mask_nan
        patterns, indexes = unique2d(mask_valid[imputable_rows][:, sampled_cols])

        pre_iy_trial, pre_ix_trial, _, _ = nan_positions_subset(
            iy,
            ix,
            mask_trainable_rows,
            _index_to_mask(sampled_cols, n),
        )

        for k in range(len(patterns)):
            index_predict = imputable_rows[indexes == k]
            usable_cols = sampled_cols[patterns[k]].astype(np.uint32)
            mask_usable_cols = _index_to_mask(usable_cols, n)
            if len(usable_cols):
                iy_trial, ix_trial, _, _ = nan_positions_subset(
                    pre_iy_trial,
                    pre_ix_trial,
                    mask_trainable_rows,
                    mask_usable_cols,
                )
                rows, cols = optimask(
                    iy=iy_trial, ix=ix_trial, rows=trainable_rows, cols=usable_cols, global_matrix_size=(m, n)
                )
                if not len(rows) or not len(cols):
                    continue # Not enough data to train a model

                X_train = _subset(X=x, rows=rows, columns=cols)
                self.estimator.fit(X=X_train, y=x[rows, i])
                x_imputed[index_predict, i] = self.estimator.predict(_subset(X=x, rows=index_predict, columns=cols))


    def __call__(
        self,
        x: np.ndarray,
        rows_to_impute: None | int | Iterable[int] = None,
        cols_to_impute: None | int | Iterable[int] = None,
        n_nearest_features: None | float | int = None,
    ):
        n_nearest_features = self._validate_input(x, rows_to_impute, cols_to_impute, n_nearest_features)

        m, n = x.shape
        rows_to_impute = _process_to_impute(size=m, to_impute=rows_to_impute)
        cols_to_impute = _process_to_impute(size=n, to_impute=cols_to_impute)
        mask_rows_to_impute = _mask_index_to_impute(size=m, to_impute=rows_to_impute)

        scores = scoring(x, cols_to_impute) if n_nearest_features is not None else None

        x_imputed = x.copy()
        mask_nan, iy, ix, _, _ = nan_positions(x)

        for i, col in enumerate(tqdm(cols_to_impute, leave=False, disable=(not self.verbose))):
            self._impute_col(x, x_imputed, col, mask_nan, mask_rows_to_impute, iy, ix, n_nearest_features, scores)

        return x_imputed
