"""Core implementation of the DataFiller imputer."""

from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from ._optimask import optimask


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
    if np.isnan(col_means).any():
        raise ValueError("One or more columns are all NaNs, which is not supported.")
    nan_mask = np.isnan(x)
    xp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return xp


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


class MultivariateImputer:
    """Imputes missing values in a 2D numpy array.

    This class uses a model-based approach to fill in missing values, where
    each feature with missing values is predicted using other features in the
    dataset. It is designed to be efficient, using Numba for critical parts
    and finding optimal data subsets for model training.

    Args:
        estimator (RegressorMixin, optional): A scikit-learn compatible
            regressor. Defaults to `LinearRegression()`.
        verbose (int, optional): The verbosity level. Defaults to 0.
        min_samples_train (int, optional): The minimum number of samples
            required to train a model. Defaults to 50.

    .. code-block:: python

        import numpy as np
        from datafiller import MultivariateImputer

        # Create a matrix with missing values
        X = np.array([
            [1, 2, 3],
            [4, np.nan, 6],
            [7, 8, 9]
        ])

        # Create an imputer and fill the missing values
        imputer = MultivariateImputer()
        X_imputed = imputer(X)

        print(X_imputed)
    """

    def __init__(
        self,
        estimator: RegressorMixin = Ridge(),
        verbose: int = 0,
        min_samples_train: int = 50,
        rng: Union[int, None] = None,
    ):
        self.estimator = estimator
        self.verbose = int(verbose)
        self.min_samples_train = min_samples_train
        self._rng = np.random.RandomState(rng)

    def _validate_input(
        self,
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

    def _get_sampled_cols(
        self,
        n_features: int,
        n_nearest_features: int | None,
        scores: np.ndarray | None,
        scores_index: int,
    ) -> np.ndarray:
        """Selects the feature columns to use for imputing a specific column.

        If `n_nearest_features` is specified, it selects a subset of features
        based on the provided scores. Otherwise, it returns all features.

        Args:
            n_features: The total number of features.
            n_nearest_features: The number of features to select.
            scores: A matrix of scores for feature selection.
            scores_index: The index of the column being imputed in the
                scores matrix.

        Returns:
            An array of column indices to use for imputation.
        """
        if n_nearest_features is not None:
            p = scores[scores_index] / scores[scores_index].sum()
            p[np.isnan(p)] = 0
            if p.sum() == 0:
                p = None
            sampled_cols = self._rng.choice(
                a=np.arange(n_features),
                size=n_nearest_features,
                replace=False,
                p=p,
            )
            return np.sort(sampled_cols)
        return np.arange(n_features)

    def _impute_col(
        self,
        x: np.ndarray,
        x_imputed: np.ndarray,
        col_to_impute: int,
        mask_nan: np.ndarray,
        mask_rows_to_impute: np.ndarray,
        iy: np.ndarray,
        ix: np.ndarray,
        n_nearest_features: int | None,
        scores: np.ndarray | None,
        scores_index: int,
    ) -> None:
        """Imputes all missing values in a single column.

        It identifies patterns of missingness, finds optimal data subsets for
        training, fits the estimator, and predicts the missing values.

        Args:
            x: The original data matrix.
            x_imputed: The matrix where imputed values are stored.
            col_to_impute: The index of the column to impute.
            mask_nan: A boolean mask of NaNs for the entire matrix.
            mask_rows_to_impute: A boolean mask of rows to be imputed.
            iy: Row indices of all NaNs.
            ix: Column indices of all NaNs.
            n_nearest_features: The number of features to use.
            scores: The feature selection scores.
            scores_index: The index of the column being imputed in the
                scores matrix.
        """
        m, n = x.shape

        sampled_cols = self._get_sampled_cols(n, n_nearest_features, scores, scores_index)

        imputable_rows = _imputable_rows(mask_nan=mask_nan, col=col_to_impute, mask_rows_to_impute=mask_rows_to_impute)
        if not len(imputable_rows):
            return

        trainable_rows = _trainable_rows(mask_nan=mask_nan, col=col_to_impute)
        if not len(trainable_rows):
            return  # Cannot impute if no training data is available for this column

        mask_trainable_rows = _index_to_mask(trainable_rows, m)
        mask_valid = ~mask_nan
        patterns, indexes = unique2d(mask_valid[imputable_rows][:, sampled_cols])

        pre_iy_trial, pre_ix_trial = nan_positions_subset(
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
                iy_trial, ix_trial = nan_positions_subset(
                    pre_iy_trial,
                    pre_ix_trial,
                    mask_trainable_rows,
                    mask_usable_cols,
                )
                rows, cols = optimask(
                    iy=iy_trial,
                    ix=ix_trial,
                    rows=trainable_rows,
                    cols=usable_cols,
                    global_matrix_size=(m, n),
                )
                if (len(rows) < self.min_samples_train) or (not len(cols)):
                    continue  # Not enough data to train a model

                X_train = _subset(X=x, rows=rows, columns=cols)
                y_train = _subset_one_column(X=x, rows=rows, col=col_to_impute)
                self.estimator.fit(X=X_train, y=y_train)
                x_imputed[index_predict, col_to_impute] = self.estimator.predict(
                    _subset(X=x, rows=index_predict, columns=cols)
                )

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        rows_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        cols_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        n_nearest_features: None | float | int = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Imputes missing values in the input data.

        The method can handle both NumPy arrays and pandas DataFrames.

        Args:
            x: The input data matrix with missing values (NaNs).
                Can be a numpy array or a pandas DataFrame.
            rows_to_impute: The rows to impute. The interpretation of this
                argument depends on the type of `x`.
                - If `x` is a NumPy array, this must be a list of integer indices.
                - If `x` is a pandas DataFrame, this must be a list of index labels.
                If None, all rows are considered for imputation. Defaults to None.
            cols_to_impute: The columns to impute. The interpretation of this
                argument depends on the type of `x`.
                - If `x` is a NumPy array, this must be a list of integer indices.
                - If `x` is a pandas DataFrame, this must be a list of column labels.
                If None, all columns are considered for imputation. Defaults to None.
            n_nearest_features: The number of features to use for
                imputation. If it's an int, it's the absolute number of
                features. If it's a float, it's the fraction of features to
                use. If None, all features are used. Defaults to None.

        Returns:
            The imputed data matrix. The return type will match the input type
            (NumPy array or pandas DataFrame).
        """
        is_df = isinstance(x, pd.DataFrame)
        if is_df:
            original_index = x.index
            original_columns = x.columns
            rows_to_impute = _dataframe_rows_to_impute_to_indices(rows_to_impute, original_index)
            cols_to_impute = _dataframe_cols_to_impute_to_indices(cols_to_impute, original_columns)
            x = x.to_numpy(dtype=np.float64)

        n_nearest_features = self._validate_input(x, rows_to_impute, cols_to_impute, n_nearest_features)

        m, n = x.shape
        rows_to_impute = _process_to_impute(size=m, to_impute=rows_to_impute)
        cols_to_impute = _process_to_impute(size=n, to_impute=cols_to_impute)
        mask_rows_to_impute = _mask_index_to_impute(size=m, to_impute=rows_to_impute)

        scores = scoring(x, cols_to_impute) if n_nearest_features is not None else None

        x_imputed = x.copy()
        mask_nan, iy, ix = nan_positions(x)

        for i, col in enumerate(tqdm(cols_to_impute, leave=False, disable=(not self.verbose))):
            self._impute_col(x, x_imputed, col, mask_nan, mask_rows_to_impute, iy, ix, n_nearest_features, scores, i)

        if is_df:
            return pd.DataFrame(x_imputed, index=original_index, columns=original_columns)

        return x_imputed
