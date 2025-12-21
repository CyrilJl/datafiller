import numpy as np
import pandas as pd
from typing import Iterable, Union

from ..estimators.elm import _random_projection_relu
from .imputer import MultivariateImputer
from ..estimators.ridge import FastRidge
from ._utils import _dataframe_cols_to_impute_to_indices, _dataframe_rows_to_impute_to_indices


class ELMImputer(MultivariateImputer):
    """
    Imputes missing values using an Extreme Learning Machine-based approach.

    This imputer first generates a set of random projections of the input data
    and then uses a `MultivariateImputer` on the concatenated data (original
    features + projected features) to fill in the missing values. The imputation
    is only performed on the original features.

    Args:
        n_features (int): The number of features in the random projection.
        alpha (float): The regularization strength for the FastRidge regressor
            used for imputation.
        random_state (int): A seed for the random number generator for
            reproducibility.
        verbose (int): The verbosity level.
        min_samples_train (int): The minimum number of samples required to
            train a model.
        scoring (str or callable): The scoring function to use for feature
            selection.
    """

    def __init__(
        self,
        n_features: int = 100,
        alpha: float = 1.0,
        random_state: int = 0,
        verbose: int = 0,
        min_samples_train: int | None = None,
        scoring: Union[str, callable] = "default",
    ):
        super().__init__(
            estimator=FastRidge(alpha=alpha),
            verbose=verbose,
            min_samples_train=min_samples_train,
            rng=random_state,
            scoring=scoring,
        )
        self.n_features = n_features
        self.random_state = random_state
        self.projection_ = None
        self.bias_ = None

    def _initialize_projection(self, n_input_features: int):
        """Initializes the random projection matrix."""
        rng = np.random.RandomState(self.random_state)
        self.projection_ = rng.randn(n_input_features, self.n_features).astype(np.float32)
        self.bias_ = rng.randn(self.n_features).astype(np.float32)

    def __call__(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        rows_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        cols_to_impute: None | int | Iterable[int] | Iterable[str] = None,
        n_nearest_features: None | float | int = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        is_df = isinstance(x, pd.DataFrame)
        if is_df:
            original_columns = x.columns
            original_index = x.index
            rows_to_impute = _dataframe_rows_to_impute_to_indices(rows_to_impute, original_index)
            cols_to_impute = _dataframe_cols_to_impute_to_indices(cols_to_impute, original_columns)
            x = x.to_numpy(dtype=np.float32)
        elif cols_to_impute is None:
            cols_to_impute = np.arange(x.shape[1])

        if self.projection_ is None:
            self._initialize_projection(x.shape[1])

        x_projected = _random_projection_relu(np.nan_to_num(x), self.projection_, self.bias_)
        x_concatenated = np.concatenate([x, x_projected], axis=1)

        imputed_concatenated = super().__call__(
            x=x_concatenated,
            rows_to_impute=rows_to_impute,
            cols_to_impute=cols_to_impute,
            n_nearest_features=n_nearest_features,
        )

        imputed_original = imputed_concatenated[:, : x.shape[1]]

        if is_df:
            return pd.DataFrame(imputed_original, columns=original_columns)

        return imputed_original
