import numpy as np


def fit_ridge_from_gram(
    gram: np.ndarray,
    n_samples: int,
    alpha: float,
    fit_intercept: bool,
) -> tuple[np.ndarray, float]:
    """Solve the same ridge problem as :meth:`FastRidge.fit` from a Gram matrix.

    Args:
        gram: The `(k + 2, k + 2)` Gram matrix of the augmented training
            matrix `[X, y, 1]`, i.e. `Z.T @ Z` with `Z = [X, y, ones]`.
        n_samples: The number of training samples the Gram was computed on.
        alpha: The regularization strength.
        fit_intercept: Whether to fit an intercept.

    Returns:
        A tuple `(coef, intercept)`.
    """
    k = gram.shape[0] - 2
    sxx = gram[:k, :k]
    sxy = gram[:k, k]
    if fit_intercept:
        sx = gram[:k, k + 1]
        sy = gram[k, k + 1]
        A = sxx - np.outer(sx, sx) / n_samples
        b = sxy - sx * (sy / n_samples)
    else:
        A = sxx.copy()
        b = sxy
    A.flat[:: k + 1] += alpha
    coef = np.linalg.solve(A, b)
    if fit_intercept:
        intercept = sy / n_samples - (sx / n_samples) @ coef
    else:
        intercept = 0.0
    return coef, intercept


class FastRidge:
    """
    A simplified Ridge regressor.

    This implementation is designed for speed and assumes that the input data
    is well-behaved (e.g., no NaNs, correct dtypes). It is not a full-featured
    scikit-learn estimator but provides the necessary `fit` and `predict`
    methods for use within the DataFiller.

    Args:
        alpha (float): The regularization strength. Defaults to 0.01.
        fit_intercept (bool): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations.
            Defaults to True.
    """

    def __init__(self, alpha: float = 1e-2, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastRidge":
        """
        Fits the Ridge regression model.

        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The target values.

        Returns:
            self: The fitted regressor.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples = X.shape[0]
        Xt = X.T

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            A = Xt @ X - np.float32(n_samples) * np.outer(X_mean, X_mean)
            b = Xt @ y - np.float32(n_samples) * X_mean * y_mean
        else:
            X_mean = None
            y_mean = np.float32(0.0)
            A = Xt @ X
            b = Xt @ y

        A.flat[:: A.shape[0] + 1] += self.alpha
        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = y_mean - (X_mean @ self.coef_)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted model.

        Args:
            X (np.ndarray): The data to predict on.

        Returns:
            np.ndarray: The predicted values.
        """
        X = np.asarray(X, dtype=np.float32)
        return X @ self.coef_ + self.intercept_
