import numpy as np


class FastRidge:
    """A simplified Ridge regressor.

    This implementation is designed for speed and assumes that the input data
    is well-behaved (e.g., no NaNs, correct dtypes). It is not a full-featured
    scikit-learn estimator but provides the necessary `fit` and `predict`
    methods for use within the DataFiller.

    Args:
        alpha (float): The regularization strength. Defaults to 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastRidge":
        """Fits the Ridge regression model.

        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The target values.

        Returns:
            self: The fitted regressor.
        """
        A = X.T @ X
        A[np.diag_indices_from(A)] += self.alpha
        self.coef_ = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the fitted model.

        Args:
            X (np.ndarray): The data to predict on.

        Returns:
            np.ndarray: The predicted values.
        """
        return X @ self.coef_
