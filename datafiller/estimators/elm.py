import numpy as np
from numba import jit, prange

from .ridge import FastRidge


@jit(nopython=True, parallel=True)
def _random_projection_relu(X, W, bias):
    """
    Computes the random projection of X and applies the ReLU activation.
    """
    n_samples, n_features = X.shape
    n_components = W.shape[1]
    projected = np.empty((n_samples, n_components), dtype=np.float32)
    for i in prange(n_samples):
        for j in range(n_components):
            proj = bias[j]
            for k in range(n_features):
                proj += X[i, k] * W[k, j]
            projected[i, j] = proj
    return np.maximum(projected, 0)


class ExtremeLearningMachine:
    """
    An Extreme Learning Machine (ELM) estimator.
    This implementation uses a random projection, a ReLU activation, and a
    FastRidge regressor. It is designed for speed and assumes that the input
    data is well-behaved.
    Args:
        n_features (int): The number of features in the random projection.
        alpha (float): The regularization strength for the FastRidge regressor.
        random_state (int): A seed for the random number generator for
            reproducibility.
    """

    def __init__(
        self,
        n_features: int = 100,
        alpha: float = 1.0,
        random_state: int = 0,
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.random_state = random_state
        self.projection_ = None
        self.bias_ = None
        self.ridge_ = FastRidge(alpha=self.alpha)

    def _initialize_projection(self, n_input_features: int):
        """Initializes the random projection matrix."""
        rng = np.random.RandomState(self.random_state)
        self.projection_ = rng.randn(n_input_features, self.n_features).astype(np.float32)
        self.bias_ = rng.randn(self.n_features).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtremeLearningMachine":
        """
        Fits the ELM model.
        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The target values.
        Returns:
            self: The fitted estimator.
        """
        n_samples, n_input_features = X.shape
        if self.projection_ is None:
            self._initialize_projection(n_input_features)

        X_projected = _random_projection_relu(X, self.projection_, self.bias_)
        self.ridge_.fit(X_projected, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted model.
        Args:
            X (np.ndarray): The data to predict on.
        Returns:
            np.ndarray: The predicted values.
        """
        X_projected = _random_projection_relu(X, self.projection_, self.bias_)
        return self.ridge_.predict(X_projected)
