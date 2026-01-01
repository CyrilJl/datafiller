import numpy as np
from numba import njit, prange

from .ridge import FastRidge


@njit(parallel=True, cache=True)
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
            if proj < 0:
                proj = 0
            projected[i, j] = proj
    return projected


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
        normalize (bool): If True, standardize features before projection.
    """

    def __init__(
        self,
        n_features: int = 100,
        alpha: float = 1.0,
        random_state: int = 0,
        normalize: bool = True,
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.random_state = random_state
        self.normalize = normalize
        self.projection_ = None
        self.bias_ = None
        self.mean_ = None
        self.scale_ = None
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

        if self.normalize:
            self.mean_ = X.mean(axis=0).astype(np.float32)
            scale = X.std(axis=0).astype(np.float32)
            self.scale_ = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
            X_work = (X - self.mean_) / self.scale_
        else:
            self.mean_ = None
            self.scale_ = None
            X_work = X

        X_projected = _random_projection_relu(X_work, self.projection_, self.bias_)
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
        X_work = X
        if self.normalize:
            if self.mean_ is None or self.scale_ is None:
                raise ValueError("Model is not fitted or normalization statistics missing.")
            X_work = (X - self.mean_) / self.scale_
        X_projected = _random_projection_relu(X_work, self.projection_, self.bias_)
        return self.ridge_.predict(X_projected)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "n_features": self.n_features,
            "alpha": self.alpha,
            "random_state": self.random_state,
            "normalize": self.normalize,
        }

    def set_params(self, **params) -> "ExtremeLearningMachine":
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        # Re-initialize FastRidge if alpha is changed
        if "alpha" in params:
            self.ridge_ = FastRidge(alpha=self.alpha)
        return self
