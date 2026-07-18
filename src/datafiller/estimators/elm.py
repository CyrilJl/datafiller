import numpy as np

from .ridge import FastRidge, fit_ridge_from_gram

# Above this many rows, fit/predict process the projected features in chunks
# so the full (n_samples, n_features) hidden matrix is never materialized.
_CHUNK_ROWS = 65536


class ExtremeLearningMachine:
    """
    An Extreme Learning Machine (ELM) estimator.

    This implementation uses a random projection, a ReLU activation, and a
    FastRidge regressor. It is designed for speed and assumes that the input
    data is well-behaved.

    The random projection is sampled lazily for each input width and cached,
    so a single instance can be refit on data with varying numbers of input
    features (as happens inside the imputers) while staying reproducible.
    Weights and bias are fan-in scaled (`1 / sqrt(n_input_features)`) so the
    pre-activation variance, and therefore the effective strength of `alpha`,
    stays consistent across the different input widths the imputers refit
    this estimator with.

    The hidden width is also capped by the number of training samples seen at
    fit time (see `min_samples_per_feature`): the imputers refit this
    estimator on a small, pattern-specific subset of rows for every distinct
    missingness pattern, and a hidden layer wider than that subset turns the
    internal ridge fit into a severely underdetermined (more parameters than
    samples) problem regardless of regularization.

    Args:
        n_features (int): The maximum number of features in the random
            projection.
        alpha (float): The regularization strength for the FastRidge regressor.
        random_state (int): A seed for the random number generator for
            reproducibility.
        min_samples_per_feature (int): The minimum number of training samples
            required per hidden feature. At fit time, the hidden width is
            capped to `n_samples // min_samples_per_feature` (at least 1) so
            it never exceeds what the available data can support. Set to 0
            to disable the cap and always use `n_features`. Defaults to 5.
    """

    def __init__(
        self,
        n_features: int = 100,
        alpha: float = 1.0,
        random_state: int = 0,
        min_samples_per_feature: int = 5,
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.random_state = random_state
        self.min_samples_per_feature = min_samples_per_feature
        self.projection_ = None
        self.bias_ = None
        self.ridge_ = FastRidge(alpha=self.alpha)
        self._projections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.n_features_used_ = None

    def _projection(self, n_input_features: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the cached (weights, bias) pair for this input width."""
        cached = self._projections.get(n_input_features)
        if cached is None:
            rng = np.random.RandomState(self.random_state)
            scale = np.float32(1.0 / np.sqrt(n_input_features))
            cached = (
                rng.randn(n_input_features, self.n_features).astype(np.float32) * scale,
                rng.randn(self.n_features).astype(np.float32) * scale,
            )
            self._projections[n_input_features] = cached
        self.projection_, self.bias_ = cached
        return cached

    def _effective_n_features(self, n_samples: int) -> int:
        """Caps the hidden width so it never exceeds what `n_samples` can support."""
        if self.min_samples_per_feature <= 0:
            return self.n_features
        return min(self.n_features, max(1, n_samples // self.min_samples_per_feature))

    @staticmethod
    def _project(X: np.ndarray, W: np.ndarray, bias: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        projected = np.matmul(X, W, out=out)
        projected += bias
        np.maximum(projected, 0.0, out=projected)
        return projected

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtremeLearningMachine":
        """
        Fits the ELM model.

        Args:
            X (np.ndarray): The training data.
            y (np.ndarray): The target values.

        Returns:
            self: The fitted estimator.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        W, bias = self._projection(X.shape[1])
        k = self._effective_n_features(n_samples)
        self.n_features_used_ = k
        if k < self.n_features:
            W = W[:, :k]
            bias = bias[:k]

        if n_samples <= _CHUNK_ROWS:
            self.ridge_.fit(self._project(X, W, bias), y)
            return self

        # Large fits: accumulate the augmented Gram matrix of [H, y, 1] chunk
        # by chunk and solve the same ridge problem from it, keeping peak
        # memory bounded by the chunk size instead of n_samples.
        y = np.asarray(y, dtype=np.float32)
        gram = np.zeros((k + 2, k + 2), dtype=np.float64)
        buffer = np.empty((_CHUNK_ROWS, k + 2), dtype=np.float32)
        buffer[:, k + 1] = 1.0
        for start in range(0, n_samples, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, n_samples)
            z = buffer[: stop - start]
            self._project(X[start:stop], W, bias, out=z[:, :k])
            z[:, k] = y[start:stop]
            gram += z.T @ z
        coef, intercept = fit_ridge_from_gram(
            gram=gram,
            n_samples=n_samples,
            alpha=self.ridge_.alpha,
            fit_intercept=self.ridge_.fit_intercept,
        )
        self.ridge_.coef_ = coef.astype(np.float32)
        self.ridge_.intercept_ = np.float32(intercept)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the fitted model.

        Args:
            X (np.ndarray): The data to predict on.

        Returns:
            np.ndarray: The predicted values.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        W, bias = self._projection(X.shape[1])
        k = self.n_features_used_ if self.n_features_used_ is not None else self.n_features
        if k < self.n_features:
            W = W[:, :k]
            bias = bias[:k]

        if n_samples <= _CHUNK_ROWS:
            return self.ridge_.predict(self._project(X, W, bias))

        predictions = np.empty(n_samples, dtype=np.float32)
        buffer = np.empty((_CHUNK_ROWS, k), dtype=np.float32)
        for start in range(0, n_samples, _CHUNK_ROWS):
            stop = min(start + _CHUNK_ROWS, n_samples)
            projected = self._project(X[start:stop], W, bias, out=buffer[: stop - start])
            predictions[start:stop] = self.ridge_.predict(projected)
        return predictions

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
            "min_samples_per_feature": self.min_samples_per_feature,
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
        # Cached projections depend on n_features and random_state
        if "n_features" in params or "random_state" in params:
            self._projections = {}
            self.projection_ = None
            self.bias_ = None
        return self
