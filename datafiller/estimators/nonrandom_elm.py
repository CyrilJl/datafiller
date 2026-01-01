import numpy as np


def _erf(x: np.ndarray) -> np.ndarray:
    """Approximate error function for numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    absx = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * absx)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    y = 1.0 - poly * np.exp(-absx * absx)
    return sign * y


def _erf_inv(x: np.ndarray) -> np.ndarray:
    """Approximate inverse error function for numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
    a = 0.147
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    approx = np.sign(x) * np.sqrt(np.sqrt(first * first - ln / a) - first)
    # Two Newton refinements for better accuracy.
    for _ in range(2):
        err = _erf(approx) - x
        deriv = 2.0 / np.sqrt(np.pi) * np.exp(-approx * approx)
        approx = approx - err / deriv
    return approx


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def _nngp_kernel_erf(X: np.ndarray) -> np.ndarray:
    """Compute the NNGP kernel matrix for an erf-activated shallow network."""
    n_features = X.shape[1]
    cov = (X @ X.T) / float(n_features)
    var = np.clip(np.diag(cov), 0.0, None)
    denom = np.sqrt((1.0 + 2.0 * var)[:, None] * (1.0 + 2.0 * var)[None, :])
    ratio = np.clip((2.0 * cov) / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    return (2.0 / np.pi) * np.arcsin(ratio)


class NonRandomExtremeLearningMachine:
    """
    Non-Random Extreme Learning Machine (ENR-ELM) for regression.

    This estimator implements the approximated and incremental ENR-ELM algorithms
    from 2411.16229v2. It uses an erf activation, a data-dependent hidden-layer
    weights matrix, and a selection mechanism for the number of neurons.

    Args:
        n_neurons (int): Maximum number of hidden neurons to select.
        method (str): "approx" for A-ENR-ELM or "incremental" for I-ENR-ELM.
        tol (float): Convergence tolerance for the incremental method.
        epsilon (float): Step size for the incremental method, in (0, 1].
        pinv_rcond (float): Cutoff for the pseudo-inverse in phase (1).
    """

    def __init__(
        self,
        n_neurons: int = 100,
        method: str = "approx",
        tol: float = 1e-4,
        epsilon: float = 1.0,
        pinv_rcond: float = 1e-12,
    ):
        self.n_neurons = n_neurons
        self.method = method
        self.tol = tol
        self.epsilon = epsilon
        self.pinv_rcond = pinv_rcond
        self.weights_ = None
        self.beta_ = None
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.s_mean_ = None
        self.selected_indices_ = None
        self._center_s = False

    def _fit_phase_one(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = _nngp_kernel_erf(X)
        _, U = np.linalg.eigh(K)
        X_cov = X.T @ X
        X_cov_inv = np.linalg.pinv(X_cov, rcond=self.pinv_rcond)
        W_hat = _erf_inv(U.T) @ X @ X_cov_inv
        return W_hat, U

    def _fit_incremental(self, S: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[int]]:
        n_samples, n_features = S.shape
        max_neurons = min(self.n_neurons, n_features)
        norms = np.sum(S * S, axis=0)
        beta = np.zeros(n_features, dtype=np.float64)
        yhat = np.zeros(n_samples, dtype=np.float64)
        r = y - yhat
        y_norm = np.linalg.norm(y)
        if y_norm == 0.0:
            return beta, []
        rold_norm = np.inf
        selected: list[int] = []
        while len(selected) < max_neurons:
            correlations = S.T @ r
            j = int(np.argmax(np.abs(correlations)))
            if norms[j] == 0.0:
                break
            step = self.epsilon * correlations[j] / norms[j]
            beta[j] += step
            yhat += step * S[:, j]
            r = y - yhat
            r_norm = np.linalg.norm(r)
            if j not in selected:
                selected.append(j)
            if (rold_norm - r_norm) / y_norm < self.tol:
                break
            rold_norm = r_norm
        return beta, selected

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NonRandomExtremeLearningMachine":
        """Fits the ENR-ELM model."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        Xs, self.x_mean_, self.x_std_ = _standardize(X)
        self.y_mean_ = y.mean()
        y_centered = y - self.y_mean_

        W_hat, U = self._fit_phase_one(Xs)
        max_neurons = min(self.n_neurons, U.shape[0])

        if self.method == "approx":
            scores = np.abs(U.T @ y_centered)
            order = np.argsort(-scores)
            selected = order[:max_neurons]
            self.beta_ = (U[:, selected].T @ y_centered).astype(np.float64)
            self.weights_ = W_hat[selected, :].astype(np.float64)
            self.s_mean_ = None
            self._center_s = False
            self.selected_indices_ = selected.tolist()
            return self

        if self.method != "incremental":
            raise ValueError("method must be 'approx' or 'incremental'")

        S = _erf(W_hat @ Xs.T).T
        self.s_mean_ = S.mean(axis=0)
        S_centered = S - self.s_mean_
        beta_full, selected = self._fit_incremental(S_centered, y_centered)
        self.beta_ = beta_full[selected].astype(np.float64)
        self.weights_ = W_hat[selected, :].astype(np.float64)
        self.s_mean_ = self.s_mean_[selected]
        self._center_s = True
        self.selected_indices_ = selected
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts outputs using the fitted ENR-ELM model."""
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.x_mean_) / self.x_std_
        S = _erf(self.weights_ @ Xs.T).T
        if self._center_s:
            S = S - self.s_mean_
        return self.y_mean_ + (S @ self.beta_)

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "n_neurons": self.n_neurons,
            "method": self.method,
            "tol": self.tol,
            "epsilon": self.epsilon,
            "pinv_rcond": self.pinv_rcond,
        }

    def set_params(self, **params) -> "NonRandomExtremeLearningMachine":
        """Set the parameters of this estimator."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
