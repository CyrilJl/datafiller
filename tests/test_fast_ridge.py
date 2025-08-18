import numpy as np
import pytest
from sklearn.linear_model import Ridge
from datafiller.multivariate._fast_ridge import FastRidge

@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fast_ridge_vs_sklearn(fit_intercept):
    """
    Compare FastRidge with sklearn's Ridge to ensure correctness.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 10
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    # Add a constant offset to y to test intercept
    if fit_intercept:
        y += 5

    alpha = 1.0

    # Fit FastRidge
    fast_ridge = FastRidge(alpha=alpha, fit_intercept=fit_intercept)
    fast_ridge.fit(X, y)
    fast_ridge_preds = fast_ridge.predict(X)

    # Fit sklearn's Ridge
    sklearn_ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver='cholesky')
    sklearn_ridge.fit(X, y)
    sklearn_preds = sklearn_ridge.predict(X)

    # Compare coefficients and intercept
    np.testing.assert_allclose(fast_ridge.coef_, sklearn_ridge.coef_, rtol=1e-5)
    if fit_intercept:
        np.testing.assert_allclose(fast_ridge.intercept_, sklearn_ridge.intercept_, rtol=1e-5)

    # Compare predictions
    np.testing.assert_allclose(fast_ridge_preds, sklearn_preds, rtol=1e-5)
