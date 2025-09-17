import numpy as np
import pytest
from sklearn.linear_model import Ridge

from datafiller.estimators.ridge import FastRidge
from datafiller.estimators.elm import ExtremeLearningMachine


@pytest.fixture
def data():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.array([1, 2, 3], dtype=np.float32)
    return X, y


def test_fast_ridge_fit_predict(data):
    X, y = data
    ridge = FastRidge(alpha=1.0)
    ridge.fit(X, y)
    preds = ridge.predict(X)
    assert preds.shape == (3,)

    # Compare with sklearn's Ridge
    sklearn_ridge = Ridge(alpha=1.0)
    sklearn_ridge.fit(X, y)
    sklearn_preds = sklearn_ridge.predict(X)
    np.testing.assert_allclose(preds, sklearn_preds, rtol=1e-4)


def test_fast_ridge_no_intercept(data):
    X, y = data
    ridge = FastRidge(alpha=1.0, fit_intercept=False)
    ridge.fit(X, y)
    preds = ridge.predict(X)
    assert ridge.intercept_ == 0.0

    # Compare with sklearn's Ridge
    sklearn_ridge = Ridge(alpha=1.0, fit_intercept=False)
    sklearn_ridge.fit(X, y)
    sklearn_preds = sklearn_ridge.predict(X)
    np.testing.assert_allclose(preds, sklearn_preds, rtol=1e-4)


def test_elm_fit_predict(data):
    X, y = data
    elm = ExtremeLearningMachine(n_features=10, random_state=0)
    elm.fit(X, y)
    preds = elm.predict(X)
    assert preds.shape == (3,)


def test_elm_reproducibility(data):
    X, y = data
    elm1 = ExtremeLearningMachine(n_features=10, random_state=0)
    elm1.fit(X, y)
    preds1 = elm1.predict(X)

    elm2 = ExtremeLearningMachine(n_features=10, random_state=0)
    elm2.fit(X, y)
    preds2 = elm2.predict(X)

    np.testing.assert_allclose(preds1, preds2, rtol=1e-4)


def test_elm_different_random_state(data):
    X, y = data
    elm1 = ExtremeLearningMachine(n_features=10, random_state=0)
    elm1.fit(X, y)
    preds1 = elm1.predict(X)

    elm2 = ExtremeLearningMachine(n_features=10, random_state=1)
    elm2.fit(X, y)
    preds2 = elm2.predict(X)

    assert not np.allclose(preds1, preds2, rtol=1e-4)
