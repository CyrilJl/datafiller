import numpy as np
import pytest
from sklearn.linear_model import Ridge

from datafiller.estimators.elm import ExtremeLearningMachine
from datafiller.estimators.ridge import FastRidge


@pytest.fixture
def data():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.array([1, 2, 3], dtype=np.float32)
    return X, y


def test_fast_ridge_fit_predict(data):
    X, y = data
    ridge = FastRidge(alpha=1.0, fit_intercept=True)
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


def test_elm_refit_varying_feature_counts():
    # Inside the imputers, one ELM instance is refit on column subsets of
    # varying widths; each width must get its own (reproducible) projection.
    rng = np.random.default_rng(0)
    y = rng.random(50).astype(np.float32)
    X5 = rng.random((50, 5)).astype(np.float32)
    X8 = rng.random((50, 8)).astype(np.float32)

    elm = ExtremeLearningMachine(n_features=10, random_state=0)
    elm.fit(X5, y)
    preds5 = elm.predict(X5)
    elm.fit(X8, y)
    preds8 = elm.predict(X8)

    assert np.isfinite(preds5).all()
    assert np.isfinite(preds8).all()

    fresh = ExtremeLearningMachine(n_features=10, random_state=0)
    fresh.fit(X8, y)
    np.testing.assert_allclose(preds8, fresh.predict(X8), rtol=1e-5)


def test_elm_chunked_fit_matches_single_shot(monkeypatch):
    import datafiller.estimators.elm as elm_mod

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    y = rng.standard_normal(300).astype(np.float32)

    ref = ExtremeLearningMachine(n_features=16, random_state=0).fit(X, y).predict(X)
    monkeypatch.setattr(elm_mod, "_CHUNK_ROWS", 64)
    chunked = ExtremeLearningMachine(n_features=16, random_state=0).fit(X, y).predict(X)
    np.testing.assert_allclose(chunked, ref, rtol=1e-3, atol=1e-3)


def test_elm_caps_hidden_width_by_samples():
    # A pattern-specific fit inside the imputers can have far fewer rows than
    # `n_features`; the hidden width must shrink to keep the internal ridge
    # fit from becoming (hidden_units > n_samples) underdetermined.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = rng.standard_normal(20).astype(np.float32)

    elm = ExtremeLearningMachine(n_features=100, random_state=0, min_samples_per_feature=5)
    elm.fit(X, y)

    assert elm.n_features_used_ == 4  # 20 samples // 5 per feature
    assert elm.ridge_.coef_.shape == (4,)
    preds = elm.predict(X)
    assert preds.shape == (20,)
    assert np.isfinite(preds).all()


def test_elm_uses_full_width_with_ample_samples():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 4)).astype(np.float32)
    y = rng.standard_normal(1000).astype(np.float32)

    elm = ExtremeLearningMachine(n_features=50, random_state=0, min_samples_per_feature=5)
    elm.fit(X, y)

    assert elm.n_features_used_ == 50
    assert elm.ridge_.coef_.shape == (50,)


def test_elm_min_samples_per_feature_zero_disables_cap():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    y = rng.standard_normal(10).astype(np.float32)

    elm = ExtremeLearningMachine(n_features=50, random_state=0, min_samples_per_feature=0)
    elm.fit(X, y)

    assert elm.n_features_used_ == 50
    assert elm.ridge_.coef_.shape == (50,)


def test_elm_hidden_width_used_for_prediction_matches_fit():
    # predict() must slice the same capped width used at fit time, not the
    # full `n_features`, even though the cached projection is stored at full
    # width for reuse across differently-sized fits.
    rng = np.random.default_rng(0)
    X_small = rng.standard_normal((15, 3)).astype(np.float32)
    y_small = rng.standard_normal(15).astype(np.float32)
    X_large = rng.standard_normal((500, 3)).astype(np.float32)

    elm = ExtremeLearningMachine(n_features=40, random_state=0, min_samples_per_feature=5)
    elm.fit(X_small, y_small)
    assert elm.n_features_used_ == 3
    preds = elm.predict(X_large)
    assert preds.shape == (500,)
    assert np.isfinite(preds).all()


def test_elm_projection_is_fan_in_scaled():
    # Pre-activation variance should stay roughly constant across input
    # widths instead of growing with n_input_features (no fan-in scaling
    # would make the same `alpha` regularize inconsistently across the
    # different column subsets the imputers refit this estimator with).
    rng = np.random.default_rng(0)
    n_samples = 5000
    elm = ExtremeLearningMachine(n_features=200, random_state=0, min_samples_per_feature=0)

    stds = []
    for n_in in (5, 50):
        X = rng.standard_normal((n_samples, n_in)).astype(np.float32)
        W, bias = elm._projection(n_in)
        pre_activation = X @ W + bias
        stds.append(pre_activation.std())

    assert stds[1] / stds[0] < 1.5


def test_elm_different_random_state(data):
    X, y = data
    elm1 = ExtremeLearningMachine(n_features=10, random_state=0)
    elm1.fit(X, y)
    preds1 = elm1.predict(X)

    elm2 = ExtremeLearningMachine(n_features=10, random_state=1)
    elm2.fit(X, y)
    preds2 = elm2.predict(X)

    assert not np.allclose(preds1, preds2, rtol=1e-4)
