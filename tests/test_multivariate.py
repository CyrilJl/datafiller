import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

import datafiller.multivariate.imputer as multivariate_imputer_module
from datafiller.datasets import load_titanic
from datafiller.estimators.ridge import FastRidge
from datafiller.exceptions import DataFillerValueError
from datafiller.multivariate import MultivariateImputer
from datafiller.multivariate._numba_utils import (
    complete_rows_excluding,
    nan_cols_csc,
    nan_positions,
)
from datafiller.multivariate._scoring import scoring


def _reference_complete_rows(mask_nan: np.ndarray, usable_cols: np.ndarray) -> np.ndarray:
    """Rows without NaNs in the usable columns (oracle for complete_rows_excluding)."""
    return np.flatnonzero(~mask_nan[:, usable_cols].any(axis=1)).astype(np.uint32)


def _reference_preimpute(x: np.ndarray) -> np.ndarray:
    """Fill NaNs with column means (oracle for the fused scoring computation)."""
    xp = x.copy()
    col_means = np.nanmean(x, axis=0)
    nan_mask = np.isnan(x)
    xp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return xp


@pytest.fixture
def nan_array():
    rng = np.random.default_rng(0)
    n_samples = 500
    n_features = 10
    mean = np.linspace(0.0, 1.0, n_features)
    cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i - j), (n_features, n_features))
    x = rng.multivariate_normal(mean, cov, size=n_samples)
    n_nans = int(x.size * 0.10)
    nan_indices = rng.choice(x.size, size=n_nans, replace=False)
    x.flat[nan_indices] = np.nan
    return x


@pytest.fixture
def titanic_mixed_df():
    df = load_titanic()

    cols = ["sex", "age", "fare", "embarked", "deck"]
    df = df[cols].copy()
    df["sex"] = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df["deck"] = df["deck"].astype("category")
    return df


def test_multivariate_imputer_less_nans(nan_array):
    imputer = MultivariateImputer()
    imputed_array = imputer(nan_array)
    assert np.isnan(imputed_array).sum() < np.isnan(nan_array).sum()


def test_multivariate_imputer_handles_mar_missingness():
    rng = np.random.default_rng(42)
    n_samples = 300
    observed_driver = rng.normal(size=n_samples)
    support_feature = rng.normal(size=n_samples)
    target = 2.0 * observed_driver - 0.5 * support_feature + rng.normal(scale=0.05, size=n_samples)
    x_complete = np.column_stack([target, observed_driver, support_feature]).astype(np.float32)
    x = x_complete.copy()

    missing_mask = observed_driver > np.quantile(observed_driver, 0.65)
    x[missing_mask, 0] = np.nan

    imputed = MultivariateImputer(rng=0)(x, cols_to_impute=[0], n_nearest_features=2)

    assert not np.isnan(imputed[:, 0]).any()
    assert np.mean(np.abs(imputed[missing_mask, 0] - x_complete[missing_mask, 0])) < 0.15


def test_multivariate_imputer_reuses_training_subset_for_multiple_prediction_patterns():
    class CountingRegressor:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, X, y):
            self.fit_calls += 1
            self.value_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.value_, dtype=np.float32)

    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, np.nan],
            [2.0, 2.0, np.nan],
            [3.0, 3.0, np.nan],
            [4.0, 4.0, np.nan],
            [np.nan, 5.0, np.nan],
            [np.nan, 6.0, 6.0],
        ],
        dtype=np.float32,
    )
    regressor = CountingRegressor()

    imputed = MultivariateImputer(regressor=regressor, min_samples_train=3)(x, cols_to_impute=[0])

    assert regressor.fit_calls == 1
    assert not np.isnan(imputed[5:, 0]).any()


def test_multivariate_imputer_dataframe_support(nan_array):
    df = pd.DataFrame(nan_array, columns=[f"col_{i}" for i in range(nan_array.shape[1])])
    imputer = MultivariateImputer()
    imputed_df = imputer(df)
    assert isinstance(imputed_df, pd.DataFrame)
    assert np.isnan(imputed_df.values).sum() < np.isnan(df.values).sum()


def test_multivariate_imputer_categorical_dataframe_support(titanic_mixed_df):
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(titanic_mixed_df)
    assert list(imputed_df.columns) == list(titanic_mixed_df.columns)
    assert imputed_df["embarked"].isna().sum() < titanic_mixed_df["embarked"].isna().sum()
    assert imputed_df["deck"].isna().sum() < titanic_mixed_df["deck"].isna().sum()
    assert set(imputed_df["sex"].dropna().unique()).issubset({"male", "female"})
    assert set(imputed_df["embarked"].dropna().unique()).issubset({"C", "Q", "S"})
    assert set(imputed_df["deck"].dropna().unique()).issubset({"A", "B", "C", "D", "E", "F", "G", "T"})


def test_multivariate_imputer_cols_to_impute(nan_array):
    imputer = MultivariateImputer()
    imputed_array = imputer(nan_array, cols_to_impute=[1, 3])
    assert np.isnan(imputed_array[:, 0]).sum() == np.isnan(nan_array[:, 0]).sum()
    assert np.isnan(imputed_array[:, 1]).sum() == 0
    assert np.isnan(imputed_array[:, 2]).sum() == np.isnan(nan_array[:, 2]).sum()
    assert np.isnan(imputed_array[:, 3]).sum() == 0


def test_multivariate_imputer_rows_to_impute(nan_array):
    imputer = MultivariateImputer()
    imputed_array = imputer(nan_array, rows_to_impute=[1, 3])
    assert np.isnan(imputed_array[0, :]).sum() == np.isnan(nan_array[0, :]).sum()
    assert np.isnan(imputed_array[1, :]).sum() == 0
    assert np.isnan(imputed_array[2, :]).sum() == np.isnan(nan_array[2, :]).sum()
    assert np.isnan(imputed_array[3, :]).sum() == 0


def test_multivariate_imputer_reproducible_numeric(nan_array):
    imputer1 = MultivariateImputer(rng=0)
    imputer2 = MultivariateImputer(rng=0)
    imputed1 = imputer1(nan_array, n_nearest_features=3)
    imputed2 = imputer2(nan_array, n_nearest_features=3)
    np.testing.assert_allclose(imputed1, imputed2, equal_nan=True)


def test_multivariate_imputer_reproducible_mixed_types(titanic_mixed_df):
    imputer1 = MultivariateImputer(rng=0)
    imputer2 = MultivariateImputer(rng=0)
    imputed1 = imputer1(titanic_mixed_df, n_nearest_features=3)
    imputed2 = imputer2(titanic_mixed_df, n_nearest_features=3)
    pd.testing.assert_frame_equal(imputed1, imputed2)


def test_multivariate_imputer_min_samples_train(nan_array):
    imputer = MultivariateImputer(min_samples_train=nan_array.shape[0] + 1, fallback=None)
    imputed_array = imputer(nan_array)
    # With an unreachable min_samples_train and no fallback, no imputation happens
    assert np.isnan(imputed_array).sum() == np.isnan(nan_array).sum()


def test_multivariate_imputer_default_min_samples_train_is_calibrated():
    assert MultivariateImputer().min_samples_train == 20


def test_multivariate_imputer_set_params_resolves_default_min_samples_train():
    imputer = MultivariateImputer(min_samples_train=1)
    imputer.set_params(min_samples_train=None)
    assert imputer.min_samples_train == 20


def test_multivariate_imputer_set_params_resets_default_regressor_and_classifier():
    imputer = MultivariateImputer()

    custom_regressor = Ridge()
    imputer.set_params(regressor=custom_regressor)
    assert imputer.regressor is custom_regressor
    imputer.set_params(regressor=None)
    assert isinstance(imputer.regressor, FastRidge)

    custom_classifier = DecisionTreeClassifier(max_depth=2)
    imputer.set_params(classifier=custom_classifier)
    assert imputer.classifier is custom_classifier
    imputer.set_params(classifier=None)
    assert isinstance(imputer.classifier, DecisionTreeClassifier)
    assert imputer.classifier.max_depth == 4


def test_multivariate_imputer_set_params_rng_refreshes_default_classifier():
    imputer = MultivariateImputer()
    imputer.set_params(rng=7)
    assert imputer.classifier.random_state == 7

    custom_classifier = DecisionTreeClassifier(max_depth=2, random_state=0)
    imputer.set_params(classifier=custom_classifier, rng=3)
    assert imputer.classifier is custom_classifier
    assert imputer.classifier.random_state == 0


def test_multivariate_imputer_fallback_fills_with_column_mean(nan_array):
    # Unreachable threshold: every imputed cell must come from the fallback
    imputer = MultivariateImputer(min_samples_train=nan_array.shape[0] + 1)
    imputed_array = imputer(nan_array)
    assert not np.isnan(imputed_array).any()
    col_means = np.nanmean(nan_array, axis=0)
    iy, ix = np.nonzero(np.isnan(nan_array))
    np.testing.assert_allclose(imputed_array[iy, ix], col_means[ix], rtol=1e-5)


def test_multivariate_imputer_fallback_respects_rows_and_cols_to_impute(nan_array):
    imputer = MultivariateImputer(min_samples_train=nan_array.shape[0] + 1)
    imputed_array = imputer(nan_array, cols_to_impute=[1, 3])
    assert np.isnan(imputed_array[:, [1, 3]]).sum() == 0
    assert np.isnan(imputed_array[:, 0]).sum() == np.isnan(nan_array[:, 0]).sum()


def test_multivariate_imputer_fallback_mode_for_categoricals():
    n = 12
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "a", "a", "b", None, "a", "a", None, "b", "a", "a", "a"]),
            "value": np.arange(n, dtype=float),
        }
    )
    df.loc[3, "value"] = np.nan
    imputer = MultivariateImputer(min_samples_train=n + 1, rng=0)
    imputed_df = imputer(df)
    assert imputed_df["cat"].isna().sum() == 0
    # mode of observed categories is "a"
    assert (imputed_df.loc[[4, 7], "cat"] == "a").all()


def test_multivariate_imputer_invalid_fallback_raises():
    with pytest.raises(ValueError, match="fallback"):
        MultivariateImputer(fallback="median")

    imputer = MultivariateImputer()
    with pytest.raises(ValueError, match="fallback"):
        imputer.set_params(fallback="median")
    assert imputer.fallback == "simple"


def test_multivariate_imputer_boolean_support():
    df = pd.DataFrame(
        {
            "flag": pd.Series([True, False, None, True, None], dtype="boolean"),
            "value": [1.0, 2.0, 3.0, np.nan, 5.0],
        }
    )
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(df)
    assert imputed_df["flag"].isna().sum() < df["flag"].isna().sum()
    assert imputed_df["flag"].dtype == "boolean"
    assert set(imputed_df.columns) == {"flag", "value"}


def test_multivariate_imputer_preserves_numeric_dtypes():
    df = pd.DataFrame(
        {
            "count": pd.Series([1, 2, None, 4, 5], dtype="Int64"),
            "value": pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float64"),
        }
    )
    imputer = MultivariateImputer(rng=0)
    imputed_df = imputer(df)
    assert imputed_df["count"].dtype == df["count"].dtype
    assert imputed_df["value"].dtype == df["value"].dtype


@pytest.mark.parametrize("use_df", [False, True])
def test_multivariate_imputer_n_nearest_features_tracking(nan_array, use_df):
    if use_df:
        x = pd.DataFrame(nan_array, columns=[f"col_{i}" for i in range(nan_array.shape[1])])
        cols_with_nans = x.columns[x.isnull().any()].tolist()
    else:
        x = nan_array
        cols_with_nans = np.where(np.isnan(x).any(axis=0))[0]

    imputer = MultivariateImputer(rng=0)
    n_nearest_features = 2
    imputer(x, n_nearest_features=n_nearest_features)

    assert imputer.imputation_features_ is not None
    assert set(imputer.imputation_features_.keys()) == set(cols_with_nans)

    for col, features in imputer.imputation_features_.items():
        if use_df:
            assert isinstance(features, list)
            assert all(isinstance(f, str) for f in features)
        else:
            assert isinstance(features, np.ndarray)
        assert len(features) <= n_nearest_features
        assert col not in features


def test_multivariate_imputer_selects_top_scoring_nearest_features():
    scores = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.9, np.nan, 0.8, 0.2],
        ]
    )
    imputer = MultivariateImputer(rng=0)

    selected = imputer._get_sampled_cols(
        n_features=5,
        col_to_impute=2,
        n_nearest_features=2,
        scores=scores,
        scores_index=2,
    )

    np.testing.assert_array_equal(selected, np.array([1, 3]))


def test_complete_rows_excluding_returns_rows_without_nans():
    mask_nan = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ],
        dtype=bool,
    )
    iy, ix = np.nonzero(mask_nan)
    col_ptr, col_rows = nan_cols_csc(iy.astype(np.uint32), ix.astype(np.uint32), mask_nan.shape[1])
    row_nan_count = mask_nan.sum(axis=1).astype(np.uint32)
    hits = np.zeros(len(mask_nan), dtype=np.uint32)
    stamp = np.full(len(mask_nan), -1, dtype=np.int64)

    # Usable columns {1, 2}, i.e. column 0 is excluded.
    rows = complete_rows_excluding(row_nan_count, col_ptr, col_rows, np.array([0], dtype=np.uint32), hits, stamp, 1)

    np.testing.assert_array_equal(rows, np.array([0, 3], dtype=np.uint32))


def test_complete_rows_excluding_matches_reference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 12)).astype(np.float32)
    x[rng.random(x.shape) < 0.15] = np.nan

    mask_nan, iy, ix = nan_positions(x)
    row_nan_count = mask_nan.sum(axis=1).astype(np.uint32)
    col_ptr, col_rows = nan_cols_csc(iy, ix, x.shape[1])
    hits = np.zeros(len(x), dtype=np.uint32)
    stamp = np.full(len(x), -1, dtype=np.int64)
    all_cols = np.arange(x.shape[1], dtype=np.uint32)

    for epoch in range(1, 30):
        usable_mask = rng.random(x.shape[1]) < 0.7
        usable = all_cols[usable_mask]
        excluded = all_cols[~usable_mask]
        expected = _reference_complete_rows(mask_nan, usable)
        actual = complete_rows_excluding(row_nan_count, col_ptr, col_rows, excluded, hits, stamp, epoch)
        np.testing.assert_array_equal(actual, expected)


def test_gram_fast_path_matches_materialized_ridge():
    class MaterializedRidge(FastRidge):
        """Same math as FastRidge but bypasses the Gram-based fast path."""

    rng = np.random.default_rng(11)
    latent = rng.normal(size=(500, 4)).astype(np.float32)
    x_true = latent @ rng.normal(size=(4, 10)).astype(np.float32)
    x = x_true.copy()
    x[rng.random(x.shape) < 0.1] = np.nan

    imputed_fast = MultivariateImputer(rng=0)(x)
    imputed_reference = MultivariateImputer(regressor=MaterializedRidge(), rng=0)(x)

    np.testing.assert_allclose(imputed_fast, imputed_reference, rtol=1e-3, atol=1e-4)


def test_scoring_matches_preimpute_reference():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 8)).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.nan
    cols_to_impute = np.array([0, 3, 6])

    with np.errstate(all="ignore"):
        xp = _reference_preimpute(x)
        xp_standard = (xp - np.mean(xp, axis=0)) / np.std(xp, axis=0)
        corr = np.dot(xp_standard[:, cols_to_impute].T, xp_standard) / len(x)
        isfinite = np.isfinite(x).astype(np.float32)
        in_common = np.dot(isfinite[:, cols_to_impute].T, isfinite) / len(x)
        expected = in_common * np.abs(corr)

    actual = scoring(x, cols_to_impute)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def test_multivariate_imputer_uses_complete_case_fast_path(monkeypatch):
    x = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [np.nan, 40.0, 400.0],
        ]
    )

    def fail_optimask(**kwargs):
        raise AssertionError("optimask should not run when complete-case rows are sufficient")

    monkeypatch.setattr(multivariate_imputer_module, "optimask", fail_optimask)

    imputed = MultivariateImputer(min_samples_train=2, rng=0)(x)

    assert not np.isnan(imputed[3, 0])


def test_multivariate_imputer_falls_back_to_optimask_when_complete_cases_are_insufficient(monkeypatch):
    x = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, np.nan],
            [3.0, 3.0, np.nan],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, np.nan],
            [np.nan, 6.0, 6.0],
        ]
    )
    optimask_calls = 0
    original_optimask = multivariate_imputer_module.optimask

    def counting_optimask(**kwargs):
        nonlocal optimask_calls
        optimask_calls += 1
        return original_optimask(**kwargs)

    monkeypatch.setattr(multivariate_imputer_module, "optimask", counting_optimask)

    imputed = MultivariateImputer(min_samples_train=3, rng=0)(x)

    assert optimask_calls >= 1
    assert not np.isnan(imputed[5, 0])


def _float64_reference_imputation(x: np.ndarray, **kwargs) -> np.ndarray:
    """Impute through a float64 materialized ridge, bypassing the Gram fast path."""

    class Float64Ridge(FastRidge):
        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            n = X.shape[0]
            if self.fit_intercept:
                x_mean, y_mean = X.mean(axis=0), y.mean()
                A = X.T @ X - n * np.outer(x_mean, x_mean)
                b = X.T @ y - n * x_mean * y_mean
            else:
                x_mean, y_mean = None, 0.0
                A, b = X.T @ X, X.T @ y
            A.flat[:: A.shape[0] + 1] += self.alpha
            self.coef_ = np.linalg.solve(A, b)
            self.intercept_ = (y_mean - x_mean @ self.coef_) if self.fit_intercept else 0.0
            return self

        def predict(self, X):
            return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    return MultivariateImputer(regressor=Float64Ridge(), rng=0)(x.astype(np.float64), **kwargs)


@pytest.mark.parametrize("normalize", [True, False])
def test_gram_cache_matches_high_precision_reference(normalize):
    """The cached per-pattern Grams must reproduce an exact float64 solve.

    Uses an off-centre, wide-scaled matrix: that is where accumulating the Gram
    of `[X, y, 1]` is least forgiving, because the intercept correction cancels
    most of its magnitude.
    """
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(800, 5)).astype(np.float32)
    x = (latent @ rng.normal(size=(5, 14)).astype(np.float32)) * 5.0 + 120.0
    x[rng.random(x.shape) < 0.15] = np.nan

    actual = MultivariateImputer(rng=0)(x, normalize=normalize)
    expected = _float64_reference_imputation(x, normalize=normalize)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    spread = np.nanstd(expected, axis=0)
    np.testing.assert_array_less(np.nan_to_num(np.abs(actual - expected) / spread), 1e-4)


def test_gram_path_handles_patterns_beyond_the_cache_budget(monkeypatch):
    """Rows whose NaN pattern is not cached are accumulated directly, with the same result."""
    import datafiller.multivariate._gram as gram_module

    rng = np.random.default_rng(9)
    latent = rng.normal(size=(600, 4)).astype(np.float32)
    x = latent @ rng.normal(size=(4, 12)).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.nan

    reference = MultivariateImputer(rng=0)(x)
    # A budget of one byte leaves room for no cached group at all, so every row
    # with NaNs takes the on-demand path.
    monkeypatch.setattr(gram_module, "_GRAM_CACHE_BUDGET_BYTES", 1)
    uncached = MultivariateImputer(rng=0)(x)

    np.testing.assert_allclose(uncached, reference, rtol=1e-5, atol=1e-6)


def test_normalization_leaves_observed_values_untouched():
    """Standardizing must not round-trip the cells that were already observed."""
    rng = np.random.default_rng(2)
    x = (rng.normal(size=(300, 6)) * 1000.0 + 50_000.0).astype(np.float32)
    x[rng.random(x.shape) < 0.1] = np.nan
    observed = ~np.isnan(x)

    imputed = MultivariateImputer(rng=0)(x)

    np.testing.assert_array_equal(imputed[observed], x[observed])


def test_scoring_column_means_shortcut_matches_recomputation():
    """Passing known column means must not change the scores."""
    rng = np.random.default_rng(6)
    x = (rng.normal(size=(400, 9)) * 3 + 7).astype(np.float32)
    x[rng.random(x.shape) < 0.2] = np.nan
    cols = np.array([1, 4, 8])

    with np.errstate(all="ignore"):
        means = np.nanmean(x, axis=0)
    np.testing.assert_allclose(scoring(x, cols, column_means=means), scoring(x, cols), rtol=1e-5, atol=1e-8)


def test_scoring_chunked_fallback_matches_fused_kernel(monkeypatch):
    """The row-chunked path used for many targets agrees with the fused kernel."""
    import datafiller.multivariate._scoring as scoring_module

    rng = np.random.default_rng(8)
    x = (rng.normal(size=(350, 11)) * 2 - 4).astype(np.float32)
    x[rng.random(x.shape) < 0.18] = np.nan
    cols = np.arange(11)

    fused = scoring(x, cols)
    monkeypatch.setattr(scoring_module, "_MAX_FUSED_TARGETS", 0)
    monkeypatch.setattr(scoring_module, "_CHUNK_ROWS", 64)
    chunked = scoring(x, cols)

    np.testing.assert_array_equal(np.isnan(fused), np.isnan(chunked))
    np.testing.assert_allclose(np.nan_to_num(fused), np.nan_to_num(chunked), rtol=1e-4, atol=1e-6)


def test_column_nan_stats_matches_numpy():
    rng = np.random.default_rng(12)
    x = (rng.normal(size=(200, 7)) * 4 + 3).astype(np.float32)
    x[rng.random(x.shape) < 0.25] = np.nan

    mask_nan, counts, sums, has_inf = MultivariateImputer._column_nan_stats(x)

    np.testing.assert_array_equal(mask_nan, np.isnan(x))
    np.testing.assert_array_equal(counts, np.count_nonzero(~np.isnan(x), axis=0))
    np.testing.assert_allclose(sums, np.nansum(x, axis=0), rtol=1e-6)
    assert not has_inf


def test_column_nan_stats_detects_infinity_and_call_rejects_it():
    x = np.array([[1.0, 2.0], [3.0, np.inf]])
    assert MultivariateImputer._column_nan_stats(x)[3]
    with pytest.raises(DataFillerValueError, match="infinity"):
        MultivariateImputer()(x)


def test_standardization_leaves_unselected_columns_alone():
    rng = np.random.default_rng(13)
    x = (rng.normal(size=(150, 5)) * 6 + 20).astype(np.float32)
    x[rng.random(x.shape) < 0.1] = np.nan
    _, counts, sums, _ = MultivariateImputer._column_nan_stats(x)

    means, scales = MultivariateImputer._standardization(x, counts, sums, np.array([0, 3]))

    np.testing.assert_array_equal(means[[1, 2, 4]], 0.0)
    np.testing.assert_array_equal(scales[[1, 2, 4]], 1.0)
    with np.errstate(all="ignore"):
        np.testing.assert_allclose(means[[0, 3]], np.nanmean(x, axis=0)[[0, 3]], rtol=1e-6)
        np.testing.assert_allclose(scales[[0, 3]], np.nanstd(x, axis=0)[[0, 3]], rtol=1e-6)
