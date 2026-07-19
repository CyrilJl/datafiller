import numpy as np
import pytest

from datafiller import MultivariateImputer

pl = pytest.importorskip("polars")


def test_multivariate_imputer_polars_mixed_schema_round_trip():
    df = pl.DataFrame(
        {
            "count": pl.Series([1, 2, None, 4, 5, 6], dtype=pl.Int64),
            "value": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
            "label": ["a", "a", None, "b", "a", "b"],
            "flag": pl.Series([True, False, None, True, False, True], dtype=pl.Boolean),
        }
    )

    imputed = MultivariateImputer(min_samples_train=10, rng=0)(df)

    assert isinstance(imputed, pl.DataFrame)
    assert imputed.columns == df.columns
    assert imputed.schema == df.schema
    assert imputed.null_count().to_numpy().sum() == 0
    assert imputed.select(pl.col("value").is_nan().sum()).item() == 0


def test_multivariate_imputer_polars_categorical_and_enum_support():
    df = pl.DataFrame(
        {
            "category": pl.Series(["a", "a", None, "b", "a"], dtype=pl.Categorical),
            "enum": pl.Series(["x", "y", None, "x", "x"], dtype=pl.Enum(["x", "y"])),
            "value": [1.0, 2.0, 3.0, 4.0, None],
        }
    )

    imputed = MultivariateImputer(min_samples_train=10, rng=0)(df)

    assert imputed.schema == df.schema
    assert imputed.null_count().to_numpy().sum() == 0
    assert set(imputed["category"].cast(pl.String).to_list()) <= {"a", "b"}
    assert set(imputed["enum"].cast(pl.String).to_list()) <= {"x", "y"}


def test_multivariate_imputer_polars_row_and_column_selectors():
    df = pl.DataFrame({"a": [1.0, None, 3.0, None], "b": [1.0, None, 3.0, None]})

    imputed = MultivariateImputer(min_samples_train=10)(df, rows_to_impute=[1], cols_to_impute=["a"])

    assert imputed["a"][1] is not None
    assert imputed["a"][3] is None
    assert imputed["b"].null_count() == df["b"].null_count()


def test_multivariate_imputer_polars_fallback_none_preserves_null_and_nan():
    df = pl.DataFrame({"value": [1.0, None, np.nan, 4.0], "feature": [1.0, 2.0, 3.0, 4.0]})

    imputed = MultivariateImputer(min_samples_train=10, fallback=None)(df)

    assert imputed["value"][1] is None
    assert np.isnan(imputed["value"][2])


def test_multivariate_imputer_polars_tracks_feature_names():
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0], "c": [4.0, 3.0, 2.0, 1.0]})
    imputer = MultivariateImputer(rng=0)

    imputer(df, n_nearest_features=1)

    assert set(imputer.imputation_features_) == {"a"}
    assert all(isinstance(feature, str) for feature in imputer.imputation_features_["a"])


def test_multivariate_imputer_polars_rejects_lazy_and_unsupported_dtypes():
    with pytest.raises(TypeError, match="collect"):
        MultivariateImputer()(pl.DataFrame({"a": [1.0]}).lazy())

    with pytest.raises(ValueError, match="Unsupported Polars dtype"):
        MultivariateImputer()(pl.DataFrame({"date": [None]}, schema={"date": pl.Date}))


def test_multivariate_imputer_polars_preserves_all_null_columns():
    df = pl.DataFrame({"empty": [None, None, None], "value": [1.0, None, 3.0]})

    imputed = MultivariateImputer(min_samples_train=4)(df)

    assert imputed.schema == df.schema
    assert imputed["empty"].null_count() == df.height
    assert imputed["value"].null_count() == 0


def test_multivariate_imputer_polars_matches_numpy_for_numeric_data():
    values = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=np.float32)
    expected = MultivariateImputer(min_samples_train=10)(values)

    actual = MultivariateImputer(min_samples_train=10)(pl.DataFrame(values, schema=["a", "b"]))

    np.testing.assert_allclose(actual.to_numpy(), expected)
