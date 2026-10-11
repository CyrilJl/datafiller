import numpy as np
import pandas as pd
import pytest

from datafiller.timeseries import TimeSeriesImputer


@pytest.fixture
def nan_df():
    rng = pd.date_range("2020-01-01", periods=10, freq="D")
    data = {"value": [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10], "value2": [1, 2, 3, 4, 5, 6, 7, 8, np.nan, 10]}
    return pd.DataFrame(data, index=rng)


def test_timeseries_imputer_less_nans(nan_df):
    imputer = TimeSeriesImputer()
    imputed_df = imputer(nan_df)
    assert np.isnan(imputed_df.values).sum() < np.isnan(nan_df.values).sum()


def test_timeseries_imputer_lags(nan_df):
    imputer = TimeSeriesImputer(lags=[1, -1])
    imputed_df = imputer(nan_df)
    assert np.isnan(imputed_df.values).sum() < np.isnan(nan_df.values).sum()


def test_timeseries_imputer_cols_to_impute(nan_df):
    imputer = TimeSeriesImputer()
    imputed_df = imputer(nan_df, cols_to_impute=["value"])
    assert np.isnan(imputed_df["value"]).sum() == 0
    assert np.isnan(imputed_df["value2"]).sum() == np.isnan(nan_df["value2"]).sum()


def test_timeseries_imputer_rows_to_impute(nan_df):
    imputer = TimeSeriesImputer()
    imputed_df = imputer(nan_df, rows_to_impute=nan_df.index[2:7])
    # NaNs outside the range should still be there
    assert np.isnan(imputed_df.loc["2020-01-09", "value2"])
    # NaNs inside the range should be imputed
    assert not np.isnan(imputed_df.loc["2020-01-03", "value"])
    assert not np.isnan(imputed_df.loc["2020-01-07", "value"])


def test_timeseries_imputer_interpolate(nan_df):
    imputer = TimeSeriesImputer(interpolate_gaps_less_than=2)
    imputed_df = imputer(nan_df)
    # The first NaN in 'value' is a gap of 1, so it should be interpolated
    assert not np.isnan(imputed_df.loc["2020-01-03", "value"])
    # The second NaN in 'value' is a gap of 1, so it should be interpolated
    assert not np.isnan(imputed_df.loc["2020-01-07", "value"])
    # The NaN in 'value2' is a gap of 1, so it should be interpolated
    assert not np.isnan(imputed_df.loc["2020-01-09", "value2"])


def test_timeseries_imputer_reindexes_and_imputes_missing_timestamp_block():
    full_index = pd.date_range("2024-01-01", periods=48, freq="h")
    t = np.arange(len(full_index), dtype=np.float32)
    df = pd.DataFrame(
        {
            "load": 0.2 * t + np.sin(2 * np.pi * t / 24),
            "temperature": 10 + np.cos(2 * np.pi * t / 24),
        },
        index=full_index,
    )
    missing_index = full_index[18:24]
    df_missing = df.drop(index=missing_index)

    imputer = TimeSeriesImputer(lags=(1, 2, -1, -2), rng=0)
    imputed_df = imputer(df_missing)

    assert imputed_df.index.equals(full_index)
    assert not imputed_df.loc[missing_index].isna().any().any()
    assert (imputed_df.loc[missing_index] - df.loc[missing_index]).abs().to_numpy().mean() < 0.2


def test_timeseries_imputer_time_features_do_not_duplicate_user_columns():
    full_index = pd.date_range("2024-01-01", periods=36, freq="h")
    t = np.arange(len(full_index), dtype=np.float32)
    df = pd.DataFrame(
        {
            "__time_trend": t,
            "load": np.sin(2 * np.pi * t / 24),
        },
        index=full_index,
    )
    df.loc[full_index[10:14], "load"] = np.nan

    imputed_df = TimeSeriesImputer(lags=(1, -1), rng=0)(df)

    assert list(imputed_df.columns) == ["__time_trend", "load"]
    assert imputed_df.shape == df.shape
    assert not imputed_df.loc[full_index[10:14], "load"].isna().any()


@pytest.fixture
def mixed_nan_df():
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=200, freq="h")
    values = np.sin(2 * np.pi * np.arange(len(index)) / 24) + 0.05 * rng.normal(size=len(index))
    # Categorical column strongly tied to the numeric one so it is learnable.
    labels = np.where(values > 0, "high", "low")
    df = pd.DataFrame(
        {
            "value": values,
            "level": pd.Categorical(labels, categories=["low", "high"]),
        },
        index=index,
    )
    df.loc[index[10:14], "value"] = np.nan
    df.loc[index[50:60], "level"] = np.nan
    df.loc[index[100], ["value", "level"]] = np.nan
    return df


def test_timeseries_imputer_categorical_columns(mixed_nan_df):
    imputer = TimeSeriesImputer(lags=[1, 2, -1, -2], rng=0)
    imputed_df = imputer(mixed_nan_df)

    assert list(imputed_df.columns) == ["value", "level"]
    assert imputed_df.shape == mixed_nan_df.shape
    assert isinstance(imputed_df["level"].dtype, pd.CategoricalDtype)
    assert list(imputed_df["level"].cat.categories) == ["low", "high"]
    assert not imputed_df.isna().any().any()
    # Categorical predictions should track the signal, not just the mode.
    filled = imputed_df.loc[mixed_nan_df.index[50:60], "level"]
    expected = np.where(imputed_df.loc[mixed_nan_df.index[50:60], "value"] > 0, "high", "low")
    assert (filled.to_numpy().astype(str) == expected).mean() >= 0.8


def test_timeseries_imputer_categorical_object_and_bool_columns():
    index = pd.date_range("2020-01-01", periods=120, freq="h")
    t = np.arange(len(index))
    df = pd.DataFrame(
        {
            "value": np.sin(2 * np.pi * t / 24),
            "label": np.where(t % 24 < 12, "day", "night").astype(object),
            "flag": (t % 2 == 0),
        },
        index=index,
    )
    df.loc[index[30:33], "label"] = np.nan
    df["flag"] = df["flag"].astype("boolean")
    df.loc[index[40:42], "flag"] = pd.NA
    df.loc[index[60:62], "value"] = np.nan

    imputed_df = TimeSeriesImputer(lags=[1, -1], rng=0)(df)

    assert not imputed_df["value"].isna().any()
    assert not imputed_df["label"].isna().any()
    assert not imputed_df["flag"].isna().any()
    assert set(imputed_df["label"].unique()) <= {"day", "night"}


def test_timeseries_imputer_categorical_cols_and_rows_to_impute(mixed_nan_df):
    imputer = TimeSeriesImputer(lags=[1, -1], rng=0)
    imputed_df = imputer(mixed_nan_df, cols_to_impute=["level"])
    assert not imputed_df["level"].isna().any()
    assert imputed_df["value"].isna().sum() == mixed_nan_df["value"].isna().sum()

    imputed_df = imputer(mixed_nan_df, rows_to_impute=mixed_nan_df.index[50:60])
    assert not imputed_df.loc[mixed_nan_df.index[50:60], "level"].isna().any()
    assert imputed_df.loc[mixed_nan_df.index[100], "level"] is np.nan or pd.isna(
        imputed_df.loc[mixed_nan_df.index[100], "level"]
    )


def test_timeseries_imputer_categorical_n_nearest_features(mixed_nan_df):
    imputer = TimeSeriesImputer(lags=[1, -1], rng=0)
    imputer(mixed_nan_df, n_nearest_features=3)
    assert imputer.imputation_features_ is not None
    assert set(imputer.imputation_features_.keys()) == {"value", "level"}
    for features in imputer.imputation_features_.values():
        assert all(isinstance(f, str) for f in features)


def test_timeseries_imputer_invalid_lags():
    with pytest.raises(ValueError):
        TimeSeriesImputer(lags=[1, 0])


def test_timeseries_imputer_set_params_propagates_fallback():
    imputer = TimeSeriesImputer()
    assert imputer.multivariate_imputer.fallback == "simple"
    imputer.set_params(fallback=None)
    assert imputer.multivariate_imputer.fallback is None


def test_timeseries_imputer_n_nearest_features_tracking(nan_df):
    imputer = TimeSeriesImputer(rng=0, lags=[1, -1])
    n_nearest_features = 3
    imputer(nan_df, n_nearest_features=n_nearest_features)

    assert imputer.imputation_features_ is not None

    cols_with_nans = nan_df.columns[nan_df.isnull().any()].tolist()
    assert set(imputer.imputation_features_.keys()) == set(cols_with_nans)

    for col, features in imputer.imputation_features_.items():
        assert isinstance(col, str)
        assert isinstance(features, list)
        assert all(isinstance(f, str) for f in features)
        assert len(features) <= n_nearest_features
        assert col not in features
        # Check that generated temporal context features are present.
        assert any(("_lag_" in f) or f.startswith("__time_") for f in features)


def test_timeseries_imputer_keeps_fully_missing_original_column():
    """A column with no observed value must survive to the output layout.

    Generated lag/lead features that are entirely NaN are dropped, but dropping
    an original column would leave the result missing it entirely.
    """
    index = pd.date_range("2021-01-01", periods=60, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=60),
            "b": rng.normal(size=60),
            "dead": np.full(60, np.nan),
        },
        index=index,
    )
    df.loc[df.index[10:15], "a"] = np.nan

    imputed = TimeSeriesImputer(lags=(1, -1), rng=0)(df)

    assert list(imputed.columns) == ["a", "b", "dead"]
    assert not imputed["a"].isna().any()
    assert imputed["dead"].isna().all()


def test_timeseries_imputer_lag_matrix_matches_shifted_frames():
    """The preallocated lag matrix must match a concat of shifted frames."""
    from datafiller.timeseries._utils import build_lag_matrix

    index = pd.date_range("2020-01-01", periods=12, freq="D")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(12, 3)), index=index, columns=["a", "b", "c"])
    df.iloc[[2, 7], [0, 2]] = np.nan
    lags = [1, 3, -2]

    values = df.to_numpy()
    sources = np.array([j for _ in lags for j in range(3)], dtype=np.int64)
    shifts = np.array([lag for lag in lags for _ in range(3)], dtype=np.int64)
    actual = build_lag_matrix(values, sources, shifts, np.zeros((12, 0)))
    expected = pd.concat([df] + [df.shift(lag) for lag in lags], axis=1).to_numpy()

    np.testing.assert_array_equal(actual, expected)


def test_timeseries_imputer_lag_matrix_gathers_arbitrary_pairs():
    """Non-uniform (source, shift) pairs, including ones the kernel can coalesce, match shifted series."""
    from datafiller.timeseries._utils import build_lag_matrix

    index = pd.date_range("2020-01-01", periods=15, freq="D")
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(15, 4)), index=index, columns=["a", "b", "c", "d"])
    df.iloc[[1, 9], [1, 3]] = np.nan
    pairs = [(2, 1), (2, 4), (0, -1), (1, -1), (2, -1), (3, 2), (1, 2)]
    time_features = rng.normal(size=(15, 2))

    actual = build_lag_matrix(
        df.to_numpy(),
        np.array([j for j, _ in pairs], dtype=np.int64),
        np.array([s for _, s in pairs], dtype=np.int64),
        time_features,
    )
    expected = np.column_stack(
        [df.to_numpy()] + [df.iloc[:, j].shift(s).to_numpy() for j, s in pairs] + [time_features]
    )

    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Per-column and duration lags
# ---------------------------------------------------------------------------


def _golden_frame():
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=60, freq="h")
    t = np.arange(60)
    df = pd.DataFrame(
        {
            "a": np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=60),
            "b": 0.05 * t + 0.1 * rng.normal(size=60),
            "c": rng.normal(size=60),
        },
        index=index,
    )
    mask = rng.random(df.shape) < 0.15
    return df.mask(mask), mask


def test_timeseries_imputer_iterable_lags_match_release_outputs():
    """Imputations with iterable lags are frozen at their v0.3.3 values."""
    # fmt: off
    golden = np.array([
        -0.00322389602661133, 0.6436628699302673, -0.2559864819049835, 0.9372581839561462, -0.6028189063072205,
        0.3926200866699219, 0.26610419154167175, 0.56221604347229, 0.6828031539916992, 0.7291069626808167,
        0.6086652874946594, 0.8718618154525757, 1.040459394454956, -0.3374992907047272, 0.2701805531978607,
        -0.11822107434272766, -0.8562461137771606, 1.3699486255645752, -0.47643613815307617, 0.9691806435585022,
        -0.838117241859436, 1.5978220701217651, 1.7331643104553223, 1.758444905281067, 0.02916000038385391,
        0.20251613855361938, 1.9289424419403076, -0.9858637452125549, 2.474794864654541, 0.6967207789421082,
        -0.07014787197113037, 2.5411877632141113, 2.7516815662384033, 2.965233087539673, -0.16304358839988708,
    ])
    # fmt: on
    df, mask = _golden_frame()
    imputer = TimeSeriesImputer(lags=(1, 2, -1), rng=0)
    out = imputer(df)

    np.testing.assert_allclose(out.to_numpy()[mask], golden, rtol=1e-5, atol=1e-6)
    lagged = [f"{col}_lag_{lag}" for lag in (1, 2, -1) for col in "abc"]
    assert imputer.feature_names_[:12] == ["a", "b", "c", *lagged]


def _forecast_frame(seed=0, n=240, n_noise=8, horizon=12):
    """A target driven by an exogenous series plus AR(1) noise, with its last `horizon` rows blanked."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    driver = np.sin(2 * np.pi * np.arange(n) / 24) + 0.3 * rng.normal(size=n)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + 0.2 * rng.normal()
    data = {"y": driver + noise, "driver": driver}
    for k in range(n_noise):
        data[f"x{k}"] = rng.normal(size=n)
    df = pd.DataFrame(data, index=index)
    truth = df["y"].iloc[-horizon:].copy()
    df.iloc[-horizon:, 0] = np.nan
    return df, truth, index[-horizon - 1]


@pytest.mark.parametrize("n_nearest_features", [None, 3])
def test_timeseries_imputer_mapping_lags_match_hand_made_columns(n_nearest_features):
    df, _, _ = _forecast_frame(n_noise=2)
    df.iloc[[30, 31, 100], 0] = np.nan
    df.iloc[[50, 120], 1] = np.nan

    mapped = TimeSeriesImputer(lags={"y": (1, 2)}, rng=0)
    out = mapped(df, cols_to_impute=["y"], n_nearest_features=n_nearest_features)

    manual_df = df.assign(y_lag_1=df["y"].shift(1), y_lag_2=df["y"].shift(2))
    manual = TimeSeriesImputer(lags=(), rng=0)
    expected = manual(manual_df, cols_to_impute=["y"], n_nearest_features=n_nearest_features)

    assert mapped.feature_names_ == manual.feature_names_
    np.testing.assert_array_equal(out["y"].to_numpy(), expected["y"].to_numpy())
    assert mapped.imputation_features_ == manual.imputation_features_


def test_timeseries_imputer_mapping_lags_layout_and_unlagged_target():
    df, _, _ = _forecast_frame(n_noise=1)
    df.iloc[40:44, 1] = np.nan

    imputer = TimeSeriesImputer(lags={"x0": (2, -1), "y": (1,)}, rng=0)
    out = imputer(df)

    # Frame column order, then lags in the order written; `driver` gets no lag.
    assert imputer.feature_names_ == [
        "y",
        "driver",
        "x0",
        "y_lag_1",
        "x0_lag_2",
        "x0_lag_-1",
        "__time_trend",
        "__time_day_sin",
        "__time_day_cos",
        "__time_week_sin",
        "__time_week_cos",
    ]
    assert not out["driver"].isna().any()


def test_timeseries_imputer_mapping_empty_iterable_means_no_lags():
    df, _, _ = _forecast_frame(n_noise=1)
    imputer = TimeSeriesImputer(lags={"y": (), "x0": (1,)}, add_time_features=False)
    imputer(df)
    assert imputer.feature_names_ == ["y", "driver", "x0", "x0_lag_1"]


def test_timeseries_imputer_duration_lags_equal_steps():
    df, _, cutoff = _forecast_frame(n_noise=1)

    by_steps = TimeSeriesImputer(lags={"y": (1, 24, -1), "driver": (2,)})
    by_duration = TimeSeriesImputer(lags={"y": ("1h", "1D", pd.Timedelta(hours=-1)), "driver": (2,)})
    expected = by_steps(df, after=cutoff)
    actual = by_duration(df, after=cutoff)

    assert by_duration.feature_names_ == by_steps.feature_names_
    assert "y_lag_24" in by_duration.feature_names_
    assert "y_lag_-1" in by_duration.feature_names_
    pd.testing.assert_frame_equal(actual, expected)


def test_timeseries_imputer_iterable_duration_lags_equal_steps():
    df, _, _ = _forecast_frame(n_noise=1)
    pd.testing.assert_frame_equal(
        TimeSeriesImputer(lags=("2h", 1, "-1h", np.timedelta64(3, "h")))(df),
        TimeSeriesImputer(lags=(2, 1, -1, 3))(df),
    )


@pytest.mark.parametrize(
    ("index", "lag"),
    [
        # Daylight saving time makes one day 23 hours long; "1D" is still one step.
        (pd.date_range("2024-03-25", periods=14, freq="D", tz="Europe/Paris"), "1D"),
        (pd.date_range("2024-01-07", periods=14, freq="W"), "7D"),
    ],
)
def test_timeseries_imputer_duration_lags_use_the_nominal_frequency(index, lag):
    df = pd.DataFrame({"y": np.sin(np.arange(len(index), dtype=float)), "x": np.arange(len(index), dtype=float)})
    df.index = index
    df.iloc[[4, 9], 0] = np.nan
    pd.testing.assert_frame_equal(
        TimeSeriesImputer(lags={"y": (lag,)})(df),
        TimeSeriesImputer(lags={"y": (1,)})(df),
    )


def test_timeseries_imputer_feature_names_kept_when_a_call_fails():
    df, _, _ = _forecast_frame(n_noise=1)
    imputer = TimeSeriesImputer(lags=(1,))
    imputer(df)
    expected = imputer.feature_names_
    imputer.set_params(lags=(1, 2))
    with pytest.raises(KeyError):
        imputer(df, cols_to_impute=["missing"])
    assert imputer.feature_names_ == expected


@pytest.mark.parametrize(
    ("freq", "lag", "match"),
    [
        ("h", "90min", r"Lag '90min' is not a non-zero whole number of steps .*0 days 01:00:00"),
        ("D", "1h", r"Lag '1h' is not a non-zero whole number of steps .*1 days"),
        ("MS", "30D", r"Duration lag '30D' needs an index with a fixed-length frequency, got 'MS'"),
    ],
)
def test_timeseries_imputer_duration_lag_must_be_whole_steps(freq, lag, match):
    index = pd.date_range("2024-01-01", periods=30, freq=freq)
    df = pd.DataFrame({"y": np.arange(30, dtype=float)}, index=index)
    df.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        TimeSeriesImputer(lags={"y": (lag,)})(df)
    # Integer lags need no fixed-length frequency.
    assert not TimeSeriesImputer(lags={"y": (1,)})(df).isna().any().any()


def test_timeseries_imputer_duplicate_lags_raise():
    df, _, _ = _forecast_frame(n_noise=1)
    with pytest.raises(ValueError, match="duplicates another lag of 1 steps"):
        TimeSeriesImputer(lags=(1, "1h"))(df)
    with pytest.raises(ValueError, match="of column 'y' duplicates"):
        TimeSeriesImputer(lags={"y": (24, "1D")})(df)
    with pytest.raises(ValueError, match="duplicates"):
        TimeSeriesImputer(lags=(1, 1))(df)


def test_timeseries_imputer_unknown_lag_column_raises():
    df, _, _ = _forecast_frame(n_noise=1)
    with pytest.raises(ValueError, match=r"columns not found in the data: \['z'\]"):
        TimeSeriesImputer(lags={"y": (1,), "z": (1,)})(df)
    # Keys are labels, not positions.
    with pytest.raises(ValueError, match=r"columns not found in the data: \[0\]"):
        TimeSeriesImputer(lags={0: (1,)})(df)


def test_timeseries_imputer_mapping_lags_with_integer_column_labels():
    df, _, cutoff = _forecast_frame(n_noise=1)
    renamed = df.set_axis(range(df.shape[1]), axis=1)
    imputer = TimeSeriesImputer(lags={0: (1, "2h")})
    out = imputer(renamed, after=cutoff)

    assert imputer.feature_names_[: df.shape[1] + 2] == [*range(df.shape[1]), "0_lag_1", "0_lag_2"]
    expected = TimeSeriesImputer(lags={df.columns[0]: (1, 2)})(df, after=cutoff)
    np.testing.assert_array_equal(out.to_numpy(), expected.to_numpy())


@pytest.mark.parametrize(
    ("lags", "match"),
    [
        ({"y": (1, 0)}, "cannot contain 0"),
        ((1, "0h"), "cannot contain 0"),
        ({"y": 1}, "lags must be an iterable"),
        ("1h", "lags must be an iterable"),
        ((True,), "lags must be an iterable"),
        (("soon",), "Could not parse 'soon'"),
        (("24",), "'24' has no unit"),
        ((lag for lag in (1, 2)), "not an iterator"),
        ({"y": iter((1, 2))}, "not an iterator"),
    ],
)
def test_timeseries_imputer_invalid_lag_specs(lags, match):
    with pytest.raises(ValueError, match=match):
        TimeSeriesImputer(lags=lags)
    with pytest.raises(ValueError, match=match):
        TimeSeriesImputer().set_params(lags=lags)


def test_timeseries_imputer_categorical_path_with_mapping_lags(mixed_nan_df):
    imputer = TimeSeriesImputer(lags={"level": (1, -1), "value": ("2h",)}, rng=0)
    out = imputer(mixed_nan_df)

    assert imputer.feature_names_[:5] == ["value", "level", "value_lag_2", "level_lag_1", "level_lag_-1"]
    assert isinstance(out["level"].dtype, pd.CategoricalDtype)
    assert not out.isna().any().any()

    # Same names and order as the numeric path for the same mapping.
    numeric = mixed_nan_df.assign(level=(mixed_nan_df["level"] == "high").astype(float))
    numeric_imputer = TimeSeriesImputer(lags={"level": (1, -1), "value": ("2h",)}, rng=0)
    numeric_imputer(numeric)
    assert numeric_imputer.feature_names_ == imputer.feature_names_


def test_timeseries_imputer_target_only_lags_beat_lagging_every_column():
    """Regression guard for the forecasting pattern, not a general claim."""
    lags = (1, 2, 3, 24)
    df, truth, cutoff = _forecast_frame(seed=0)

    def rmse(imputer):
        out = imputer(df, cols_to_impute=["y"], after=cutoff)
        return float(np.sqrt(((out["y"].iloc[-len(truth) :] - truth) ** 2).mean()))

    every_column = rmse(TimeSeriesImputer(lags=lags, add_time_features=False))
    target_only = rmse(TimeSeriesImputer(lags={"y": lags}, add_time_features=False))
    assert target_only < every_column
