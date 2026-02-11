import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datafiller import MultivariateImputer, TimeSeriesImputer


def test_multivariate_imputer_pipeline_fit_transform():
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )
    pipe = Pipeline([("imputer", MultivariateImputer())])
    X_imputed = pipe.fit_transform(X)

    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()


def test_timeseries_imputer_pipeline_fit_transform():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]}, index=idx)

    pipe = Pipeline([("imputer", TimeSeriesImputer(lags=(1,)))])
    df_imputed = pipe.fit_transform(df)

    assert isinstance(df_imputed, pd.DataFrame)
    assert df_imputed.shape == df.shape
    assert not df_imputed.isna().any().any()
