"""Optional Polars dataframe encoding helpers."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..exceptions import DataFillerValueError


def _get_polars():
    import polars

    return polars


def is_polars_dataframe(value: object) -> bool:
    """Return whether *value* is an eager Polars DataFrame."""
    polars = sys.modules.get("polars")
    return polars is not None and isinstance(value, polars.DataFrame)


def is_polars_lazyframe(value: object) -> bool:
    """Return whether *value* is a Polars LazyFrame."""
    polars = sys.modules.get("polars")
    return polars is not None and isinstance(value, polars.LazyFrame)


def polars_cols_to_indices(cols_to_impute, columns: list[str]) -> np.ndarray:
    """Convert Polars column names to integer positions."""
    if cols_to_impute is None:
        return np.arange(len(columns))

    requested = (
        [cols_to_impute]
        if isinstance(cols_to_impute, str) or not isinstance(cols_to_impute, Iterable)
        else list(cols_to_impute)
    )
    names: list[str] = []
    for name in requested:
        if not isinstance(name, str):
            raise DataFillerValueError("cols_to_impute must contain Polars column names as strings.")
        names.append(name)

    positions = {name: i for i, name in enumerate(columns)}
    missing = [name for name in names if name not in positions]
    if missing:
        raise DataFillerValueError(f"Column names not found in Polars DataFrame: {missing}")
    return np.array([positions[name] for name in names], dtype=np.int64)


def validate_polars_rows(rows_to_impute, height: int):
    """Validate the positional row selector used by Polars DataFrames."""
    if rows_to_impute is None or isinstance(rows_to_impute, (int, np.integer)):
        return None if rows_to_impute is None else int(rows_to_impute)
    if isinstance(rows_to_impute, np.ndarray):
        return rows_to_impute
    if isinstance(rows_to_impute, str) or not isinstance(rows_to_impute, Iterable):
        raise DataFillerValueError("rows_to_impute must contain integer row positions for a Polars DataFrame.")
    rows = list(rows_to_impute)
    if not all(isinstance(row, (int, np.integer)) and 0 <= row < height for row in rows):
        raise DataFillerValueError(f"rows_to_impute must contain integer row positions between 0 and {height - 1}.")
    return rows


def _is_categorical_dtype(dtype: Any) -> bool:
    pl = _get_polars()
    return dtype == pl.Boolean or dtype == pl.String or dtype.base_type() in {pl.Categorical, pl.Enum}


def encode_polars_dataframe(df) -> dict:
    """Encode an eager Polars DataFrame into the imputer's numeric representation."""
    pl = _get_polars()
    encoded_arrays = []
    encoded_feature_names: list[str] = []
    main_column_indices: list[int] = []
    categorical_targets: dict[int, list] = {}
    encoded_index_to_original: dict[int, str] = {}
    original_dtypes = dict(df.schema)
    null_masks: dict[str, np.ndarray] = {}
    numeric_main_indices: list[int] = []

    for col in df.columns:
        series = df.get_column(col)
        dtype = series.dtype
        main_idx = len(encoded_feature_names)
        encoded_index_to_original[main_idx] = col
        main_column_indices.append(main_idx)
        encoded_feature_names.append(col)
        null_masks[col] = series.is_null().to_numpy()

        if _is_categorical_dtype(dtype) or dtype == pl.Null:
            categories = [] if dtype == pl.Null else series.drop_nulls().unique(maintain_order=True).to_list()
            category_to_code = {value: i for i, value in enumerate(categories)}
            values = series.to_list()
            codes = np.array(
                [np.nan if value is None else category_to_code[value] for value in values], dtype=np.float32
            )
            categorical_targets[main_idx] = categories
            encoded_arrays.append(codes.reshape(-1, 1))

            if categories:
                missing = np.isnan(codes)
                dummies = np.column_stack([(codes == i).astype(np.float32) for i in range(len(categories))])
                dummies[missing] = np.nan
                encoded_arrays.append(dummies)
                encoded_feature_names.extend(f"{col}_{category}" for category in categories)
        elif dtype.is_numeric():
            encoded_arrays.append(series.cast(pl.Float32).to_numpy().reshape(-1, 1))
            numeric_main_indices.append(main_idx)
        else:
            raise DataFillerValueError(
                f"Unsupported Polars dtype for column {col!r}: {dtype}. "
                "Supported dtypes are numeric, Boolean, String, Categorical, and Enum."
            )

    encoded_matrix = np.concatenate(encoded_arrays, axis=1).astype(np.float32, copy=False)
    return {
        "data": encoded_matrix,
        "main_column_indices": np.array(main_column_indices, dtype=np.int64),
        "encoded_feature_names": encoded_feature_names,
        "categorical_targets": categorical_targets,
        "encoded_index_to_original": encoded_index_to_original,
        "original_columns": list(df.columns),
        "original_dtypes": original_dtypes,
        "null_masks": null_masks,
        "numeric_main_indices": np.array(numeric_main_indices, dtype=np.int64),
    }


def decode_polars_dataframe(x_imputed: np.ndarray, metadata: dict):
    """Restore an imputed matrix to the original Polars schema."""
    pl = _get_polars()
    series_out = []
    for i, col in enumerate(metadata["original_columns"]):
        encoded_idx = metadata["main_column_indices"][i]
        col_data = x_imputed[:, encoded_idx]
        dtype = metadata["original_dtypes"][col]

        if encoded_idx in metadata["categorical_targets"]:
            categories = metadata["categorical_targets"][encoded_idx]
            decoded: list[Any] = [None] * len(col_data)
            if categories:
                for row, value in enumerate(col_data):
                    if np.isfinite(value):
                        decoded[row] = categories[int(value)]
            series = pl.Series(col, decoded, dtype=dtype)
        elif dtype.is_integer():
            decoded = [None if np.isnan(value) else int(np.rint(value)) for value in col_data]
            series = pl.Series(col, decoded, dtype=dtype)
        else:
            null_mask = metadata["null_masks"][col]
            decoded = [None if null_mask[row] and np.isnan(value) else value for row, value in enumerate(col_data)]
            series = pl.Series(col, decoded, dtype=dtype)
        series_out.append(series)

    return pl.DataFrame(series_out)
