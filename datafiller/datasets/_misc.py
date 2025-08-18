import numpy as np
import pandas as pd
from typing import Union


def add_mar(df: pd.DataFrame, nan_ratio: float) -> pd.DataFrame:
    """Add Missing At Random (MAR) values to a DataFrame.

    Args:
        df: Input dataframe
        nan_ratio: Ratio of values to set as NaN (between 0 and 1)

    Returns:
        DataFrame with MAR values

    Raises:
        ValueError: If nan_ratio is not between 0 and 1
    """
    if not 0 <= nan_ratio <= 1:
        raise ValueError("nan_ratio must be between 0 and 1")

    df_copy = df.copy()
    mask = np.random.random(df_copy.shape) < nan_ratio
    df_copy[mask] = np.nan
    return df_copy


def add_contiguous_missing(df: pd.DataFrame, frac_columns: float, length: Union[int, float]) -> pd.DataFrame:
    """Add contiguous blocks of missing values to random columns.

    Args:
        df: Input dataframe
        frac_columns: Fraction of columns to modify (0-1)
        length: Length of missing block (int for absolute, float for relative)

    Returns:
        DataFrame with contiguous missing blocks

    Raises:
        ValueError: If frac_columns or length are invalid
    """
    if not 0 <= frac_columns <= 1:
        raise ValueError("frac_columns must be between 0 and 1")

    df_copy = df.copy()
    cols_to_modify = np.random.choice(df_copy.columns, size=int(len(df_copy.columns) * frac_columns), replace=False)

    for col in cols_to_modify:
        n_rows = len(df_copy)
        block_length = int(n_rows * length) if isinstance(length, float) else length
        block_length = min(block_length, n_rows)

        start_idx = np.random.randint(0, n_rows - block_length + 1)
        df_copy.loc[df_copy.index[start_idx : start_idx + block_length], col] = np.nan

    return df_copy
