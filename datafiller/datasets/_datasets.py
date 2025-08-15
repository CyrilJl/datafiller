import pandas as pd

# This file will contain functions to fetch popular time series datasets.


def load_pems_bay() -> pd.DataFrame:
    """Downloads and returns the PEMS-BAY dataset.

    The dataset is downloaded and cached using pooch.

    Returns:
        pd.DataFrame: The PEMS-BAY dataset, with a DatetimeIndex named 'time' and
            a frequency of 5 minutes.

    Raises:
        ImportError: If pooch is not installed.
    """
    try:
        import pooch
    except ImportError:
        raise ImportError("pooch is required to download the dataset. Please install it with `pip install pooch`.")

    url = "https://zenodo.org/records/5724362/files/PEMS-BAY.csv"
    known_hash = "md5:c8dea58987a5882e946217c22fdb8256"

    # Use pooch to download and cache the dataset
    file_path: str = pooch.retrieve(
        url=url,
        known_hash=known_hash,
        progressbar=True,
    )

    # Load the dataset with pandas
    df: pd.DataFrame = pd.read_csv(file_path, index_col=0, parse_dates=[0])
    df = df.rename_axis(index="time", columns="sensor_id")
    df = df.asfreq("5min")

    return df
