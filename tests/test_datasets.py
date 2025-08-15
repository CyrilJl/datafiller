import pytest
import pandas as pd
from pytest_mock import MockerFixture

from datafiller.datasets import load_pems_bay


def test_load_pems_bay_success() -> None:
    """Tests that the PEMS-BAY dataset is loaded correctly."""
    df = load_pems_bay()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freqstr == "5min"


def test_load_pems_bay_no_pooch(mocker: MockerFixture) -> None:
    """Tests that an ImportError is raised if pooch is not installed.

    Args:
        mocker: The pytest-mock mocker fixture.
    """
    mocker.patch.dict("sys.modules", {"pooch": None})
    with pytest.raises(ImportError, match="pooch is required"):
        load_pems_bay()
