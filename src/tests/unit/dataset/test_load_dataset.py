import os
import pathlib
from contextlib import nullcontext as doesnotraise

import a2.dataset
import a2.utils.file_handling
import a2.utils.testing
import numpy as np
import pytest
import pytest_cases
import xarray

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


@pytest_cases.parametrize(
    "to_test, expected",
    [
        (np.nan, np.nan),
        ("{'a': 1, 'b': 2}", {"a": 1, "b": 2}),
        ("{a: 1, b: 2}", Exception()),
    ],
)
def test__convert_str_to_dict(to_test, expected):
    with pytest.raises(type(expected)) if isinstance(expected, Exception) else doesnotraise():
        assert a2.dataset.load_dataset._convert_str_to_dict(to_test) == expected or (
            np.isnan(to_test) and np.isnan(expected)
        )


def test_save_dataset(tmp_path, fake_add_locations_dataframe_solution):
    directory = tmp_path / "test_save_dataset/"
    directory.mkdir()

    filename = "tmp_test.nc"
    a2.dataset.load_dataset.save_dataset(
        fake_add_locations_dataframe_solution, directory / filename, no_conversion=False
    )
    assert os.path.isfile(directory / filename)


def test_save_dataset_split(tmp_path, fake_dataset_precipitation):
    directory = tmp_path / "test_save_dataset/"
    directory.mkdir()
    a2.dataset.load_dataset.save_dataset_split(
        fake_dataset_precipitation,
        split_by="year",
        prefix=directory / "ds_",
        key_time="time_half",
    )
    weather_filenames = a2.utils.file_handling.get_all_files(directory / "ds_*.nc")
    xarray.testing.assert_equal(xarray.open_mfdataset(weather_filenames), fake_dataset_precipitation)


def assert_equal_jsons(fake_dataset_precipitation):
    with pytest.raises(NotImplementedError):
        a2.dataset.load_dataset.save_dataset_split(
            fake_dataset_precipitation,
            split_by="doesnt_exist",
            prefix=FILE_LOC / "ds_",
        )


def test_load_tweets_dataframe_from_jsons(fake_tweets_json_filepath, fake_tweets_dataframe):
    df = a2.dataset.load_dataset.load_tweets_dataframe_from_jsons([fake_tweets_json_filepath])
    a2.utils.testing.assert_equal_pandas_dataframe(df, fake_tweets_dataframe)


def test_load_weather_stations(fake_weather_station_dataframe):
    df = a2.dataset.load_dataset.load_weather_stations(DATA_FOLDER / "weather_stations_test.csv")
    a2.utils.testing.assert_equal_pandas_dataframe(df, fake_weather_station_dataframe)
