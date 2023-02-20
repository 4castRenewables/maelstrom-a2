import io
import logging
import pathlib
import sys
from contextlib import nullcontext as doesnotraise

import a2.dataset.utils_dataset
import a2.utils.file_handling
import a2.utils.testing
import numpy as np
import pytest
import pytest_cases
import xarray


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


def test_print_tweet_sample(fake_dataset_print):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    a2.dataset.utils_dataset.print_tweet_sample(fake_dataset_print, n=2)
    sys.stdout = sys.__stdout__  # reset redirect
    assert len(captured_output.getvalue()) == 96


def test_print_tweet_groupby(fake_dataset_tweets):
    io_capture = a2.utils.testing.IOCapture()
    ds_grouped = a2.dataset.utils_dataset.print_tweet_groupby(
        fake_dataset_tweets, group_by="author_id", n=2, n_groups=1
    )
    assert len(io_capture.return_capture_stop()) == 168
    io_capture = a2.utils.testing.IOCapture()
    a2.dataset.utils_dataset.print_tweet_groupby(
        fake_dataset_tweets,
        group_by="author_id",
        n=2,
        n_groups=1,
        ds_grouped=ds_grouped,
    )
    assert len(io_capture.return_capture_stop()) == 168


def test_print_tweet_groupby_dataarray_groupby(
    fake_dataset_tweets_no_precipitation,
):
    io_capture = a2.utils.testing.IOCapture()
    a2.dataset.utils_dataset.print_tweet_groupby(
        fake_dataset_tweets_no_precipitation,
        group_by=fake_dataset_tweets_no_precipitation.created_at.dt.day,
        n=2,
        n_groups=1,
    )
    assert len(io_capture.return_capture_stop()) == 158


def test_filter_tweets(fake_dataset_tweets):
    log_stream = io.StringIO()
    logging.basicConfig(stream=log_stream, level=logging.INFO)
    ds_filtered = a2.dataset.utils_dataset.filter_tweets(fake_dataset_tweets, terms=["hi"])
    xarray.testing.assert_equal(ds_filtered, fake_dataset_tweets.sel(index=[1, 2]))


def test_info_tweets_to_text(fake_dataset_tweets_and_precipitation):
    text = a2.dataset.utils_dataset.info_tweets_to_text(
        a2.dataset.load_dataset.reset_index_coordinate(fake_dataset_tweets_and_precipitation).sel(index=slice(2))
    )
    a2.utils.testing.print_text_as_csv(text)
    text_assert = "created_at: 2017-01-02T02:02:02.000000000\nlatitude_rounded: 51.5\nlongitude_rounded: -0.1\ntext: hi, is it raining?\ntp: 9.909272193908691e-07\ncreated_at: 2017-01-02T05:02:02.000000000\nlatitude_rounded: 51.5\nlongitude_rounded: -0.1\ntext: is it raining?\ntp: 0.0\ncreated_at: 2017-01-04T04:44:04.000000000\nlatitude_rounded: 51.4\nlongitude_rounded: 0.2\ntext: maybe it's raining\ntp: 0.0\n"  # noqa: E501
    assert text == text_assert


def test_add_precipitation_memory_efficient(
    tmp_path,
    fake_dataset_tweets_no_precipitation,
    fake_dataset_precipitation,
    fake_dataset_tweets_and_precipitation,
):
    directory = tmp_path / "test_add_precipitation_memory_efficient/"
    directory.mkdir()

    a2.dataset.load_dataset.save_dataset_split(
        fake_dataset_precipitation,
        split_by="day",
        prefix=directory / "ds_",
        key_time="time_half",
    )
    weather_filenames = a2.utils.file_handling.get_all_files(directory / "ds_*.nc")
    ds_tweets_precipitation = a2.dataset.utils_dataset.add_precipitation_memory_efficient(
        fake_dataset_tweets_no_precipitation,
        weather_filenames,
        key_precipitation_tweets="tp",
    )
    xarray.testing.assert_equal(ds_tweets_precipitation, fake_dataset_tweets_and_precipitation)


@pytest_cases.parametrize(
    "data, n, expected",
    [
        (np.array(["this"]), None, "------------------------------\nthis\n"),
        (np.array(["this"]), -1, "------------------------------\nthis\n"),
    ],
)
def test_print_sample(data, n, expected):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    a2.dataset.utils_dataset.print_sample(data, n_sample=1, n=n)
    sys.stdout = sys.__stdout__  # reset redirect
    assert captured_output.getvalue().strip(r"\n").strip("'") == expected


@pytest_cases.parametrize(
    "variable, drop_variable, expected",
    [
        ("time_h", "time_h", "dataset"),
        ("non-existent-variable", "time_h", ValueError()),
    ],
)
def test_add_field(fake_dataset_add_fields, variable, drop_variable, expected):
    with pytest.raises(type(expected)) if isinstance(expected, Exception) else doesnotraise():
        ds_dropped = fake_dataset_add_fields.drop_vars(drop_variable)
        ds_added_back = a2.dataset.utils_dataset.add_field(
            ds_dropped,
            variable,
            coordinates="index",
            overwrite=False,
            rename_coordinate=None,
        )
        xarray.testing.assert_equal(ds_added_back, fake_dataset_add_fields)


@pytest_cases.parametrize(
    "data_var, expected_data",
    [
        (np.array([1, None, np.nan]), np.array([False, True, True])),
    ],
)
def test_is_na(data_var, expected_data):
    coords = dict(index=np.arange(len(data_var)))
    ds = xarray.Dataset(data_vars=dict(field=(["index"], data_var)), coords=coords)
    expected = xarray.DataArray(data=expected_data, coords=coords)
    result = a2.dataset.utils_dataset.is_na(ds, field="field", check=None, dims=None)
    xarray.testing.assert_equal(result, expected)
