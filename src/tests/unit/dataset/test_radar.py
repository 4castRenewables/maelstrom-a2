import gzip
import pathlib

import a2.dataset
import a2.utils
import distutils.dir_util
import numpy as np
import pytest
import xarray

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


def setup_directory(tmp_path, name_subdirectory):
    directory = tmp_path / name_subdirectory / "badc/"
    directory.mkdir(parents=True)
    folder_tarball_origin = DATA_FOLDER / "radar/badc"
    distutils.dir_util.copy_tree(folder_tarball_origin.__str__(), directory.__str__())
    return directory


def test_assign_radar_to_tweets(
    fake_dataset_tweets_and_precipitation_no_radar, fake_dataset_tweets_and_precipitation_radar
):
    ds_tweets = a2.dataset.load_dataset.reset_index_coordinate(fake_dataset_tweets_and_precipitation_no_radar)
    ds_radar = a2.dataset.radar.assign_radar_to_tweets(
        ds_tweets,
        key_tweets=None,
        round_ngt_offset=500,
        round_ngt_decimal=-3,
        round_time_to_base=5,
        path_to_dapceda=None,
        processes=1,
    )
    xarray.testing.assert_equal(ds_radar, fake_dataset_tweets_and_precipitation_radar)


def test_all_nimrod_daily_tarball_to_netcdf(tmp_path, fake_dataset_tweets_no_radar, fake_dataset_tweets_radar):
    directory = setup_directory(tmp_path, "test_all_nimrod_daily_tarball_to_netcdf")

    a2.dataset.radar.all_nimrod_daily_tarball_to_netcdf(
        base_folder=directory,
        chunk_size=1,
        processes=1,
    )
    a2.dataset.radar.merge_nc_files_on_daily_basis(base_folder=directory, processes=1, remove_files=True)
    ds_radar = a2.dataset.radar.assign_radar_to_tweets_from_netcdf(
        ds_tweets=fake_dataset_tweets_no_radar, base_folder=directory
    )

    xarray.testing.assert_equal(ds_radar, fake_dataset_tweets_radar)


def test_nimrod_class_query():
    filename = DATA_FOLDER / "radar/metoffice-c-band-rain-radar_uk_202002070000_1km-composite.dat.gz"
    compare = "NIMROD file raw header fields listed by element number:\nGeneral (Integer) header entries:\n  1 \t 2020\n  2 \t 2\n  3 \t 7\n  4 \t 0\n  5 \t 0\n  6 \t 0\n  7 \t 2020\n  8 \t 2\n  9 \t 7\n  10 \t 0\n  11 \t 0\n  12 \t 1\n  13 \t 2\n  14 \t -32767\n  15 \t 0\n  16 \t 2175\n  17 \t 1725\n  18 \t 2\n  19 \t 213\n  20 \t 5\n  21 \t -32767\n  22 \t 0\n  23 \t 11\n  24 \t 0\n  25 \t -1\n  26 \t -32767\n  27 \t -32767\n  28 \t -32767\n  29 \t -32767\n  30 \t -32767\n  31 \t -32767\nGeneral (Real) header entries:\n  32 \t 9999.0\n  33 \t -32767.0\n  34 \t 1549500.0\n  35 \t 1000.0\n  36 \t -404500.0\n  37 \t 1000.0\n  38 \t -1.0\n  39 \t 0.0\n  40 \t 0.0\n  41 \t -32767.0\n  42 \t -32767.0\n  43 \t -32767.0\n  44 \t -32767.0\n  45 \t -32767.0\n  46 \t -32767.0\n  47 \t -32767.0\n  48 \t -32767.0\n  49 \t -32767.0\n  50 \t -32767.0\n  51 \t -32767.0\n  52 \t -32767.0\n  53 \t -32767.0\n  54 \t -32767.0\n  55 \t -32767.0\n  56 \t -32767.0\n  57 \t -32767.0\n  58 \t -32767.0\n  59 \t -32767.0\nData Specific (Real) header entries (0):\nData Specific (Integer) header entries (12):\n  108 \t 0\n  109 \t -5124\n  110 \t 94\n  111 \t 0\n  112 \t 0\n  113 \t 0\n  114 \t 0\n  115 \t 0\n  116 \t 0\n  117 \t 1\n  118 \t -32767\n  119 \t -32767\nCharacter header entries:\n  105 Units:            mm/h*32\x00\n  106 Data source:      Plr single site radars\x00\x00\n  107 Title of field:   Rainfall rate Composite\n\nValidity Time:  00:00 on 07/02/2020\nEasting range:  -405000.0 - 1320000.0 (at pixel steps of 1000.0)\nNorthing range: -625000.0 - 1550000.0 (at pixel steps of 1000.0)\nImage size: 2175 rows x 1725 cols\n"  # noqa
    with gzip.open(filename, "rb") as file_content:
        nimrod = a2.dataset.radar.Nimrod(file_content=file_content)
        io_capture = a2.utils.testing.IOCapture()
        nimrod.query()
        query_as_string = io_capture.return_capture_stop()
    assert query_as_string == compare


@pytest.mark.optional
def test_nimrod_ds_cumulative_from_time(tmp_path, fake_cumulative_dataset):
    setup_directory(tmp_path, "test_nimrod_ds_cumulative_from_time")

    ds_radar = a2.dataset.radar.nimrod_ds_cumulative_from_time(
        path_to_dapceda=tmp_path / "test_nimrod_ds_cumulative_from_time/",
        time=np.datetime64("2020-02-07T00:05:00.000000000"),
        time_delta=10,
        time_delta_units="m",
    )
    xarray.testing.assert_equal(ds_radar, fake_cumulative_dataset)


def test_time_series_from_files(tmp_path):
    setup_directory(tmp_path, "test_time_series_from_files")

    time, tp = a2.dataset.radar.time_series_from_files(
        time_start=np.datetime64("2020-02-07T00:00:00.000000000"),
        time_end=np.datetime64("2020-02-07T00:05:00.000000000"),
        longitudes=[-5.44],
        latitudes=[58.17],
        path_to_dapceda=tmp_path / "test_time_series_from_files/",
        time_delta=5,
        time_delta_units="m",
        processes=1,
    )
    assert np.array_equal(
        time, np.array([np.datetime64("2020-02-07T00:00:00.000000000"), np.datetime64("2020-02-07T00:05:00.000000000")])
    )
    assert np.array_equal(tp, np.array([[[0.015625]], [[0]]]))
