import pathlib

import a2.dataset
import a2.utils
import numpy as np
import xarray

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


def test_add_station_precipitation(fake_tweets_stations, fake_tweets_no_stations, fake_weather_stations):
    ds = a2.dataset.stations.add_station_precipitation(
        ds=fake_tweets_no_stations,
        df_stations=fake_weather_stations,
        processes=1,
    )
    xarray.testing.assert_equal(ds, fake_tweets_stations)


def test_get_counts_station(fake_weather_stations):
    fake_weather_stations = a2.dataset.stations.add_station_number(fake_weather_stations)
    station_numbers, counts = a2.dataset.stations.get_counts_station(fake_weather_stations)
    assert np.array_equal(station_numbers, [1, 0])
    assert np.array_equal(counts, [2, 2])


def test_get_time_series_from_station_number(fake_weather_stations):
    fake_weather_stations = a2.dataset.stations.add_station_number(fake_weather_stations)
    time, tp = a2.dataset.stations.get_time_series_from_station_number(fake_weather_stations, 0)
    time_expected = np.array(
        [
            "2018-08-15T11:00:00.000000000",
            "2018-08-15T21:00:00.000000000",
        ],
        dtype=np.datetime64,
    )
    tp_expected = np.array([5.0, 9.6], dtype=float)
    assert np.array_equal(time, time_expected)
    assert np.array_equal(tp, tp_expected)
