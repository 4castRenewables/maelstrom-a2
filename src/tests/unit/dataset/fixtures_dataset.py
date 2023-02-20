import pathlib

import a2.dataset
import a2.utils
import numpy as np
import pandas as pd
import xarray
from pytest_cases import fixture

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


@fixture()
def fake_dataset_print():
    text = ["hi, is it raining?", "is it raining?"]
    index = np.arange(2)
    return xarray.Dataset(data_vars=dict(text=(["index"], text)), coords=dict(index=index))


@fixture()
def basic_data_vars():
    text = np.array(["hi, is it raining?", "is it raining?", "maybe it's raining"], dtype=object)
    source = np.array(["A", "B", "A"], dtype=object)
    author_id = np.array([2.2e8, 1.1e8, 1.1e8], np.float64)
    id = np.array(["111", "222", "333"], dtype=object)
    data_vars = dict(
        text=(["index"], text),
        source=(["index"], source),
        author_id=(["index"], author_id),
        id=(["index"], id),
    )
    return data_vars


@fixture()
def precipitation_tweets_data_vars():
    time = np.array(
        ["2017-01-02T02:02:02.000000000", "2017-01-02T05:02:02.000000000", "2017-01-04T04:44:04.000000000"],
        dtype=np.datetime64,
    )
    created_at = time
    time_half = np.array(
        ["2017-01-02T02:30:00.000000000", "2017-01-02T05:30:00.000000000", "2017-01-04T04:30:00.000000000"],
        dtype=np.datetime64,
    )
    longitude = np.array([-0.123, -0.12, 0.19], dtype=np.float64)
    longitude_rounded = np.array([-0.1, -0.1, 0.2], dtype=np.float64)
    latitude = np.array([51.489, 51.523, 51.423], dtype=np.float64)
    latitude_rounded = np.array([51.5, 51.5, 51.4], dtype=np.float64)
    tp = np.array([9.909272193908691e-07, 0.0, 0.0], dtype=np.float64)
    data_vars = dict(
        time=(["index"], time),
        created_at=(["index"], created_at),
        time_half=(["index"], time_half),
        longitude=(["index"], longitude),
        longitude_rounded=(["index"], longitude_rounded),
        latitude=(["index"], latitude),
        latitude_rounded=(["index"], latitude_rounded),
        tp=(["index"], tp),
    )
    return data_vars


def dataset_from_data_variables(data_vars):
    n_entries = get_n_entries(data_vars)
    index = np.arange(n_entries)
    return xarray.Dataset(
        data_vars=data_vars,
        coords=dict(index=index),
    )


def get_n_entries(data_vars):
    n_per_variable = [len(v[1]) for k, v in data_vars.items()]
    if max(n_per_variable) != min(n_per_variable):
        raise ValueError(f"{n_per_variable=} not consistent!")
    n_entries = n_per_variable[0]
    return n_entries


@fixture()
def fake_dataset_tweets(basic_data_vars):
    return dataset_from_data_variables(basic_data_vars)


@fixture()
def fake_dataset_precipitation():
    return a2.dataset.load_dataset.load_tweets_dataset(
        DATA_FOLDER / "test_dataset_precipitation.nc", raw=True, reset_index_raw=False
    )


@fixture()
def fake_dataset_tweets_no_precipitation(fake_dataset_tweets_and_precipitation):
    return fake_dataset_tweets_and_precipitation.drop_vars(
        [
            "time",
            "time_half",
            "longitude_rounded",
            "latitude_rounded",
            "tp",
        ]
    )


@fixture()
def fake_dataset_tweets_and_precipitation(basic_data_vars, precipitation_tweets_data_vars):
    data_vars = {**basic_data_vars, **precipitation_tweets_data_vars}
    return dataset_from_data_variables(data_vars)


@fixture()
def fake_dataset_tweets_and_precipitation_radar(fake_dataset_tweets_radar):
    fake_dataset_tweets_radar["time_radar_int"] = (
        ["index"],
        np.array([1581033600000000000, 1581033600000000000, 1581033900000000000, 1581034200000000000], dtype=np.int64),
    )
    fake_dataset_tweets_radar["tp_mm_radar"] = (
        ["index"],
        np.array([0.010416666666666666, 0.015625, 0.0, 0.0], dtype=np.float64),
    )
    return fake_dataset_tweets_radar


@fixture()
def fake_dataset_tweets_and_precipitation_no_radar(fake_dataset_tweets_and_precipitation_radar):
    return fake_dataset_tweets_and_precipitation_radar.drop_vars(
        ["time_radar_int", "x_ngt", "x_ngt_rounded", "tp_mm_radar", "y_ngt_rounded", "time_radar", "y_ngt"]
    )


@fixture()
def fake_dataset_tweets_no_radar(fake_dataset_tweets_radar):
    return fake_dataset_tweets_radar.drop_vars(
        ["x_ngt", "x_ngt_rounded", "tp_mm_radar", "y_ngt_rounded", "time_radar", "y_ngt"]
    )


@fixture()
def fake_dataset_tweets_radar():
    created_at = np.array(
        [
            "2020-02-07T00:01:05.000000000",
            "2020-02-07T00:02:05.000000000",
            "2020-02-07T00:06:15.000000000",
            "2020-02-07T00:12:25.000000000",
        ],
        dtype=np.datetime64,
    )
    latitude = np.array([58.16, 58.17, 58.18, 58.19], dtype=np.float64)
    longitude = np.array([-5.48, -5.46, -5.44, -5.42], dtype=np.float64)
    x_ngt = np.array([195867.44770916324, 197100.58069717628, 198333.06975428865, 199564.91472113034], dtype=np.float64)
    x_ngt_rounded = np.array([195500.0, 196500.0, 197500.0, 199500.0], dtype=np.float64)
    y_ngt = np.array([924797.4971129517, 925849.40613528, 926901.6865361285, 927954.3381499702], dtype=np.float64)
    y_ngt_rounded = np.array([924500.0, 925500.0, 926500.0, 927500.0], dtype=np.float64)
    tp_mm_radar = np.array([0.010416666666666666, 0.015625, 0.0, np.nan], dtype=np.float64)
    time_radar = np.array(
        [
            "2020-02-07T00:00:00.000000000",
            "2020-02-07T00:00:00.000000000",
            "2020-02-07T00:05:00.000000000",
            "2020-02-07T00:10:00.000000000",
        ],
        dtype=np.datetime64,
    )
    index = np.array([0, 1, 2, 3], dtype=np.int64)
    return xarray.Dataset(
        data_vars=dict(
            created_at=(["index"], created_at),
            latitude=(["index"], latitude),
            longitude=(["index"], longitude),
            x_ngt=(["index"], x_ngt),
            x_ngt_rounded=(["index"], x_ngt_rounded),
            y_ngt=(["index"], y_ngt),
            y_ngt_rounded=(["index"], y_ngt_rounded),
            tp_mm_radar=(["index"], tp_mm_radar),
            time_radar=(["index"], time_radar),
        ),
        coords=dict(index=index),
    )


@fixture()
def fake_dataset_add_fields():
    time = np.array(
        [
            "2017-01-01T00:00:05.000000000",
            "2017-01-02T00:05:00.000000000",
            "2017-01-03T00:29:00.000000000",
        ],
        dtype=np.datetime64,
    )
    time_h = np.array(
        [
            "2017-01-01T00:00:00.000000000",
            "2017-01-02T00:00:00.000000000",
            "2017-01-03T00:00:00.000000000",
        ],
        dtype=np.datetime64,
    )
    index = np.arange(3)
    return xarray.Dataset(
        data_vars=dict(time=(["index"], time), time_h=(["index"], time_h)),
        coords=dict(index=index),
    )


@fixture()
def fake_cumulative_dataset():
    return a2.dataset.load_dataset.load_tweets_dataset(
        DATA_FOLDER / "metoffice-c-band-rain-radar_uk_202002070005_10min_cumulative_tp_1km-composite.dat.gz.nc"
    )


@fixture()
def fake_tweets_stations():
    created_at = np.array(
        [
            "2018-07-02T04:25:00.000000000",
            "2018-08-15T07:10:00.000000000",
            "2018-08-15T06:30:00.000000000",
            "2018-08-15T10:05:00.000000000",
            "2018-08-15T20:05:00.000000000",
        ],
        dtype=np.datetime64,
    )
    latitude = np.array([58.16, 58.17, 58.18, 58.19, 53.4], dtype=np.float64)
    longitude = np.array([-5.48, -5.46, -5.44, -5.42, -3.05], dtype=np.float64)
    index = np.arange(5)
    station_latitude = np.array([54.33, 54.33, 54.33, 54.33, 53.497], dtype=np.float64)
    station_longitude = np.array([-7.595, -7.595, -7.595, -7.595, -3.058], dtype=np.float64)
    station_distance_km = np.array(
        [446.1022758112755, 447.5258055236938, 448.9507964700915, 450.3772319127779, 10.808661986574972],
        dtype=np.float64,
    )
    station_tp_mm = np.array([np.nan, 4.6, 0.0, np.nan, 9.6], dtype=float)
    ds = xarray.Dataset(
        data_vars=dict(
            created_at=(["index"], created_at),
            time=(["index"], created_at),
            latitude=(["index"], latitude),
            longitude=(["index"], longitude),
            station_latitude=(["index"], station_latitude),
            station_longitude=(["index"], station_longitude),
            station_distance_km=(["index"], station_distance_km),
            station_tp_mm=(["index"], station_tp_mm),
        ),
        coords=dict(index=index),
    )
    ds = a2.dataset.utils_dataset.add_field(
        ds,
        variable="time_half",
    )
    return ds


@fixture()
def fake_tweets_no_stations(fake_tweets_stations):
    return fake_tweets_stations.drop_vars(
        ["station_latitude", "station_longitude", "station_distance_km", "station_tp_mm"]
    )


@fixture()
def fake_weather_stations():
    latitude = [54.33, 54.33, 53.497, 53.497]
    longitude = [-7.595, -7.595, -3.058, -3.058]
    ob_end_time = np.array(
        [
            "2018-08-15T07:00:00.000000000",
            "2018-08-15T08:00:00.000000000",
            "2018-08-15T11:00:00.000000000",
            "2018-08-15T21:00:00.000000000",
        ],
        dtype=np.datetime64,
    )
    prcp_amt = np.array([0.0, 4.6, 5.0, 9.6], dtype=float)
    return pd.DataFrame(data=dict(latitude=latitude, longitude=longitude, ob_end_time=ob_end_time, prcp_amt=prcp_amt))


@fixture()
def fake_tweets_json_data():
    data = [
        {
            "data": [
                {
                    "reply_settings": "everyone",
                    "geo": {
                        "coordinates": {"type": "Point", "coordinates": [-0.123, 51.489]},
                        "place_id": "fakeplaceid",
                    },
                    "author_id": "111",
                    "created_at": "2017-01-02T02:02:02.000Z",
                    "public_metrics": {
                        "retweet_count": 0,
                        "reply_count": 0,
                        "like_count": 0,
                        "quote_count": 0,
                        "impression_count": 0,
                    },
                    "conversation_id": "111",
                    "id": "111",
                    "edit_history_tweet_ids": ["111"],
                    "lang": "en",
                    "text": "hi, is it raining?",
                },
                {
                    "reply_settings": "everyone",
                    "author_id": "222",
                    "created_at": "2017-01-02T05:02:02.000Z",
                    "public_metrics": {
                        "retweet_count": 0,
                        "reply_count": 1,
                        "like_count": 12,
                        "quote_count": 0,
                        "impression_count": 0,
                    },
                    "conversation_id": "222",
                    "id": "222",
                    "geo": {"place_id": "fakeplaceid2"},
                    "edit_history_tweet_ids": ["222"],
                    "lang": "en",
                    "text": "is it raining?",
                },
            ],
            "meta": {"newest_id": "111", "oldest_id": "222", "result_count": 2, "next_token": "faketoken"},
        }
    ]
    return data


@fixture()
def fake_tweets_dataframe():
    data = {
        "reply_settings": ["everyone", "everyone"],
        "geo.coordinates.coordinates": [[-0.123, 51.489], pd.NA],
        "geo.coordinates.type": ["Point", pd.NA],
        "geo.place_id": ["fakeplaceid", "fakeplaceid2"],
        "author_id": ["111", "222"],
        "created_at": ["2017-01-02T02:02:02.000Z", "2017-01-02T05:02:02.000Z"],
        "public_metrics.retweet_count": [0, 0],
        "public_metrics.reply_count": [0, 1],
        "public_metrics.like_count": [0, 12],
        "public_metrics.quote_count": [0, 0],
        "public_metrics.impression_count": [0, 0],
        "conversation_id": ["111", "222"],
        "id": ["111", "222"],
        "edit_history_tweet_ids": [
            ["111"],
            ["222"],
        ],
        "lang": ["en", "en"],
        "text": ["hi, is it raining?", "is it raining?"],
    }
    return pd.DataFrame(data=data, index=np.arange(2))


@fixture()
def fake_tweets_json_filepath(tmp_path, fake_tweets_json_data):
    directory = tmp_path / "tweets_json/"
    directory.mkdir()
    filepath = directory / "fake_tweets.json"
    a2.utils.file_handling.json_dump(filepath, data=fake_tweets_json_data)
    return filepath


@fixture()
def fake_weather_station_dataframe():
    latitude = [54.33, 54.33, 54.33, 54.33, 54.33, 54.33, 54.33, 54.33, 54.33]
    longitude = [-7.595, -7.595, -7.595, -7.595, -7.595, -7.595, -7.595, -7.595, -7.595]
    index = pd.MultiIndex.from_arrays([latitude, longitude], names=["latitude", "longitude"])
    data = {
        "ob_end_time": np.array(
            [
                "2018-07-02 05:00:00",
                "2018-08-15 13:00:00",
                "2018-08-15 12:00:00",
                "2018-08-15 11:00:00",
                "2018-08-15 10:00:00",
                "2018-08-15 09:00:00",
                "2018-08-15 08:00:00",
                "2018-08-15 07:00:00",
                "2018-08-15 06:00:00",
            ],
            dtype=np.datetime64,
        ),
        "prcp_amt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "station_number": [82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0],
    }
    return pd.DataFrame(data=data, index=index)
