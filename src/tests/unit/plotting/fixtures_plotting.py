import pathlib

import a2.dataset.load_dataset
import distutils
import numpy as np
import pandas as pd
import xarray
from pytest_cases import fixture

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


def setup_directory(tmp_path, name_subdirectory):
    directory = tmp_path / name_subdirectory / "badc/"
    directory.mkdir(parents=True)
    folder_tarball_origin = DATA_FOLDER / "radar/badc"
    distutils.dir_util.copy_tree(folder_tarball_origin.__str__(), directory.__str__())
    return directory


@fixture()
def fake_prediction():
    prediction_probabilities = np.array(
        [
            0.7991671,
            0.87987778,
            0.96945542,
            0.08077779,
            0.91080405,
            0.19031019,
            0.65461306,
            0.73965592,
            0.55324229,
            0.05229064,
            0.75468454,
            0.27704755,
        ]
    )
    predictions = np.round(prediction_probabilities)
    truth = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    return truth, predictions, prediction_probabilities


@fixture()
def fake_prediction_dataset(fake_prediction):
    truth, predictions, prediction_probabilities = fake_prediction
    index = np.arange(len(truth))
    return xarray.Dataset(
        data_vars=dict(
            truth=(["index"], truth),
            predictions=(["index"], predictions),
            prediction_probabilities=(["index"], prediction_probabilities),
        ),
        coords=dict(index=index),
    )


@fixture()
def fake_prediction_certainties():
    return np.array(
        [
            [1.0, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 3.0],
            [1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan],
        ]
    )


@fixture()
def fake_classification_report():
    return {
        "not raining": {
            "precision": 0.25,
            "recall": 0.125,
            "f1-score": 0.16666666666666666,
            "support": 8,
        },
        "raining": {
            "precision": 0.125,
            "recall": 0.25,
            "f1-score": 0.16666666666666666,
            "support": 4,
        },
        "accuracy": 0.16666666666666666,
        "macro avg": {
            "precision": 0.1875,
            "recall": 0.1875,
            "f1-score": 0.16666666666666666,
            "support": 12,
        },
        "weighted avg": {
            "precision": 0.20833333333333334,
            "recall": 0.16666666666666666,
            "f1-score": 0.16666666666666666,
            "support": 12,
        },
    }


@fixture()
def fake_roc_rates():
    tpr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 0.75, 0.75, 1.0])
    fpr = np.array(
        [
            0.0,
            0.125,
            0.25,
            0.375,
            0.5,
            0.625,
            0.625,
            0.75,
            0.875,
            0.875,
            0.875,
            1.0,
            1.0,
        ]
    )
    return tpr, fpr


@fixture()
def test_gaussian_random_samples():
    return np.array(
        [
            0.71002236,
            0.94300828,
            0.86026197,
            0.77227397,
            0.67138715,
            0.93061088,
            0.84489104,
            0.86303883,
            0.84506427,
            0.82114455,
            0.72800416,
            0.72417037,
            0.8472057,
            0.51922188,
            0.75815384,
            0.76883033,
            0.83744374,
            0.75848778,
            0.67386742,
            0.8224193,
            0.82515845,
            0.81864363,
            0.68923002,
            0.80043705,
            0.81878476,
            0.83471344,
            0.63445122,
            0.64513854,
            0.77049124,
            0.7676375,
            0.88071731,
            0.62566055,
            0.8350904,
            0.7064904,
            0.89838212,
            0.72501736,
            0.93722991,
            0.74375508,
            0.76571236,
            0.76116484,
        ]
    )


@fixture()
def test_symlog_histogram_results():
    return np.array(
        [
            [-1.00000000e00, 0.00000000e00],
            [-7.94328235e-01, 0.00000000e00],
            [-6.30957344e-01, 0.00000000e00],
            [-5.01187234e-01, 0.00000000e00],
            [-3.98107171e-01, 0.00000000e00],
            [-3.16227766e-01, 0.00000000e00],
            [-2.51188643e-01, 0.00000000e00],
            [-1.99526231e-01, 0.00000000e00],
            [-1.58489319e-01, 0.00000000e00],
            [-1.25892541e-01, 0.00000000e00],
            [-1.00000000e-01, 0.00000000e00],
            [-8.00000000e-02, 0.00000000e00],
            [-6.00000000e-02, 0.00000000e00],
            [-4.00000000e-02, 0.00000000e00],
            [-2.00000000e-02, 0.00000000e00],
            [-1.38777878e-17, 0.00000000e00],
            [2.00000000e-02, 0.00000000e00],
            [4.00000000e-02, 0.00000000e00],
            [6.00000000e-02, 0.00000000e00],
            [8.00000000e-02, 0.00000000e00],
            [1.00000000e-01, 0.00000000e00],
            [1.29154967e-01, 0.00000000e00],
            [1.66810054e-01, 0.00000000e00],
            [2.15443469e-01, 0.00000000e00],
            [2.78255940e-01, 0.00000000e00],
            [3.59381366e-01, 0.00000000e00],
            [4.64158883e-01, 1.00000000e00],
            [5.99484250e-01, 2.00000000e01],
            [7.74263683e-01, 1.90000000e01],
        ]
    )


@fixture()
def test_log_histogram_results():
    return np.array(
        [
            [1.00000000e-02, 0.00000000e00],
            [1.27427499e-02, 0.00000000e00],
            [1.62377674e-02, 0.00000000e00],
            [2.06913808e-02, 0.00000000e00],
            [2.63665090e-02, 0.00000000e00],
            [3.35981829e-02, 0.00000000e00],
            [4.28133240e-02, 0.00000000e00],
            [5.45559478e-02, 0.00000000e00],
            [6.95192796e-02, 0.00000000e00],
            [8.85866790e-02, 0.00000000e00],
            [1.12883789e-01, 0.00000000e00],
            [1.43844989e-01, 0.00000000e00],
            [1.83298071e-01, 0.00000000e00],
            [2.33572147e-01, 0.00000000e00],
            [2.97635144e-01, 0.00000000e00],
            [3.79269019e-01, 0.00000000e00],
            [4.83293024e-01, 1.00000000e00],
            [6.15848211e-01, 2.00000000e01],
            [7.84759970e-01, 1.90000000e01],
        ]
    )


@fixture()
def test_symlog_linthresh_histogram_results():
    return np.array(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-0.8, 0.0],
            [-0.6, 0.0],
            [-0.4, 0.0],
            [-0.2, 0.0],
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 1.0],
            [0.6, 20.0],
            [0.8, 19.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )


@fixture()
def fake_dataset_tweets_weather_map():
    text = ["hi, is it raining?", "is it raining?"]
    source = ["A", "B"]
    author_id = [2.2e8, 1.1e8]
    id = ["111", "222"]
    created_at = [
        np.datetime64("2017-01-01T14:14:14.000000000"),
        np.datetime64("2017-01-05T03:03:03.000000000"),
    ]
    longitude = [-0.1, 0.2]
    latitude = [51.5, 51.4]
    index = np.arange(2)
    ds = xarray.Dataset(
        data_vars=dict(
            text=(["index"], text),
            source=(["index"], source),
            author_id=(["index"], author_id),
            id=(["index"], id),
            created_at=(["index"], created_at),
            longitude=(["index"], longitude),
            latitude=(["index"], latitude),
        ),
        coords=dict(index=index),
    )
    ds["time"] = (["index"], ds["created_at"].values)
    ds = a2.dataset.utils_dataset.add_field(ds, "time_half", coordinates=["index"])
    return ds


@fixture()
def fake_weather_maps():
    return np.array(
        [
            [0.00143794, 0.00067655, 0.00084515, 0.00047358],
            [0.00192012, 0.00157755, 0.00175582, 0.00101483],
            [0.00080005, 0.00060138, 0.00086126, 0.00076568],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


@fixture()
def fake_dataset_tweets_radar_stations_plotting(tmp_path, fake_weather_stations_plotting):
    setup_directory(tmp_path, "fake_dataset_tweets_radar_stations_plotting")
    created_at = np.array(
        [
            "2020-02-07T00:01:05.000000000",
            "2020-02-07T00:02:05.000000000",
            "2020-02-07T00:04:25.000000000",
            "2020-02-07T00:06:15.000000000",
        ],
        dtype=np.datetime64,
    )
    latitude = np.array([58.3, 56.17, 54.18, 58.19], dtype=np.float64)
    longitude = np.array([-5.65, -2.46, -6.44, -2.42], dtype=np.float64)
    index = np.array([0, 1, 2, 3], dtype=np.int64)
    ds = xarray.Dataset(
        data_vars=dict(
            created_at=(["index"], created_at),
            time=(["index"], created_at),
            latitude=(["index"], latitude),
            longitude=(["index"], longitude),
        ),
        coords=dict(index=index),
    )
    ds = a2.dataset.utils_dataset.add_field(ds, "time_half", coordinates=["index"])
    ds = a2.dataset.stations.add_station_precipitation(
        ds=ds,
        df_stations=fake_weather_stations_plotting,
        processes=1,
    )
    ds = a2.dataset.radar.assign_radar_to_tweets(
        ds,
        key_tweets=None,
        round_ngt_offset=500,
        round_ngt_decimal=-3,
        round_time_to_base=5,
        path_to_dapceda=tmp_path / "fake_dataset_tweets_radar_stations_plotting/",
        processes=1,
    )
    print(f'{tmp_path / "fake_dataset_tweets_radar_stations_plotting/"=}')
    return ds


@fixture()
def fake_weather_stations_plotting():
    latitude = [58.33, 57.33, 55.497, 56.497]
    longitude = [-4.595, -2.595, -5.058, -3.058]
    ob_end_time = np.array(
        [
            "2020-02-07T01:00:00.000000000",
            "2020-02-07T01:00:00.000000000",
            "2020-02-07T01:00:00.000000000",
            "2020-02-07T01:00:00.000000000",
        ],
        dtype=np.datetime64,
    )
    prcp_amt = np.array([1.2, 4.6, 5.0, 9.6], dtype=float)
    return pd.DataFrame(data=dict(latitude=latitude, longitude=longitude, ob_end_time=ob_end_time, prcp_amt=prcp_amt))


@fixture()
def fake_tweets_stations_plotting(fake_dataset_tweets_radar_stations_plotting):
    fake_dataset_tweets_radar_stations_plotting["created_at"] = (
        ["index"],
        pd.to_datetime(fake_dataset_tweets_radar_stations_plotting.created_at.values) + pd.Timedelta("30m"),
    )
    return fake_dataset_tweets_radar_stations_plotting


@fixture()
def fake_timeseries_data_raw():
    start = np.datetime64("2020-02-07T01:00:00.000000000")
    end = np.datetime64("2020-02-07T10:00:00.000000000")
    times = pd.date_range(start=start, end=end, periods=10)
    values = np.linspace(0, 10, 10) ** 1 / 2
    return times, values


@fixture()
def fake_timeseries_data_ds():
    start = np.datetime64("2020-02-07T01:00:00.000000000")
    end = np.datetime64("2020-02-07T10:00:00.000000000")
    times = pd.date_range(start=start, end=end, periods=10)
    values = np.linspace(0, 10, 10) ** 1 / 2
    return xarray.Dataset(coords=dict(times=times), data_vars=dict(values=(["times"], values)))


@fixture()
def fake_keywords():
    return [
        "🏔️",
        "☀️",
        "🌞",
        "⛅",
        "⛈️",
        "🌤️",
        "🌥️",
        "🌦️",
        "🌧️",
        "🌨️",
        "🌩️",
        "☔",
        "⛄",
        "blizzard",
        "cloudburst",
        "downpour",
        "drizzle",
    ]


@fixture()
def fake_text_keywords():
    return "I can see 🏔️ and ☀️ or 🌞 but a second 🌞 as well and all these: ⛅ ⛈️ 🌤️ 🌥️ 🌦️ 🌧️ 🌨️ 🌩️ ☔ ⛄ blizzard cloudburst downpour drizzle"  # noqa: 501
