import dataclasses
import functools
import logging
import typing as t

import a2.dataset.tweets
import a2.utils
import geopy.distance
import numpy as np
import pandas as pd
import xarray


@dataclasses.dataclass(frozen=True)
class KeyStations:
    """Dataclass storing key convention for weather station datasets"""

    latitude = "latitude"
    longitude = "longitude"
    time = "ob_end_time"
    time_shifted = "ob_end_time_shifted"
    precipitation = "prcp_amt"


def distance_between_coordinates_in_km(
    latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float
) -> float:
    """Computes the the distance between two pairs of long/lat"""
    return geopy.distance.geodesic((latitude_1, longitude_1), (latitude_2, longitude_2)).km


def _find_closest_coordinate_and_distance(
    tweet_latitude: float,
    tweet_longitude: float,
    weather_station_latitudes: np.ndarray,
    weather_station_longitudes: np.ndarray,
):
    """
    Find the closest weather station to a Tweet's long/lat,
    returns the weather stations coordinates and distance

    Parameters:
    ----------
    tweet_latitude: Latitude of the Tweet
    tweet_longitude: Longitude of the Tweet
    weather_station_latitudes: Latitudes of all weather stations
    weather_station_longitudes: Longitudes of all weather stations

    Returns
    (latitude, longitude) of closest weather station, distance to closest weather station
    -------
    """
    distances = np.array(
        [
            distance_between_coordinates_in_km(tweet_latitude, tweet_longitude, lat, long)
            for lat, long in zip(weather_station_latitudes, weather_station_longitudes)
        ]
    )
    index_min = np.argmin(distances)
    latitude = weather_station_latitudes[index_min]
    longitude = weather_station_longitudes[index_min]
    distance = distances[index_min]
    return (latitude, longitude), distance


def find_closest_coordinate_and_distance(
    tweet_latitudes: np.ndarray,
    tweet_longitudes: np.ndarray,
    weather_station_latitudes: np.ndarray,
    weather_station_longitudes: np.ndarray,
    processes: int = -1,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the closest weather station to a list of long/lat pairs of Tweets,
    returns the weather stations coordinates and distance
    Parallelizes `_find_closest_coordinate_and_distance`

    Parameters:
    ----------
    tweet_latitudes: Latitudes of Tweets
    tweet_longitudes: Longitudes of Tweets
    weather_station_latitudes: Latitudes of all weather stations
    weather_station_longitudes: Longitudes of all weather stations
    weather_station_longitudes: Number of tasks run in parallel

    Returns
    latitudes, longitudes, distances of/to closest weather stations
    -------
    """
    function = functools.partial(
        _find_closest_coordinate_and_distance,
        weather_station_latitudes=weather_station_latitudes,
        weather_station_longitudes=weather_station_longitudes,
    )
    results = a2.utils.utils.parallelize(
        function, args_zipped=zip(tweet_latitudes, tweet_longitudes), processes=processes
    )
    latitudes = []
    longitudes = []
    distances = []
    for x in results:
        latitude_longitude, dist = x
        latitudes.append(latitude_longitude[0])
        longitudes.append(latitude_longitude[1])
        distances.append(dist)
    return np.array(latitudes), np.array(longitudes), np.array(distances)


def unique_coordinates(df: pd.DataFrame, coordinates: list = None):
    """Finds unique coordinate pairs"""
    if coordinates is None:
        coordinates = ["latitude", "longitude"]
    return np.unique(np.array([df[x].values for x in coordinates]), axis=1)


def add_station_precipitation(
    ds: xarray.Dataset,
    df_stations: pd.DataFrame,
    key_tweets=None,
    key_stations=None,
    station_time_offset: str = "30m",
    coords: t.Optional[list] = None,
    processes: int = -1,
):
    """
    Add precipitation values from nearest weather station to Tweet's location

    Parameters:
    ----------
    ds: Dataset of Tweets
    df_stations: Dataframe with weather station data
    key_tweets: Dataclass storing key convention for tweets datasets
    key_stations: Dataclass storing key convention for weather station datasets
    station_time_offset: Time stamp of weather station measurement reduced by this amount
    coords: Coordinates of ds
    processes: Number of closest station found in parallel

    Returns
    ds with added weather station precipitation
    -------
    """
    if key_stations is None:
        key_stations = KeyStations()
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    df_stations = df_stations.copy()
    if coords is None:
        coords = list(ds.coords.keys())
    time_offset, unit_offset = a2.utils.utils.str_to_delta_time(station_time_offset)
    df_stations[key_stations.time_shifted] = pd.to_datetime(df_stations[key_stations.time]) - np.timedelta64(
        int(time_offset), unit_offset
    )  # type: ignore

    _assign_coordinates_of_nearest_station_to_tweets(ds, df_stations, key_tweets, coords, processes)

    ds[key_tweets.station_tp] = (
        coords,
        np.full_like(ds[a2.dataset.utils_dataset.get_variable_name_first(ds)].values, np.nan, dtype=float),
    )
    n_time_not_found_total = 0

    for latitude, longitude in zip(*unique_coordinates(df_stations)):
        mask_ds_station_location = (ds[key_tweets.latitude_stations] == latitude) & (
            ds[key_tweets.longitude_stations] == longitude
        )
        df_single_station = _get_station_from_lat_long(df_stations, latitude, longitude)
        times_of_tweets = ds[key_tweets.time_stations].loc[mask_ds_station_location].values
        if np.shape(times_of_tweets)[0] == 0:
            continue
        n_time_not_found, n_times_stations = _initialize_debug_counters()
        for time in times_of_tweets:
            n_times_stations += 1
            tp_match = df_single_station[key_stations.precipitation][
                df_single_station[key_stations.time_shifted] == time
            ]
            if tp_match.shape[0] > 1:
                logging.warning(
                    f"Found more than one entry ({tp_match}) for rain at weather station "
                    f"at time {time}: {tp_match=} with times "
                    f"{df_single_station[key_stations.time_shifted][df_single_station[key_stations.time_shifted] == time]}"  # noqa
                )
            elif tp_match.shape[0] == 0:
                _increment_debug_counters(n_time_not_found_total, n_time_not_found)
                continue
            ds[key_tweets.station_tp].loc[
                mask_ds_station_location & (ds[key_tweets.time_stations] == time)
            ] = tp_match.values[0]
        if n_time_not_found > 0:
            logging.info(f"Couldn't find {n_time_not_found}/{n_times_stations} for station.")
    logging.info(f"Couldn't find {n_time_not_found_total}/{ds.index.shape[0]} time values in total")
    return ds


def _assign_coordinates_of_nearest_station_to_tweets(ds, df_stations, key_tweets, coords, processes):
    tweet_latitudes = ds[key_tweets.latitude].values
    tweet_longitudes = ds[key_tweets.longitude].values

    weather_station_coordinates = unique_coordinates(df_stations)

    weather_station_latitudes = weather_station_coordinates[0, :]
    weather_station_longitudes = weather_station_coordinates[1, :]
    latitudes, longitudes, distances = find_closest_coordinate_and_distance(
        tweet_latitudes,
        tweet_longitudes,
        weather_station_latitudes,
        weather_station_longitudes,
        processes=processes,
    )
    ds[key_tweets.latitude_stations] = (coords, latitudes)
    ds[key_tweets.longitude_stations] = (coords, longitudes)
    ds[key_tweets.distance_stations] = (coords, distances)


def _get_station_from_lat_long(df_stations, latitude, longitude):
    mask_df_station_location = (df_stations["latitude"] == latitude) & (df_stations["longitude"] == longitude)
    df_single_station = df_stations.loc[mask_df_station_location]
    return df_single_station


def _increment_debug_counters(n_time_not_found_total, n_time_not_found):
    n_time_not_found_total += 1
    n_time_not_found += 1


def _initialize_debug_counters():
    n_time_not_found = 0
    n_times_stations = 0
    return n_time_not_found, n_times_stations


def add_station_number(
    df_stations: pd.DataFrame,
    key_longitude: str = "longitude",
    key_latitude: str = "latitude",
    key_station_number: str = "station_number",
):
    """Number weather stations based on unique longitude/latitude pair
    Add number as new column to df_stations"""
    station_number = 0
    df_stations[key_station_number] = np.full_like(df_stations["longitude"].values, np.nan)
    for latitude, longitude in zip(*unique_coordinates(df_stations)):
        mask_df_station_location = (df_stations[key_latitude] == latitude) & (df_stations[key_longitude] == longitude)
        df_stations.loc[mask_df_station_location, key_station_number] = station_number
        station_number += 1
    return df_stations


def get_counts_station(df_stations: pd.DataFrame, key_station_number: str = "station_number"):
    """Return station number and respective number of occurences"""
    value_counts = df_stations[key_station_number].value_counts()
    station_numbers = value_counts.index.values
    counts = value_counts.values
    return station_numbers, counts


def get_time_series_from_station_number(
    df_stations: pd.DataFrame,
    station_number: int,
    key_time: str = "ob_end_time",
    key_station_number: str = "station_number",
    key_tp: str = "prcp_amt",
):
    """Returns precipitation time series for station based on station number"""
    mask = df_stations[key_station_number] == station_number
    time, tp = df_stations[key_time].loc[mask].values, df_stations[key_tp].loc[mask].values
    mask_sort = np.argsort(time)  # type: ignore
    return time[mask_sort], tp[mask_sort]


def get_dataframe_from_station_number(
    df_stations: pd.DataFrame, station_number: int, key_station_number: str = "station_number"
):
    """Returns dataframe containing only data for specific station number"""
    mask = df_stations[key_station_number] == station_number
    return df_stations[mask]
