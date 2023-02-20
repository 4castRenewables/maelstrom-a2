import json
import logging
import os
import pathlib
import time
import typing as t

import a2.dataset.utils_dataset
import a2.utils.file_handling
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import tqdm
import tweepy
import xarray


def _get_tweepy_api(wait_on_rate_limit: bool = True) -> tweepy.API:
    """
    returns Twitter api object from tweepy

    requires environment variables TWITTER_CONSUMER_KEY and TWITTER_ACCESS_TOKEN
    to work
    Parameters:
    ----------
    wait_on_rate_limit: api waits when too many requests send until
                        resend possible

    Returns
    -------
    tweepy api object
    """
    twitter_consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
    if twitter_consumer_key is None:
        raise ValueError("tweepy api token: TWITTER_CONSUMER_KEY not found" "as environment variable!")
    twitter_access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    if twitter_access_token is None:
        raise ValueError("tweepy api token: TWITTER_ACCESS_TOKEN not found" "as environment variable!")

    auth = tweepy.OAuth1UserHandler(
        twitter_consumer_key,
        twitter_access_token,
    )
    api = tweepy.API(
        auth,
        wait_on_rate_limit=wait_on_rate_limit,
        parser=tweepy.parsers.JSONParser(),
    )
    return api


def _append_not_found(place_id: str, filename: t.Union[str, pathlib.Path]) -> None:
    """
    append place_id's to csv file
    """
    if not a2.utils.file_handling.file_exists_and_not_empty(filename):
        logging.info(f"creating new csv file: {filename}")
        a2.utils.file_handling.csv_create(
            filename,
            ["# list of place_id's that were not found by twitter api\n"],
            [place_id],
            check_is_file_or_empty=True,
        )
    else:
        a2.utils.file_handling.csv_append(filename, [place_id])


def save_location_to_file(
    place_ids: t.Union[t.Sequence[str], set[str]],
    filename_location: t.Union[str, pathlib.Path] = "./locations.json",
    filename_location_not_found: t.Union[str, pathlib.Path] = "./locations_not_found.csv",
) -> None:
    """
    saves and queries json responses of information on place_id's to file

    Parameters:
    ----------
    place_ids: place_id's to query
    filename_location: name of file where json responses saved
    filename_location_not_found: adds place_id's to this csv file
                                 which cannot be found by twitter api

    Returns
    -------
    """
    api = _get_tweepy_api(wait_on_rate_limit=True)
    queried = 0

    # remove place_id's that have already been queried
    place_ids = set(place_ids)
    if os.path.isfile(filename_location):
        df = pd.read_json(filename_location)
        place_ids = set(place_ids) - set(df["id"].values)
        del df
    if os.path.isfile(filename_location_not_found):
        df = pd.read_csv(
            filename_location_not_found,
            skiprows=1,
            skip_blank_lines=False,
            names=["id"],
        )
        place_ids = set(place_ids) - set(df["id"].values)
        del df
    if len(place_ids):
        logging.info(f"Need to query {len(place_ids)} place_id's ...")

    for p_id in tqdm.tqdm(place_ids):
        if queried >= 74:
            time.sleep(60 * 15 + 15)
            queried = 0
        if os.path.isfile(filename_location):
            with open(filename_location) as f:
                data = json.load(f)
        queried += 1
        try:
            location = api.geo_id(place_id=p_id)
            print(f"{location=}")
        except Exception as e:
            _append_not_found(p_id, filename_location_not_found)
            logging.info(f"couldn't query: {p_id}")
            logging.info(e)
            continue
        if not os.path.isfile(filename_location):
            data = [location]
        else:
            data.append(location)
        a2.utils.file_handling.json_dump(filename_location, data, log_if_new_file=True)


def convert_coordinates_to_lat_long(
    ds: xarray.Dataset,
    key_coordinates: str = "geo.coordinates.coordinates",
    prefix_lat_long: str = "",
    overwrite: bool = False,
):
    """
    converts coordinates in dataset to latitude and longitude
    and adds them as new columns

    Parameters:
    ----------
    ds: dataset with coordinates specified
    key_coordinates: key name of coordinates (in string format)
    prefix_lat_long: prefix before latitude and longitude column names
    overwrite: allow overwriting of latitude and longitude column

    Returns
    -------
    dataset with longitude/latitude
    """
    if not overwrite and (f"{prefix_lat_long}longitude" in ds or f"{prefix_lat_long}latitude" in ds):
        raise RuntimeError("Fields latitude and/or longitude already present!")
    if key_coordinates not in ds:
        raise KeyError(f"coordinate_name: {key_coordinates} not found in dataframe!")

    if f"{prefix_lat_long}longitude" not in ds or overwrite:
        ds = a2.dataset.utils_dataset.initialize_variable(ds, f"{prefix_lat_long}longitude", ["index"], values=np.nan)
    if f"{prefix_lat_long}latitude" not in ds or overwrite:
        ds = a2.dataset.utils_dataset.initialize_variable(ds, f"{prefix_lat_long}latitude", ["index"], values=np.nan)
    if not a2.dataset.utils_dataset.is_same_type_data_array(ds, key_coordinates, which_type=str):
        ds[key_coordinates] = (
            ["index"],
            a2.dataset.utils_dataset.array_elements_to_str(ds[key_coordinates].values),
        )
    if a2.dataset.utils_dataset.is_same_type_data_array(ds, key_coordinates, which_type=str):
        s = ds[key_coordinates].str.extract(r"^\[([0-9.e-]+) ?,", dim=None)
        mask = s == ""
        s[mask] = np.nan
        ds[f"{prefix_lat_long}longitude"] = s.astype(float)
        s = ds[key_coordinates].str.extract(r" ?, ?([0-9.e-]+)\]$", dim=None)
        mask = s == ""
        s[mask] = np.nan
        ds[f"{prefix_lat_long}latitude"] = s.astype(float)

    else:
        raise Exception(f"do not understand datatype of field {key_coordinates} to convert!")

    return ds


def add_locations(
    ds: xarray.Dataset,
    filename_location: t.Union[str, pathlib.Path] = "./locations.json",
    filename_location_not_found: t.Union[str, pathlib.Path] = "./locations_not_found.csv",
    download: bool = True,
    key_place_id: str = "geo.place_id",
    key_coordinates: str = "geo.coordinates.coordinates",
) -> xarray.Dataset:
    """
    Adds location information to dataset based on place_id

    Information on places queried from twitter api/loaded from file if
    available. Add fields to dataset describing place_type,
    bounding_box, full_name and marks tweets if info was added in field
    coordinates_estimated. Centroid of inner boundary box is used as
    coordinate of location.
    Note, place_id's can correspond to arbitrary large entities.
    Parameters:
    ----------
    ds: dataset with place_id's specified
    filename_location: file path to json info on locations
    filename_location_not_found: file path to csv file with place_id's
                                 that couldn't be found in twitter query

    Returns
    -------
    dataset with location information
    """
    if download:
        save_location_to_file(
            list(ds[key_place_id].values),
            filename_location=filename_location,
            filename_location_not_found=filename_location_not_found,
        )

    ds_loc = pd.read_json(filename_location).to_xarray()
    ds = a2.dataset.utils_dataset.initialize_variable(ds, "coordinates_estimated", values=0, dtype=bool)
    ds = a2.dataset.utils_dataset.initialize_variable(ds, "centroid", dtype=object)
    ds = a2.dataset.utils_dataset.initialize_variable(ds, "place_type", dtype=object)
    ds = a2.dataset.utils_dataset.initialize_variable(ds, "bounding_box", dtype=object)
    ds = a2.dataset.utils_dataset.initialize_variable(ds, "full_name", dtype=object)

    tagged_found = 0
    places_in_dataset = set(ds[key_place_id].values)
    places_in_file = set(ds_loc.id.values)

    place_not_found = places_in_dataset - places_in_file
    places_in_file_array = np.array(list(places_in_file))
    places_in_dataset_array = np.array(list(places_in_dataset))
    logging.info(f"{len(place_not_found)} places not found in file, skipping those...")
    places_to_check = places_in_file_array[np.in1d(places_in_file_array, places_in_dataset_array)]

    for place_id in tqdm.tqdm(places_to_check):
        mask = np.logical_and(
            ds[key_place_id] == place_id,
            a2.dataset.utils_dataset.is_nan(ds, key_coordinates, dims=("index",)),
        )
        # only add location to places that do not already have autotracking
        # enabled, should be redundant however
        tagged_found += np.sum(mask.values)
        mask_loc = ds_loc.id == place_id
        if np.sum(mask_loc.values) != 1:
            raise ValueError(
                f"place_id {place_id} found multiple/zero times in location"
                "json! mask_loc.values: {mask_loc.values}, "
                f"np.sum(mask_loc.values): {np.sum(mask_loc.values)}"
            )
        ds["centroid"].loc[mask] = str(ds_loc["centroid"].loc[mask_loc].values[0])
        ds["place_type"].loc[mask] = ds_loc["place_type"].loc[mask_loc].values[0]
        ds["bounding_box"].loc[mask] = ds_loc["bounding_box"].loc[mask_loc].values[0]
        ds["full_name"].loc[mask] = ds_loc["full_name"].loc[mask_loc].values[0]
        ds["coordinates_estimated"].loc[mask] = True
    logging.info(f"found locations for {tagged_found} tweets")
    return ds


def _compute_bounding_box_area(bounding_box):
    """
    see compute_area_bounding_box
    """
    if hasattr(bounding_box, "item") and callable(getattr(bounding_box, "item")):
        bounding_box = bounding_box.item()
    try:
        if bounding_box == " " or pd.isnull(bounding_box):
            return -1
        else:
            lon, lat = zip(*bounding_box["coordinates"][0])
            pa = pyproj.Proj(f"+proj=aea +lat_1={lat[0]} +lat_2={lat[2]}" f" +lon_0={lon[0] + (lon[1] - lon[0]) / 2.}")
            x, y = pa(lon, lat)
            cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
            return shapely.geometry.shape(cop).area / 1e6
    except Exception as e:
        raise Exception("Couldn't compute bounding box area of" f"bounding_box: {bounding_box}") from e


def compute_area_bounding_box(bounding_boxes: np.ndarray) -> list:
    """
    computes area of bounding box in km^2 (parallelized)

    an equal area projection is used to convert the coordinates
    (https://proj.org/operations/projections/aea.html)
    and then the area covered by the bounding box is computed.
    if no bounding box is given returns 0.
    if bounding box declaration is incorrect returns -1.
    Assumes that bounding boxes are provided in the form of dictionaries.
    Parameters:
    ----------
    bounding_boxes: array with bounding boxes

    Returns
    -------
    array with area of corresponding bounding box
    """
    bounding_box_area = a2.utils.utils.parallelize(_compute_bounding_box_area, bounding_boxes, single_arg=True)
    return bounding_box_area
