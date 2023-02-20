import pathlib

import numpy as np
import pytest
import xarray
from pytest_cases import fixture

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


@fixture()
def fake_dataset_coordinates():
    coordinates = np.array(["", "", "[-4.2, 42]"], dtype=object)
    index = np.arange(3)
    return xarray.Dataset(
        data_vars=dict(coordinates=(["index"], coordinates)),
        coords=dict(index=index),
    )


@fixture()
def fake_add_locations_dataframe(fake_add_locations_dataframe_solution):
    return fake_add_locations_dataframe_solution.drop_vars(
        [
            "coordinates_estimated",
            "centroid",
            "place_type",
            "bounding_box",
            "full_name",
        ]
    )


@fixture()
def fake_add_locations_dataframe_solution():
    place_id = np.array(
        [
            "fakeid1",
            "fakeid2",
            "fakeid3",
            "fakeid4",
        ],
        dtype=np.str_,
    )
    coordinates = np.array(["[-4.2, 4.2]", "nan", "nan", "[-42, 42]"], dtype=np.str_)
    index = np.array([0, 1, 2, 3], dtype=np.int64)
    coordinates_estimated = np.array([False, True, True, False], dtype=np.bool_)
    centroid = np.array(
        [
            np.nan,
            "[-4.2, -4.2]",
            "[4.2, 4.2]",
            np.nan,
        ],
        dtype=np.object_,
    )
    place_type = np.array([np.nan, "city", "city", np.nan], dtype=np.object_)
    bounding_box = np.array(
        [
            np.nan,
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-4.2, 42],
                        [-4.2, -42],
                        [4.2, 42],
                        [4.2, -42],
                        [-4.2, 42],
                    ]
                ],
            },
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-42, 4.2],
                        [-42, -4.2],
                        [42, 4.2],
                        [42, -4.2],
                        [-42, 4.2],
                    ]
                ],
            },
            np.nan,
        ],
        dtype=np.object_,
    )
    full_name = np.array([np.nan, "fake city, fake country", "fake city 2, fake country 2", np.nan], dtype=np.object_)
    return xarray.Dataset(
        data_vars=dict(
            place_id=(["index"], place_id),
            coordinates=(["index"], coordinates),
            coordinates_estimated=(["index"], coordinates_estimated),
            centroid=(["index"], centroid),
            place_type=(["index"], place_type),
            bounding_box=(["index"], bounding_box),
            full_name=(["index"], full_name),
        ),
        coords=dict(index=index),
    )


@fixture()
def fake_csv_content_tweets_download():
    return [
        "# info for query for json file tweets.json",
        "# keyword: sun has:geo -is:retweet (lang:en) place_country:GB -is:nullcast -from:3446146816",
        "# dates:2020-01-1T10:00:00.000Z-->2020-01-1T12:00:00.000Z",
    ]


@fixture()
def fake_tweets_json_data_no_next_token(fake_tweets_json_data):
    fake_tweets_json_data[0]["meta"].pop("next_token")
    return fake_tweets_json_data


@fixture()
def fake_tweet_location_added_dataframe():
    place_ids = np.array(["testid", "testiddoesntexist"])
    coordinates = np.array([np.nan, np.nan])
    centroid = np.array(["[0, 0]", np.nan], dtype=object)
    place_type = np.array(["city", np.nan], dtype=object)
    full_name = np.array(["Fake city, fake country", np.nan], dtype=object)
    coordinates_estimated = np.array([True, False])
    bounding_box = np.array(
        [{"type": "Polygon", "coordinates": [[[-42, 42], [42, 42], [-42, -42], [42, -42], [-42, 42]]]}, np.nan]
    )
    index = np.arange(len(place_ids))
    return xarray.Dataset(
        data_vars={
            "place_id": (["index"], place_ids),
            "coordinates": (["index"], coordinates),
            "centroid": (["index"], centroid),
            "place_type": (["index"], place_type),
            "full_name": (["index"], full_name),
            "coordinates_estimated": (["index"], coordinates_estimated),
            "bounding_box": (["index"], bounding_box),
        },
        coords=dict(index=index),
    )


@fixture()
def fake_tweet_location_to_be_added_dataframe(fake_tweet_location_added_dataframe):
    # fake_tweet_location_added_dataframe['coordinates'] = (['index'], np.array([np.nan]))
    return fake_tweet_location_added_dataframe.drop_vars(
        ["centroid", "place_type", "full_name", "coordinates_estimated", "bounding_box"]
    )


class FakeTweepyApi:
    def __init__(self, auth, wait_on_rate_limit=True, parser=None):
        pass

    def geo_id(self, place_id):
        if place_id == "testid":
            return {
                "id": "testid",
                "name": "Fake city",
                "full_name": "Fake city, fake country",
                "country": "Fake Country",
                "country_code": "FC",
                "url": "https://api.twitter.com/1.1/geo/id/testid.json",
                "place_type": "city",
                "attributes": {"geotagCount": "1"},
                "bounding_box": {
                    "type": "Polygon",
                    "coordinates": [[[-42, 42], [42, 42], [-42, -42], [42, -42], [-42, 42]]],
                },
                "centroid": [-0, 0],
                "contained_within": [
                    {
                        "id": "fakeidcounty",
                        "name": "Central Fake County",
                        "full_name": "Central Fake County, Fake Country",
                        "country": "Fake Country",
                        "country_code": "FC",
                        "url": "https://api.twitter.com/1.1/geo/id/fakeidcounty.json",
                        "place_type": "admin",
                        "attributes": {},
                        "bounding_box": {
                            "type": "Polygon",
                            "coordinates": [[[-84, 84], [-84, -84], [84, 84], [84, -84], [-84, 84]]],
                        },
                        "centroid": [0, 0],
                    }
                ],
                "polylines": [],
                "geometry": None,
            }
        raise ValueError(f"{place_id=} not found!")


@pytest.fixture()
def mock_tweepy_auth_api_geoid(mocker):
    mocker.patch("tweepy.OAuth1UserHandler", return_value="auth")
    mocker.patch("tweepy.API", return_value=FakeTweepyApi(None, None, None))
