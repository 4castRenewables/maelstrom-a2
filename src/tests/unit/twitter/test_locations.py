import pathlib

import a2.dataset.utils_dataset
import a2.twitter.locations
import a2.utils.testing
import numpy as np
import pytest
from pytest_cases import parametrize

FILE_LOC = pathlib.Path(__file__).parent


@parametrize(
    "token",
    [
        ("TWITTER_CONSUMER_KEY"),
        ("TWITTER_ACCESS_TOKEN"),
    ],
)
def test__auth(monkeypatch, token):
    monkeypatch.delenv(token)
    with pytest.raises(type(ValueError())):
        assert a2.twitter.locations._get_tweepy_api() == ValueError()


def test_convert_coordinates_to_lat_long(fake_dataset_coordinates):
    ds = a2.twitter.locations.convert_coordinates_to_lat_long(
        fake_dataset_coordinates,
        key_coordinates="coordinates",
        prefix_lat_long="",
        overwrite=False,
    )
    for x, y in zip(ds.latitude.values, np.array([np.NaN, np.NaN, 42])):
        assert x == y if ~np.isnan(x) else np.isnan(x) == np.isnan(y)
    for x, y in zip(ds.longitude.values, np.array([np.NaN, np.NaN, -4.2])):
        assert x == y if ~np.isnan(x) else np.isnan(x) == np.isnan(y)


def test_convert_coordinates_to_lat_long_errors(fake_dataset_coordinates):
    fake_dataset_coordinates["longitude"] = (
        ["index"],
        [np.nan, np.nan, np.nan],
    )
    fake_dataset_coordinates["latitude"] = (["index"], [np.nan, np.nan, np.nan])
    with pytest.raises(type(RuntimeError())):
        a2.twitter.locations.convert_coordinates_to_lat_long(
            fake_dataset_coordinates,
            key_coordinates="coordinates",
            prefix_lat_long="",
            overwrite=False,
        )


def test_convert_coordinates_to_lat_long_errors_2(fake_dataset_coordinates):
    fake_dataset_coordinates = fake_dataset_coordinates.drop_vars(["coordinates"])
    with pytest.raises(type(KeyError())):
        a2.twitter.locations.convert_coordinates_to_lat_long(
            fake_dataset_coordinates,
            key_coordinates="coordinates",
            prefix_lat_long="",
            overwrite=False,
        )


@parametrize(
    "bounding_box, expected_area_bounding_box",
    [
        (
            {"type": "Polygon", "coordinates": [[[-2.5, 52], [-2, 52.5], [-2.5, 52.5], [-2.0, 52.0], [-2.25, 52.25]]]},
            466.4259784995248,
        ),
    ],
)
def test__compute_bounding_box_area(bounding_box, expected_area_bounding_box):
    area_bounding_box = a2.twitter.locations._compute_bounding_box_area(bounding_box=bounding_box)
    assert area_bounding_box == expected_area_bounding_box
