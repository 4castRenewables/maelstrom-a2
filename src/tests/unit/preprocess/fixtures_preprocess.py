import pathlib

import numpy as np
import xarray
from pytest_cases import fixture

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


@fixture()
def fake_dataset_to_normalize_filter():
    text = ["hi, is it raining? @whatisthis", "is it raining?"]
    source = ["Instagram", "B"]
    author_id = [2.2e8, 1.1e8]
    bounding_box_area = [1, 120]
    id = ["111", "222"]
    index = np.arange(2)
    return xarray.Dataset(
        data_vars=dict(
            id=(["index"], id),
            text=(["index"], text),
            source=(["index"], source),
            author_id=(["index"], author_id),
            bounding_box_area=(["index"], bounding_box_area),
        ),
        coords=dict(index=index),
    )


@fixture()
def results_normalize_filter():
    text = ["hi, is it raining? @whatisthis", "is it raining?"]
    text_original = ["hi, is it raining? @whatisthis", "is it raining?"]
    text_normalized = ["hi , is it raining ?", "is it raining ?"]
    source = ["Instagram", "B"]
    author_id = [2.2e8, 1.1e8]
    bounding_box_area = [1, 120]
    id = ["111", "222"]
    index = np.arange(2)
    return xarray.Dataset(
        data_vars=dict(
            id=(["index"], id),
            author_id=(["index"], author_id),
            source=(["index"], source),
            text=(["index"], text),
            text_original=(["index"], text_original),
            text_normalized=(["index"], text_normalized),
            bounding_box_area=(["index"], bounding_box_area),
        ),
        coords=dict(index=index),
    ).sel(index=slice(0))


@fixture()
def results_normalize_filter_no_punctuation():
    text = ["hi, is it raining? @whatisthis", "is it raining?"]
    text_original = ["hi, is it raining? @whatisthis", "is it raining?"]
    text_normalized = ["hi is it raining", "is it raining"]
    source = ["Instagram", "B"]
    author_id = [2.2e8, 1.1e8]
    bounding_box_area = [1, 120]
    id = ["111", "222"]
    index = np.arange(2)
    return xarray.Dataset(
        data_vars=dict(
            id=(["index"], id),
            author_id=(["index"], author_id),
            source=(["index"], source),
            text=(["index"], text),
            text_original=(["index"], text_original),
            text_normalized=(["index"], text_normalized),
            bounding_box_area=(["index"], bounding_box_area),
        ),
        coords=dict(index=index),
    ).sel(index=slice(0))


@fixture()
def dataset_less_optimal_coverage():
    text_normalized = ["hi , is it raining NOTPARTOFVOCAB", "is it raining ?"]
    index = np.arange(2)
    return xarray.Dataset(
        data_vars=dict(
            text_normalized=(["index"], text_normalized),
        ),
        coords=dict(index=index),
    )
