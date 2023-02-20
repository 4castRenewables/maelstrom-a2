import pathlib

import numpy as np
import xarray
from pytest_cases import fixture

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


@fixture()
def fake_dataset_training():
    text_normalized = [
        "hi , is it raining ?",
        "is it raining ?",
        "it is sunny",
        "it is rainy",
        "it is sunny",
        "it is rainy",
        "it is sunny",
        "it is rainy",
    ]
    raining = [1, 0, 0, 1, 0, 1, 0, 1]
    index = np.arange(8)
    return xarray.Dataset(
        data_vars=dict(
            text_normalized=(["index"], text_normalized),
            raining=(["index"], raining),
        ),
        coords=dict(index=index),
    )
