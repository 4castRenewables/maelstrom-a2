import pathlib

import numpy as np
from pytest_cases import fixture

FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"


@fixture()
def vectors_for_cross_product():
    return np.arange(27).reshape((3, 3, 3)), np.arange(27).reshape((3, 3, 3))


@fixture()
def results_cross_product():
    return np.zeros(27).reshape((3, 3, 3))


@fixture()
def fake_data_to_print():
    return np.array([-42, 42], dtype=float)
