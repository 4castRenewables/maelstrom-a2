from contextlib import nullcontext as doesnotraise

import a2.utils.testing
import pandas as pd
import pytest
import pytest_cases
import xarray


@pytest_cases.parametrize(
    "drop_variable, add",
    [
        ("time", False),
        (None, True),
    ],
)
def test_assert_equal_dataset(fake_dataset_add_fields, drop_variable, add):
    if drop_variable is not None:
        ds_variant = fake_dataset_add_fields.drop_vars(drop_variable)
    if add:
        ds_variant = fake_dataset_add_fields.copy()
        ds_variant["time"] = (
            ["index"],
            fake_dataset_add_fields.time.values + pd.Timedelta("30min"),
        )
    with pytest.raises(AssertionError):
        xarray.testing.assert_equal(ds_variant, fake_dataset_add_fields)


@pytest_cases.parametrize(
    "json1, json2, expected",
    [
        ({"a": "time"}, {"a": "time"}, True),
        ({"a": "time"}, '["a", "time"]', AssertionError()),
        ({"a": "time", "b": "2"}, {"a": "time"}, AssertionError()),
        (
            {"a": "time", "s": {"s": [[1, 1], [2, 2], [3, 5]]}, "b": "2"},
            {"a": "time", "s": {"s": [1, 2, 3]}, "b": "2"},
            AssertionError(),
        ),
    ],
)
def test_assert_equal_jsons(json1, json2, expected):
    with pytest.raises(type(expected)) if isinstance(expected, Exception) else doesnotraise():
        assert a2.utils.testing.json_equal(json1, json2)


def test_check_internet_conn():
    status = a2.utils.testing.check_internet_connection()
    assert status is True


def test_print_debug(fake_data_to_print):
    io_capture = a2.utils.testing.IOCapture()
    a2.utils.testing.print_debug(fake_data_to_print)
    printed = io_capture.return_capture_stop()
    assert (
        printed == "np.min(x)=-42.0\nnp.max(x)=42.0\nnp.mean(x)=0.0\nnp.isnan(x).sum()=0\nx[:10]=array([-42.,  42.])\n"
    )
