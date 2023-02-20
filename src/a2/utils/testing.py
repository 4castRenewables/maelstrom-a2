import io
import pathlib
import sys
import typing as t

import a2.dataset
import a2.utils.file_handling
import numpy as np
import pandas as pd
import urllib3
import xarray


class IOCapture:
    """
    Used to capture terminal output as string.

    Usually only used for testing.
    """

    def __init__(self):
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output

    def stop(self) -> None:
        """
        Resets redirect
        """
        sys.stdout = sys.__stdout__

    def return_capture_stop(self) -> str:
        """
        Resets redirect and returns captured value as string
        """
        self.stop()
        return self.captured_output.getvalue()


def ordered(obj: object):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def json_equal(json1: object, json2: object) -> bool:
    """
    Assert if two json objects are equal

    Prints out more useful error messages if not the case.
    Treats empty string as nan during comparison.
    Parameters:
    ----------
    json1: First json object
    json2: Second json object

    Returns
    -------
    Bool
    """
    json1_ordered = ordered(json1)
    json2_ordered = ordered(json2)
    if json1_ordered != json2_ordered:
        if type(json1_ordered) != type(json2_ordered):
            raise AssertionError(
                f"json1_ordered: {json1_ordered} not the same type: "
                "{type(json1_ordered)} as json2_ordered: {json2_ordered} "
                "of type {type(json2_ordered)}"
            )
        if len(json1_ordered) != len(json2_ordered):
            raise AssertionError(
                f"len(json1_ordered): {len(json1_ordered)} " "!= len(json2_ordered): {len(json2_ordered)}"
            )
        if isinstance(json1_ordered, list):
            for item1, item2 in zip(json1_ordered, json2_ordered):
                json_equal(item1, item2)
        if json1_ordered != json2_ordered:
            raise AssertionError("Json (element) not equal: " f"{json1_ordered} \n != \n {json2_ordered}")
    return True


def assert_presence_variables(ds: xarray.Dataset, variables: t.Sequence[str]):
    for var in variables:
        if var not in ds.variables.keys():
            raise ValueError(f"{var} not available in ds: {ds}")


def print_text_as_csv(text: str, filename: t.Union[str, pathlib.Path] = "/tmp/tmp.txt"):
    a2.utils.file_handling.csv_create(filename, row=[text], check_is_file_or_empty=False)
    print(open(filename).read())


def check_internet_connection():
    http = urllib3.PoolManager(timeout=3.0)
    try:
        r = http.request("GET", "google.com", preload_content=False)
    except urllib3.exceptions.MaxRetryError:
        return False
    code = r.status
    r.release_conn()
    if code == 200:
        return True
    else:
        return False


def check_python_version():
    from platform import python_version

    return python_version()


def print_copy_paste_fake_dataset(ds, name="fake_dataset"):
    print(f"def {name}():")

    def print_array(values, dtype):
        if "<" in dtype.__str__():
            dtype = "str_"
        elif "object" == dtype.__str__():
            dtype = "object_"
        elif "bool" == dtype.__str__():
            dtype = "bool_"
        elif "datetime64[ns]" == dtype.__str__():
            dtype = "np.datetime64"
        return (
            "np.array(["
            + ", ".join(
                [
                    "np.nan"
                    if not isinstance(x, str) and not isinstance(x, dict) and np.isnan(x)
                    else x.__str__()
                    if not isinstance(x, str) and not isinstance(x, np.datetime64)
                    else f"'{x}'"
                    for x in values
                ]
            )
            + f"], dtype=np.{dtype})"
        )

    for k, v in ds.variables.items():
        value_array = np.array(v.values, dtype=v.values.dtype)
        print(f"{k} = {print_array(value_array, dtype=v.values.dtype)}")
    print("return xarray.Dataset(")
    print("\tdata_vars=dict(")
    for k, v in ds.variables.items():
        if k == "index":
            continue
        print(f'\t\t{k}=(["index"], {k}),')
    print("\t),")
    print("\tcoords=dict(index=index),")
    print(")")


def print_debug(x):
    print(f"{np.min(x)=}")
    print(f"{np.max(x)=}")
    print(f"{np.mean(x)=}")
    print(f"{np.isnan(x).sum()=}")
    print(f"{x[:10]=}")


def not_overlapping(left: list, right: list) -> list:
    return list(set(left) - set(right))


def assert_equal_pandas_dataframe(left: pd.DataFrame, right: pd.DataFrame) -> None:
    try:
        pd.testing.assert_frame_equal(left, right, check_like=True)
    except AssertionError as e:
        message = ""
        if not_overlapping(left.columns, right.columns):
            message += f"df2 missing {not_overlapping(left.columns, right.columns)}"
        if not_overlapping(right.columns, left.columns):
            message += f"df1 missing {not_overlapping(right.columns, left.columns)}"
        raise AssertionError(f"{message}\n{e}")
