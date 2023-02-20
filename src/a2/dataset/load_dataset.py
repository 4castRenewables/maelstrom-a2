import ast
import json
import logging
import pathlib
import typing as t

import a2.dataset
import a2.utils
import numpy as np
import pandas
import pandas as pd
import xarray


def _convert_str_to_dict(x: str) -> object:
    """
    Converts non-nan strings to dictionary/list

    Some types like dictionary are not recovered when loaded from .nc/.csv file.
    This function converts them back to original type by excluding nan-values
    as they through an exception.
    Parameters:
    ----------
    x: string to convert

    Returns
    -------
    dictionary/list/tuple if in form of string otherwise string
    """
    try:
        if pd.isnull(x) or x == "nan" or x.strip() == "":
            return np.nan
        else:
            return ast.literal_eval(x)
    except Exception as e:
        raise Exception(f"Couldn't convert {x}") from e


def convert_str_to_dict(data_array: np.ndarray, processes=-1) -> np.ndarray:
    """
    Parallelize _convert_str_to_dict
    """
    return a2.utils.utils.parallelize(
        _convert_str_to_dict,
        data_array,
        processes=processes,
        single_arg=True,
    )


def reset_index_coordinate(ds: xarray.Dataset) -> xarray.Dataset:
    """
    Resets index variable to increasing integer values starting from 0

    Parameters:
    ----------
    ds: xarray.Dataset

    Returns
    -------
    dataset with reset index variable
    """
    ds["index"] = np.arange(np.shape(ds["index"].values)[0])
    return ds


def load_tweets_dataset(
    filename: t.Union[str, pathlib.Path],
    raw: bool = False,
    reset_index_raw: bool = True,
    reset_index: bool = True,
    drop_variables: t.Optional[list[str]] = None,
    convert_bounding_box: bool = False,
    open_dataset=False,
) -> xarray.Dataset:
    """
    loads dataset from disk and converts columns into convenient data formats

    Parameters:
    ----------
    filename: Filename of cn file
    reset_index: Reset index coordinate
    raw: Just load file, no conversions
    drop_variables: List of variables to drop
    convert_bounding_box: Convert bounding box coordinates from string to dictionary
    open_dataset: Open the dataset instead of loading it in memory

    Returns
    -------
    dataset of tweets
    """
    if open_dataset:
        ds = xarray.open_dataset(filename, drop_variables=drop_variables)
    else:
        ds = xarray.load_dataset(filename, drop_variables=drop_variables)
    if reset_index_raw and "index" in ds.variables:
        ds = reset_index_coordinate(ds)
    if raw:
        return ds
    if "bounding_box" in ds.variables and convert_bounding_box:
        ds["bounding_box"] = (
            ["index"],
            convert_str_to_dict(ds["bounding_box"].values),
        )
    if "created_at" in ds.variables:
        ds["created_at"] = (["index"], pd.to_datetime(ds.created_at).values)
    if "author_id" in ds.variables:
        ds["author_id"] = (["index"], ds["author_id"].astype(int).values)
    if reset_index and "index" in ds.variables:
        ds = reset_index_coordinate(ds)
    return ds


def load_tweets_dataframe_from_jsons(
    list_of_filenames: t.Sequence,
) -> pandas.DataFrame:
    """
    loads json files and converts them to single dataframe

    Parameters:
    ----------
    list_of_filenames: list of filenames of json file

    Returns
    -------
    dataframe of tweets
    """
    dataframe_list = []
    for filename in list_of_filenames:
        with open(filename) as json_file:
            for json_response in json.loads(json_file.read()):
                dataframe_list.append(pd.json_normalize(json_response["data"]))
    return pd.concat(dataframe_list, ignore_index=True)


def load_tweets_dataset_from_jsons(
    list_of_filenames: t.Sequence,
) -> xarray.Dataset:
    """
    loads json files and converts them to single xarray dataset

    Parameters:
    ----------
    list_of_filenames: list of filenames of json file

    Returns
    -------
    xarray dataset of tweets
    """
    return load_tweets_dataframe_from_jsons(list_of_filenames).to_xarray()


def _is_coordinate(ds: xarray.Dataset, key: t.Hashable) -> bool:
    return key in ds.coords.keys()


def _get_item(value):
    try:
        return value.item()
    except AttributeError:
        return value


def any_type_present(array, types_to_check):
    return any(type(_get_item(x)) in types_to_check for x in np.ndarray.flatten(array))


def save_dataset(
    ds: xarray.Dataset,
    filename: t.Union[str, pathlib.Path] = "test.nc",
    add_attributes: str = "",
    no_conversion: bool = True,
    encode_time: bool = True,
) -> None:
    """
    saves xarray dataset to file.

    Dictionaries and lists cannot be natively saved with xarray.
    Therefore, they are converted to strings before the file is saved.
    Parameters:
    ----------
    filename: name of file dataset is saved to

    Returns
    -------
    """
    types_to_convert = [dict, list]
    if not no_conversion:
        for k, v in ds.variables.items():
            if any_type_present(ds[k].values, types_to_check=types_to_convert):
                logging.info(f"Converting field: {k} to strings!")
                if _is_coordinate(ds, k):
                    ds = a2.dataset.utils_dataset.add_coordinates(ds, k, np.array([str(x) for x in ds[k].values]))
                else:
                    ds = a2.dataset.utils_dataset.add_variable(ds, k, np.array([str(x) for x in ds[k].values]))
    attributes = ds.attrs
    if "description" in attributes:
        attributes["description"] = attributes["description"] + add_attributes
    else:
        attributes["description"] = add_attributes
    ds.attrs = attributes
    encoding = None
    if encode_time:
        keys_time = get_time_variables(ds)
        encoding = {k: {"units": "seconds since 1900-01-01"} for k in keys_time}
    ds.to_netcdf(filename, encoding=encoding)


def get_time_variables(ds):
    return [v for v in ds.variables if ds[v].dtype == "datetime64[ns]"]


def save_dataset_split(
    ds: xarray.Dataset,
    split_by: str = "day",
    key_time: str = "time",
    prefix: t.Union[str, pathlib.Path] = "ds_",
) -> None:
    """
    Splits dataset in multiple files and saves them individually.

    Splits are possible along different units of time, currently day and year.
    Parameters:
    ----------
    ds: Dataset to split
    split_by: Split by `day` or `time`
    key_time: Key name of time variable
    prefix: Prefix for new files

    Returns
    -------
    """
    if split_by == "day":
        dates, datasets = zip(*ds.resample({key_time: "1D"}))
        texts = pd.DatetimeIndex(np.array(dates)).day
    elif split_by == "year":
        dates, datasets = zip(*ds.resample({key_time: "1Y"}))
        texts = pd.DatetimeIndex(np.array(dates)).year
    else:
        raise NotImplementedError(f"split_by {split_by} not implemented yet!")
    filenames = [f"{prefix}{t}.nc" for t in texts]
    xarray.save_mfdataset(datasets, filenames)


def load_weather_stations(filename: str, drop_columns: str | list = "Unnamed: 0"):
    df = pd.read_csv(
        filename,
        dtype={"latitude": float, "longitude": float, "prcp_amt": float},
        parse_dates=["ob_end_time"],
    )
    df = df.set_index(["latitude", "longitude"])
    if drop_columns:
        df = df.drop(columns=drop_columns)
    return df


def load_radar_dataset(files):
    """load multiple netcdf radar files converted from nimrod"""
    return xarray.open_mfdataset(files, combine_attrs="drop_conflicts")
