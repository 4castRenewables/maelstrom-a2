import numpy as np
import xarray


def reset_index_coordinate(ds):
    """
    Resets index variable to increasing integer values starting from 0

    Parameters:
    ----------
    ds: xarray.Dataset

    Returns
    -------
    dataset with reset index variable
    """
    if "index" in ds.variables.keys():
        ds["index"] = np.arange(np.shape(ds["index"].values)[0])
    return ds


def load_tweets_dataset(folder):
    ds = xarray.load_dataset(folder)
    ds = reset_index_coordinate(ds)
    return ds


def get_grouped_dataset(ds, group_by, sort_by="id"):
    """
    Groups dataset by `group_by` counts occurence of unique values and sorts them

    Parameters:
    ----------
    ds: xarray.Dataset
    group_by: variable name in `ds` to group
    sort_by: unique values will be counted of `group_by` variable,
    then all fields will have number of unique values, pick one of those

    Returns
    -------
    grouped dataset with count of unique values
    """
    ds_grouped_unsorted = ds.groupby(group_by).count()
    ds_grouped = ds_grouped_unsorted.sortby(sort_by, ascending=False)
    return ds_grouped
