import itertools
import logging
import multiprocessing
from functools import wraps
from time import time

import numpy as np
import pandas as pd
import xarray


def timing(f):
    """
    Wrapper that returns execution time and arguments of function.
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.info(f"func:{f.__name__} args:[{kw}] took: {te-ts} sec")
        return result

    return wrap


def parallelize(function, args, processes=-1, single_arg=False, kwargs_as_dict=None):
    """
    parallelize function with args provided in zipped format
    ----------
    function: function to parallelize
    args: args of function in zipped format

    Returns
    -------
    function applied to args
    """
    if processes == -1:
        processes = multiprocessing.cpu_count() - 1
    if single_arg:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(function, args)
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            if kwargs_as_dict is not None:
                kwargs_iter = itertools.repeat(kwargs_as_dict)
            else:
                kwargs_iter = itertools.repeat(dict())
            results = starmap_with_kwargs(pool, function, args, kwargs_iter)
    return results


def get_random_indices(n_sample, size_data):
    return np.random.choice(
        range(size_data),
        n_sample if n_sample < size_data else size_data,
        replace=False,
    )


def starmap_with_kwargs(pool, function, args_iter, kwargs_iter):
    if kwargs_iter is None:
        args_for_starmap = zip(itertools.repeat(function), args_iter)
    else:
        args_for_starmap = zip(itertools.repeat(function), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def get_keywords_default():
    header = "🏔️ OR 🏔️ OR ☀️ OR ☀️ OR 🌞 OR ⛅ OR ⛈️ OR ⛈️ OR 🌤️ OR 🌤️ "
    "OR 🌥️ OR 🌥️ OR 🌦️ OR 🌦️ OR 🌧️ OR 🌧️ OR 🌨️ OR 🌨️ OR 🌩️ OR 🌩️ OR ☔"
    " OR ⛄ OR blizzard OR cloudburst OR downpour OR drizzle OR flash "
    "flood OR flood OR flood stage OR forecast OR freezing rain OR hail"
    " OR ice storm OR lightning OR precipitation OR rain OR rain gauge"
    " OR rain shadow OR rainbands OR rain shower OR snow OR snow shower"
    " OR snowstorm OR sun OR sunny OR thunder OR thunderstorm"
    keywords = header.split(" OR ")
    return keywords


def is_same_type_data_array(ds, field, which_type=str):
    return all([isinstance(x, which_type) for x in ds[field].values])


def is_nan(ds, field, dims="index"):
    if is_same_type_data_array(ds, field):
        return xarray.DataArray(ds[field].values == "nan", dims=dims)
    else:
        return xarray.DataArray(pd.isna(ds[field].values), dims=dims)
