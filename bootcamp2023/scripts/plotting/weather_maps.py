import pathlib
import typing as t

import a2.dataset
import a2.plotting.utils_plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
from typing import Optional


def get_info_string_tweets(
    ds: xarray.Dataset,
    fields: Optional[t.List[str]] = None,
    add: t.Optional[t.List[str]] = None,
) -> str:
    """
    Returns string giving information on fields +
    add keys of all Tweets in dataset

    Parameters:
    ----------
    ds: Dataset
    fields: Keys to print, default: "text", "latitude_rounded",
            "longitude_rounded", "created_at", "tp", "raining", "source"
    add: Additional keys to print

    Returns
    -------
    Info string
    """

    def get_present_variables(ds, fields):
        return [ds[f].values for f in fields if f in ds]

    if fields is None:
        fields = [
            "text",
            "latitude_rounded",
            "longitude_rounded",
            "created_at",
            "tp",
            "raining",
            "source",
        ]
    if add is not None:
        fields = fields + add
    fields = [f for f in fields if f in ds]
    to_print = ""
    if ds.index.shape:
        for tweet in zip(*get_present_variables(ds, fields)):
            for label, value in zip(fields, tweet):
                to_print += f"{label}: {value}\n"
    else:
        for label, value in zip(fields, get_present_variables(ds, fields)):
            to_print += f"{label}: {value}\n"
    return to_print


def plot_precipiation_map(
    ds_precipitation: xarray.Dataset,
    ds_tweets: xarray.Dataset,
    key_time: str = "created_at_h",
    key_longitude: str = "longitude_rounded",
    key_latitude: str = "latitude_rounded",
    key_tp: str = "tp",
    n_time: int = 1,
    delta_time: int = 1,
    delta_time_units: str = "h",
    delta_longitude: float = 1,
    delta_latitude: float = 1,
    filename: Optional[t.Union[str, pathlib.Path]] = None,
    print_additional: t.Optional[t.List[str]] = None,
    add_time_before_plot: t.Optional[pd.Timedelta] = None,
    return_plots: bool = False,
) -> t.Union[plt.axes, t.Tuple[plt.axes, t.List]]:
    """
    Plot precipitation map corresponding to location and time of
    given Tweets and `n_time` plots in increments of `delta_time`
    [`delta_time_units`]

    Parameters:
    ----------
    ds_precipitation: Dataset of precipitation data
    ds_tweets: Dataset of Tweets
    key_time: Key of time variable
    key_longitude: Key of longitude variable
    key_latitude: Key of latitude variable
    key_tp: Key of latitude variable
    n_time: Number of additional delta times shown
            (1x +delta time, 1x-delta time -> n_time=1)
    delta_time: Value of time increments for additional plots
    delta_time_units: Units of time increments
    delta_longitude: Increment to include in weather maps to east and west
    delta_latitude: Increment to include in weather maps to north and south
    filename: Filename to save plot
    print_additional: Show additional fields of Tweet dataset
    add_time_before_plot: Increment time values from Tweet before looking
                          for corresponding value in precipitation dataset
    return_plots: Returns plot objects

    Returns
    -------
    axes(, plot object)
    """
    if print_additional is None:
        print_additional = []
    n_rows = ds_tweets.index.shape[0]
    n_cols = 1 + 2 * n_time
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15, 5 * n_rows))
    plots = np.full(np.shape(axes), np.nan, dtype=object)
    for i_tweet, (longitude, latitude, time) in enumerate(
        zip(
            ds_tweets[key_longitude].values,
            ds_tweets[key_latitude].values,
            ds_tweets[key_time].values,
        )
    ):
        for i_time, dt in enumerate(np.arange(-1 * delta_time * n_time, delta_time * (n_time + 1), delta_time)):
            ax = axes[i_tweet, i_time]
            time_to_plot = pd.to_datetime(time) + pd.Timedelta(f"{dt}{delta_time_units}")
            if add_time_before_plot:
                time_to_plot = time_to_plot + add_time_before_plot
            pcolormesh = (
                ds_precipitation[key_tp]
                .loc[f"{time_to_plot}"]
                .plot(
                    vmin=1e-6,
                    vmax=1e-3,
                    xlim=[
                        longitude - delta_longitude,
                        longitude + delta_longitude,
                    ],
                    ylim=[latitude - delta_latitude, latitude + delta_latitude],
                    ax=ax,
                    cmap="magma_r",
                )
            )
            plots[i_tweet, i_time] = pcolormesh
            ax.set_xlabel("")
            ax.set_ylabel("")
    for index, _ in enumerate(ds_tweets.index.values):
        title = get_info_string_tweets(
            a2.dataset.load_dataset.reset_index_coordinate(ds_tweets).sel(index=index),
            add=["preds_raining"] + print_additional,
        )
        axes[index, n_time].set_title(title)
    fig.tight_layout()
    a2.plotting.utils_plotting.save_figure(fig, filename)
    if return_plots:
        return axes, plots
    return axes
