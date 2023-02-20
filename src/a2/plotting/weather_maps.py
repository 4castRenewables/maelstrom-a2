import collections
import functools
import pathlib
import typing as t
import warnings

import a2.dataset.radar
import a2.dataset.tweets
import a2.plotting.axes_utils
import a2.plotting.utils_plotting
import a2.utils.checks
import a2.utils.constants
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

TypeKeyTweets = object
TypeKeyRadar = object


def get_info_string_tweets(
    ds: xarray.Dataset,
    fields: t.List[str] = None,
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
    filename: t.Union[str, pathlib.Path] = None,
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


def circle_scatter(
    ax: plt.axes,
    x: np.ndarray,
    y: np.ndarray,
    color: t.Union[str, list, np.ndarray] = None,
    radius: t.Union[float, list, np.ndarray] = 1,
    fill: bool = True,
    cmap: matplotlib.colormaps = None,
    norm=None,
    alpha: float = 1,
    show_colorbar: bool = False,
):
    """
    Scatter plot with markers as circles, which allows for color mapping to colorbar.

    Parameters:
    ----------
    ax: matplotlib axes
    x: x coordinates on ax
    y: y coordinates on ax
    color: Color of circles, provide normed values if using colormap as color
    radius: Radius of circles in units of the plot
    fill: Wether to fille the circles with `color`
    cmap: Colormap used as color
    norm: Norm for colormap
    alpha: Transparency of circles
    show_colorbar: Show colorbar if using `cmap`

    Returns
    -------
    """

    if isinstance(radius, float) or isinstance(radius, int):
        radius = [radius for i in range(len(x))]
    if isinstance(color, str) or not isinstance(color, collections.abc.Iterable):
        color = [color for i in range(len(x))]
    if not fill:
        circles = [plt.Circle((xi, yi), color=c, radius=r, fill=fill) for xi, yi, r, c in zip(x, y, radius, color)]
        for c in circles:
            ax.add_artist(c)
    else:
        circles = [plt.Circle((xi, yi), radius=r) for xi, yi, r in zip(x, y, radius)]
        p = matplotlib.collections.PatchCollection(circles, cmap=cmap, norm=norm, alpha=alpha)
        p.set_array(color)
        ax.add_collection(p)
        cbar = None
        if cmap is not None and show_colorbar:
            cbar = plt.colorbar(p, ax=ax)
        return cbar


def _plot_radar_map_with_tweets(
    index: np.datetime64,
    ds: xarray.Dataset,
    path_to_dapceda: t.Optional[pathlib.Path] = None,
    key_tweets: TypeKeyTweets = None,
    key_radar: TypeKeyRadar = None,
    cumulative_radar: bool = False,
    cumulative_delta_time: float = 1,
    cumulative_delta_time_units: str = "h",
    vmin: t.Optional[float] = None,
    vmax: t.Optional[float] = None,
    xlim: t.Optional[list] = None,
    ylim: t.Optional[list] = None,
    fontsize: int = 20,
    marker_sizer_rescale: float = 0.01,
    colormap: str = "viridis",
    circle_alpha: float = 0.4,
    ax: t.Optional[plt.axes] = None,
    fig: t.Optional[plt.figure] = None,
    circle_size_constant: bool = False,
    circle_size: float = 0.1,
):
    """
    Plot background radar map with precipitation estimates for Tweets
    based on data from weather stations

    Parameters:
    ----------
    index: index in form of time
    ds: Tweet dataset that will be plotted
    path_to_dapceda: Path to folder with raw radar files
    key_tweets: Data class describing keys in dataframe for Tweets
    key_radar: Data class describing keys in dataframe for radar
    cumulative_radar: Radar is computed cumulatively
    cumulative_delta_time: Time range for which rada data is summed cumulatively
    cumulative_delta_time_units: Units of `cumulative_delta_time`
    vmin: Minimum value of colorscale depicting precipitation
    vmax: Maximum value of colorscale depicting precipitation
    xlim: Limits of x-axis showing longitude
    ylim: Limits of y-axis showing latitude
    fontsize: Font size of all text in plot
    marker_sizer_rescale: Rescaling factor for marker scaling (default 0.01: [km] * 0.01 ~ [deg lat/lon])
    colormap: Colormap used for precipitation
    circle_alpha: Alpha value for circle plot
    ax: Mapltotlib axes for plot
    fig: Figure of plot
    circle_size_constant: Setting circle size to constant value as specified here

    Returns
    -------
    """
    path_to_dapceda = a2.utils.file_handling.get_folder_radar(path_to_dapceda)
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    key_radar = a2.dataset.radar.get_key_radar(key_radar, cumulative_radar=cumulative_radar)
    time = index
    if ax is None:
        fig, ax = a2.plotting.utils_plotting.create_figure_axes(aspect="equal")
    if xlim is None:
        xlim = a2.utils.constants.UK_LONGITUDE_LIMIT
    if ylim is None:
        ylim = a2.utils.constants.UK_LATITUDE_LIMIT
    cmap = matplotlib.colormaps[colormap]
    if cumulative_radar:
        ds_radar = a2.dataset.radar.nimrod_ds_cumulative_from_time(
            path_to_dapceda, time, time_delta=cumulative_delta_time, time_delta_units=cumulative_delta_time_units
        )
    else:
        ds_radar = a2.dataset.radar.nimrod_ds_from_time(path_to_dapceda, time)
    if ds_radar is None:
        return
    tp = ds_radar[key_radar.tp].values
    if vmin is None:
        vmin = np.nanmin(tp)
    if vmax is None:
        vmax = np.nanmax(tp)
    norm = a2.plotting.utils_plotting.get_norm(None, vmin=vmin, vmax=vmax)
    pc = ds_radar[key_radar.tp][0].plot(
        x="longitude", y="latitude", ax=ax, xlim=xlim, ylim=ylim, norm=norm, cmap=colormap
    )
    a2.plotting.axes_utils.set_colorbar(pc.colorbar.ax, f"{key_radar.tp}", fontsize)

    size_marker = _retrieve_size_marker(
        ds, circle_size_constant, key_tweets.distance_stations, marker_sizer_rescale, circle_size
    )
    color_normed = [norm(x) for x in ds[key_tweets.station_tp].values]
    circle_scatter(
        ax,
        *(ds.longitude.values, ds.latitude.values),
        radius=size_marker,
        color=color_normed,
        cmap=cmap,
        norm=norm,
        alpha=circle_alpha,
        show_colorbar=False,
    )
    circle_scatter(ax, *(ds.longitude.values, ds.latitude.values), radius=size_marker, color="red", fill=False)
    _set_axes_weather_map(ax, xlim, ylim, fontsize, f"{pd.to_datetime(time)}")
    ax.set_aspect("equal")


def _retrieve_size_marker(
    ds: xarray.Dataset,
    circle_size_constant: bool,
    key_twitter_distance_station: str,
    marker_sizer_rescale: float,
    circle_size: float = 0.1,
) -> np.ndarray:
    """Get the size of the markers (e.g, circles) in correct shape"""
    if circle_size_constant or key_twitter_distance_station not in ds:
        size_marker = np.array([circle_size for _ in ds.latitude.values])
    else:
        size_marker = ds[key_twitter_distance_station].values * marker_sizer_rescale
    return size_marker


def _plot_tp_station_tweets(
    index,
    ds: xarray.Dataset,
    df_stations: pd.DataFrame,
    key_tweets: TypeKeyTweets = None,
    key_station_tp: str = "prcp_amt",
    key_station_time: str = "ob_end_time",
    vmin: float = None,
    vmax: float = None,
    xlim: t.Sequence = None,
    ylim: t.Sequence = None,
    time_offset_station: float = 0.5,
    time_offset_station_units: str = "h",
    fontsize: float = 20,
    marker_sizer_rescale: float = 0.01,
    colormap: str = "viridis",
    circle_alpha: float = 0.8,
    circle_size_constant: bool = False,
    circle_size: float = 0.1,
    radius_station: float = 0.1,
    ax: plt.axes = None,
    fig: plt.figure = None,
):
    """
    Plot station precipitation values and values assigned to Tweets

    Parameters:
    ----------
    index: Index in form of time
    ds: Tweet dataset that will be plotted
    df_stations: Data frame of weather station data
    key_tweets: Data class describing keys in dataframe for Tweets
    key_station_tp: Key in station dataframe for precipitation
    key_station_time: Key in station dataframe for time
    vmin: Minimum value of colorscale depicting precipitation
    vmax: Maximum value of colorscale depicting precipitation
    xlim: Limits of x-axis showing longitude
    ylim: Limits of y-axis showing latitude
    time_offset_station: Adding this offset to index (time) and get values of weather station at this time
    time_offset_station_units: Units of `time_offset_station`
    fontsize: Font size of all text in plot
    marker_sizer_rescale: Rescaling factor for marker scaling (default 0.01: [km] * 0.01 ~ [deg lat/lon])
    radius_station: Size of marker for station values
    circle_size_constant: Setting circle size to constant value as specified here
    colormap: Colormap used for precipitation
    circle_alpha: Alpha value for circle plot
    ax: Mapltotlib axes for plot
    fig: Figure of plot

    Returns
    -------
    """
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    time = index
    if ax is None:
        fig, ax = a2.plotting.utils_plotting.create_figure_axes(aspect="equal")
    if xlim is None:
        xlim = a2.utils.constants.UK_LONGITUDE_LIMIT
    if ylim is None:
        ylim = a2.utils.constants.UK_LATITUDE_LIMIT
    tp_twitter = ds[key_tweets.station_tp].values
    time_station = time + pd.Timedelta(f"{time_offset_station}{time_offset_station_units}")
    df_sel = df_stations[(df_stations[key_station_time] == pd.to_datetime(time_station))].drop_duplicates(
        keep="last", subset=["latitude", "longitude"]
    )
    tp_station = df_sel[key_station_tp].values
    if vmin is None:
        vmin = np.nanmin(tp_twitter)
    if vmax is None:
        vmax = np.nanmax(tp_twitter)
    cmap = matplotlib.colormaps[colormap]
    norm = a2.plotting.utils_plotting.get_norm(None, vmin=vmin, vmax=vmax)
    color_normed_tweets = [norm(x) for x in tp_twitter]
    size_marker = _retrieve_size_marker(
        ds, circle_size_constant, key_tweets.distance_stations, marker_sizer_rescale, circle_size
    )
    ax.text(0.1, 0.1, f"time_station: {time_station}", transform=ax.transAxes, fontsize=fontsize)
    colorbar = circle_scatter(
        ax,
        *(ds.longitude.values, ds.latitude.values),
        radius=size_marker,
        color=color_normed_tweets,
        cmap=cmap,
        norm=norm,
        alpha=circle_alpha,
        show_colorbar=True,
    )
    if colorbar is not None:
        a2.plotting.axes_utils.set_colorbar(colorbar.ax, "tp_mm", fontsize)

    color_normed_stations = [norm(x) for x in tp_station]
    circle_scatter(
        ax,
        *(df_sel.longitude.values, df_sel.latitude.values),  # type: ignore
        radius=radius_station,
        cmap=cmap,
        norm=norm,
        color=color_normed_stations,
        alpha=circle_alpha,
    )
    circle_scatter(
        ax,
        *(df_sel.longitude.values, df_sel.latitude.values),  # type: ignore
        radius=radius_station,
        color="red",
        fill=False,
    )
    circle_scatter(ax, *(ds.longitude.values, ds.latitude.values), radius=size_marker, color="blue", fill=False)

    _set_axes_weather_map(ax, xlim, ylim, fontsize, f"{pd.to_datetime(time)}")
    ax.set_aspect("equal")


def _set_axes_weather_map(ax, xlim, ylim, fontsize, title):
    """Predefine subset of axes properties required for all weather maps, based on `a2.plotting.axes_utils.set_axes`"""
    a2.plotting.axes_utils.set_axes(
        ax, xlim=xlim, ylim=ylim, fontsize=fontsize, title=title, label_x="longitude", label_y="latitude"
    )


class MapsPlotter:
    """
    Class used to plot weather maps with multiple axes.
    Determines index based on `choice_type` which indicates time shown in individual axes.
    Selects subset of dataset `ds` to be plotted on individual axes based on `selector`.
    """

    def __init__(
        self,
        ds: xarray.Dataset,
        grid_shape: t.Tuple[int, ...],
        selector: t.Callable,
        figsize_default: float = 15,
        figsize: t.Optional[t.List[float]] = None,
        key_time: str = None,
        choice_type: str = "random",
        increment_time_value: np.datetime64 = None,
        increment_time_delta: float = None,
        increment_time_delta_units: str = None,
        aspect: str = "auto",
        height_ratios: t.Optional[t.List] = None,
    ) -> None:
        """
        Initializes class including setting up selection and choice.

        Parameters:
        ----------
        ds: Dataset to be plotted
        grid_shape: Shape of axes grid (rows, columns)
        selector: Determines which Tweets to select for specific axes based on `choice_type`
        figsize_default: Default figsize along largest axis
        figsize: Define size of figure
        key_time: Choice is computed based on this key in `ds`
        choice_type: Choice for index of individual axes:
            "random": Returns random index [int] based on size of `ds`
            "random_time": Returns randomly selected time from `ds[key_time]`
            "increment_time": Returns times as increments from `increment_time_value` in steps
                              `increment_time_delta` [`increment_time_delta_units`]
        increment_time_value: Starting time for `choice_type`="increment_time"
        increment_time_delta: Delta time for `choice_type`="increment_time"
        increment_time_delta_units: Delta time units `choice_type`="increment_time"
        aspect: Aspect of axes ("auto"/"equal")
        height_ratios: Ratios of height of axes along y-axis of figure

        Returns
        -------
        """
        self.ds = ds
        self.grid_shape = grid_shape
        self.selector = selector
        self.figsize_default = figsize_default
        self.increment_time_value = increment_time_value
        self.increment_time_delta = increment_time_delta
        self.increment_time_delta_units = increment_time_delta_units
        self.figsize = self._set_figsize(figsize=figsize)
        self.key_time = key_time
        self.n_sample = self.grid_shape[0] * self.grid_shape[1]
        self.choices = self._set_choices(choice_type=choice_type)
        self.ds_selected = self._do_selection()
        self.aspect = aspect
        self.height_ratios = height_ratios

    def _validate_increment_time(self):
        a2.utils.checks.assert_type(self.increment_time_value, np.datetime64)
        a2.utils.checks.assert_type(self.increment_time_delta, float | int)
        a2.utils.checks.assert_type(self.increment_time_delta_units, str)

    def _set_figsize(self, figsize: t.Optional[t.List[float]]):
        if figsize is None:
            figsize = [0.0, 0.0]
            num_plots_max_axis = 0 if self.grid_shape[0] > self.grid_shape[1] else 1
            figsize[num_plots_max_axis] = self.figsize_default
            figsize[1 - num_plots_max_axis] = (
                self.figsize_default * self.grid_shape[1 - num_plots_max_axis] / self.grid_shape[num_plots_max_axis]
            )
            figsize = figsize[::-1]
        return figsize

    def _do_selection(self):
        ds_selected = []
        for choice in self.choices:
            ds_selected.append(self.selector(choice, self.ds))
        return ds_selected

    def _set_choices(self, choice_type: str) -> list:
        ds = self.ds
        if choice_type == "random":
            self.size_data = len(ds.index.values)
            return list(a2.utils.utils.get_random_indices(self.n_sample, self.size_data))
        elif choice_type == "random_time":
            time_values = ds[self.key_time].values
            self.size_data = len(time_values)
            return [time_values[i] for i in a2.utils.utils.get_random_indices(self.n_sample, self.size_data)]
        elif choice_type == "increment_time":
            self._validate_increment_time()
            start_time = pd.to_datetime(self.increment_time_value)  # type: ignore
            times = []
            if not isinstance(self.increment_time_delta, (int, float)):
                raise TypeError(f"{self.increment_time_delta=} should be float!")
            for i in range(self.n_sample):
                times.append(
                    start_time + pd.Timedelta(f"{i*self.increment_time_delta}{self.increment_time_delta_units}")
                )
            return times
        else:
            raise NotImplementedError(f"{choice_type=} not implemented!")

    def plot(self, plot_fn: t.Callable, processes: int = -1):
        """plot axes in parallel"""
        self.fig = a2.plotting.parallel_plotting.parallel_plot(
            plot_fn=plot_fn,
            data=zip(self.choices, self.ds_selected),
            grid_shape=self.grid_shape,
            figsize=self.figsize,
            aspect=self.aspect,
            height_ratios=self.height_ratios,
            processes=processes,
        )
        return self.fig


def plot_radar_map_with_tweets(
    ds: xarray.Dataset,
    grid_shape: t.Optional[t.Tuple[int, int]] = None,
    figsize: t.Optional[t.List] = None,
    figsize_default: float = 30,
    path_to_dapceda: t.Optional[pathlib.Path] = None,
    key_tweets: TypeKeyTweets = None,
    key_radar: TypeKeyRadar = None,
    vmin: t.Optional[float] = None,
    vmax: t.Optional[float] = None,
    xlim: t.Optional[list] = None,
    ylim: t.Optional[list] = None,
    choice_type: str = "random_time",
    choice_time: str = "time_radar",
    selection_delta_time: float = 10,
    selection_delta_time_units: str = "m",
    selection_key_twitter_time: str = "time_radar",
    selector_use_limits: t.Sequence = None,
    increment_time_value: np.datetime64 = None,
    increment_time_delta: float = 1,
    increment_time_delta_units: str = "h",
    colormap: str = "viridis",
    circle_alpha: float = 1,
    cumulative_radar: bool = False,
    cumulative_delta_time: float = 1,
    cumulative_delta_time_units: str = "h",
    circle_size_constant: bool = False,
    processes: int = -1,
) -> plt.figure:
    """
    Plot background radar map with precipitation estimates for Tweets
    based on data from weather stations

    Parameters:
    ----------
    ds: Tweet dataset that will be plotted
    grid_shape: Shape of axes grid (rows, columns), default: (2, 1)
    figsize: Define size of figure
    figsize_default: Default figsize along largest axis
    path_to_dapceda: Path to folder with raw radar files
    key_tweets: Data class describing keys in dataframe for Tweets
    key_radar: Data class describing keys in dataframe for radar
    vmin: Minimum value of colorscale depicting precipitation
    vmax: Maximum value of colorscale depicting precipitation
    xlim: Limits of x-axis showing longitude
    ylim: Limits of y-axis showing latitude
    choice_type: Choice for index of individual axes:
        "random": Returns random index [int] based on size of `ds`
        "random_time": Returns randomly selected time from `ds[key_time]`
        "increment_time": Returns times as increments from `increment_time_value`
                          in steps `increment_time_delta` [`increment_time_delta_units`]
    choice_time: Choice is computed based on this key in `ds`
    selection_delta_time: Delta time within which selection from dataset made
    selection_delta_time_units: Units for selection_delta_time
    selection_key_twitter_time: Selection from dataset for individual axes based on this time key
    selector_use_limits: Use range in time within -delta time up to +delta time,
                         include/exclude limits as [True/False, True/False]
    increment_time_value: Starting time for `choice_type`="increment_time"
    increment_time_delta: Delta time for `choice_type`="increment_time"
    increment_time_delta_units: Delta time units `choice_type`="increment_time"
    colormap: Colormap used for precipitation
    circle_alpha: Alpha value for circle plot
    cumulative_radar: Radar is computed cumulatively
    cumulative_delta_time: Time range for which rada data is summed cumulatively
    cumulative_delta_time_units: Units of `cumulative_delta_time`
    circle_size_constant: Setting circle size to constant value as specified here
    processes: Number of plots (axes) generated in parallel

    Returns
    -------
    """
    if grid_shape is None:
        grid_shape = (2, 1)
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    partial_time_selector = functools.partial(
        time_selector,
        delta_time=selection_delta_time,
        delta_time_units=selection_delta_time_units,
        key_time=selection_key_twitter_time,
        use_limits=selector_use_limits,
    )
    p = MapsPlotter(
        ds,
        grid_shape=grid_shape,
        figsize_default=figsize_default,
        figsize=figsize,
        choice_type=choice_type,
        key_time=choice_time,
        selector=partial_time_selector,
        increment_time_value=increment_time_value,
        increment_time_delta=increment_time_delta,
        increment_time_delta_units=increment_time_delta_units,
        aspect="equal",
    )
    partial_plot_radar_map_with_tweets = functools.partial(
        _plot_radar_map_with_tweets,
        path_to_dapceda=path_to_dapceda,
        key_tweets=key_tweets,
        key_radar=key_radar,
        cumulative_radar=cumulative_radar,
        cumulative_delta_time=cumulative_delta_time,
        cumulative_delta_time_units=cumulative_delta_time_units,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
        colormap=colormap,
        circle_alpha=circle_alpha,
        circle_size_constant=circle_size_constant,
    )
    fig = p.plot(partial_plot_radar_map_with_tweets, processes=processes)
    return fig


def plot_tp_station_tweets(
    ds: xarray.Dataset,
    df_stations: pd.DataFrame,
    grid_shape: t.Optional[t.Tuple] = None,
    figsize: t.Optional[t.List] = None,
    figsize_default: float = 30,
    key_tweets: TypeKeyTweets = None,
    key_station_tp: str = "prcp_amt",
    key_station_time: str = "ob_end_time",
    vmin: float = None,
    vmax: float = None,
    xlim: t.Sequence = None,
    ylim: t.Sequence = None,
    time_offset_station: float = 0.5,
    time_offset_station_units: str = "h",
    selector_key_time: str = "created_at",
    selection_delta_time: float = 0.5,
    selection_delta_time_units: str = "h",
    selector_use_limits: t.Sequence = None,
    choice_type: str = "random_time",
    choice_time: str = "time_half",
    increment_time_value=None,
    increment_time_delta: float = 1,
    increment_time_delta_units: str = "h",
    fontsize: float = 20,
    marker_sizer_rescale: float = 0.01,
    colormap: str = "viridis",
    circle_alpha: float = 0.8,
    circle_size_constant: bool = False,
    circle_size: float = 0.1,
    processes: int = -1,
) -> plt.figure:
    """
    Plot station precipitation values and values assigned to Tweets

    Parameters:
    ----------
    ds: Tweet dataset that will be plotted
    df_stations: Data frame of weather station data
    grid_shape: Shape of axes grid (rows, columns)
    figsize: Define size of figure
    figsize_default: Default figsize along largest axis
    key_tweets: Data class describing keys in dataframe for Tweets
    key_station_tp: Key in station dataframe for precipitation
    key_station_time: Key in station dataframe for time
    vmin: Minimum value of colorscale depicting precipitation
    vmax: Maximum value of colorscale depicting precipitation
    xlim: Limits of x-axis showing longitude
    ylim: Limits of y-axis showing latitude
    time_offset_station: Adding this offset to index (time) and get values of weather station at this time
    time_offset_station_units: Units of `time_offset_station`
    selector_key_time: Selection from dataset for individual axes based on this time key
    selection_delta_time: Delta time within which selection from dataset made
    selection_delta_time_units: Units for selection_delta_time
    selector_use_limits: Use range in time within -delta time up to +delta time,
                         include/exclude limits as [True/False, True/False]
    choice_type: Choice for index of individual axes:
        "random": Returns random index [int] based on size of `ds`
        "random_time": Returns randomly selected time from `ds[key_time]`
        "increment_time": Returns times as increments from `increment_time_value`
                          in steps `increment_time_delta` [`increment_time_delta_units`]
    choice_time: Choice is computed based on this key in `ds`
    increment_time_value: Starting time for `choice_type`="increment_time"
    increment_time_delta: Delta time for `choice_type`="increment_time"
    increment_time_delta_units: Delta time units `choice_type`="increment_time"
    fontsize: Font size of all text in plot
    marker_sizer_rescale: Rescaling factor for marker scaling (default 0.01: [km] * 0.01 ~ [deg lat/lon])
    colormap: Colormap used for precipitation
    circle_alpha: Alpha value for circle plot
    circle_size_constant: Setting circle size to constant value as specified here
    processes: Number of plots (axes) generated in parallel

    Returns
    -------
    figure
    """
    if grid_shape is None:
        grid_shape = (2, 1)
    if key_tweets is None:
        key_tweets = a2.dataset.tweets.KeyTweets()
    partial_time_selector = functools.partial(
        time_selector,
        delta_time=selection_delta_time,
        delta_time_units=selection_delta_time_units,
        key_time=selector_key_time,
        use_limits=selector_use_limits,
    )
    p = MapsPlotter(
        ds,
        grid_shape=grid_shape,
        figsize_default=figsize_default,
        figsize=figsize,
        choice_type=choice_type,
        key_time=choice_time,
        selector=partial_time_selector,
        increment_time_value=increment_time_value,
        increment_time_delta=increment_time_delta,
        increment_time_delta_units=increment_time_delta_units,
        aspect="equal",
        height_ratios=None,
    )
    partial_plot_tp_station_tweets = functools.partial(
        _plot_tp_station_tweets,
        df_stations=df_stations,
        key_tweets=key_tweets,
        key_station_tp=key_station_tp,
        key_station_time=key_station_time,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
        time_offset_station=time_offset_station,
        time_offset_station_units=time_offset_station_units,
        fontsize=fontsize,
        marker_sizer_rescale=marker_sizer_rescale,
        colormap=colormap,
        circle_size_constant=circle_size_constant,
        circle_size=circle_size,
        circle_alpha=circle_alpha,
    )
    fig = p.plot(partial_plot_tp_station_tweets, processes=processes)
    return fig


def time_selector(
    time,
    ds: xarray.Dataset,
    delta_time: float = 0.5,
    delta_time_units: str = "h",
    key_time: str = "created_at",
    use_limits: t.Optional[list] = None,
) -> xarray.Dataset:
    """
    Select part of dataset `ds` where `key_time` is within upper and/or lower delta time around `time`

    Parameters:
    ----------
    time: Time
    ds: Dataset to select from
    delta_time: Delta making up lower and/or upper limit of selected period of time
    delta_time_units: Units of `delta_time`
    key_time: Key in dataset that defines time
    use_limits: Use range in time within -delta time up to +delta time,
                include/exclude limits as [True/False, True/False]

    Returns
    -------
    dataset
    """
    if use_limits is None:
        use_limits = [True, True]
    time_lower_limit = pd.to_datetime(time) - pd.Timedelta(f"{delta_time}{delta_time_units}")
    time_upper_limit = pd.to_datetime(time) + pd.Timedelta(f"{delta_time}{delta_time_units}")
    if use_limits[0] and use_limits[1]:
        return ds.where((ds[key_time] >= time_lower_limit) & (ds[key_time] <= time_upper_limit), drop=True)
    elif use_limits[0]:
        return ds.where((ds[key_time] >= time_lower_limit) & (ds[key_time] <= time), drop=True)
    elif use_limits[1]:
        return ds.where((ds[key_time] >= time) & (ds[key_time] <= time_upper_limit), drop=True)
    else:
        warnings.warn(f"Time selector chosen but not limits {use_limits=} used: returning full dataset!")
        return ds


def plot_radar_from_time(
    time,
    path_to_dapceda: pathlib.Path = None,
    cumulative: bool = False,
    time_delta: float = 1,
    time_delta_units: str = "h",
    vmin: float = None,
    vmax: float = None,
    ax: plt.axes = None,
    xlim: t.Optional[list] = None,
    ylim: t.Optional[list] = None,
) -> xarray.Dataset:
    """
    Plot radar map at certain time.
    Returns radar dataset.

    Parameters:
    ----------
    time: Time
    path_to_dapceda: Path to folder with raw radar files
    cumulative: Radar is computed cumulatively
    time_delta: Time range for which rada data is summed cumulatively
    time_delta_units: Units of `time_delta`
    vmin: Minimum value of colorscale depicting precipitation
    vmax: Maximum value of colorscale depicting precipitation
    ax: Mapltotlib axes for plot
    xlim: Limits of x-axis showing longitude
    ylim: Limits of y-axis showing latitude

    Returns
    -------
    dataset
    """
    if path_to_dapceda is None:
        path_to_dapceda = a2.utils.file_handling.get_folder_radar(path_to_dapceda)
    if ax is None:
        fig, ax = a2.plotting.utils_plotting.create_figure_axes(aspect="equal")
    if xlim is None:
        xlim = a2.utils.constants.UK_LONGITUDE_LIMIT
    if ylim is None:
        ylim = a2.utils.constants.UK_LATITUDE_LIMIT
    if cumulative:
        ds = a2.dataset.radar.nimrod_ds_cumulative_from_time(
            path_to_dapceda, time, time_delta=time_delta, time_delta_units=time_delta_units
        )
        key_tp = "tp_mm_cum"
    else:
        ds = a2.dataset.radar.nimrod_ds_from_time(path_to_dapceda, time)
        key_tp = "tp_mm"

    ds[key_tp][0].plot(
        x="longitude",
        y="latitude",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        xlim=xlim,
        ylim=ylim,
    )

    return fig, ds
