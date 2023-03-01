import logging
import pathlib
import typing as t

import a2.dataset
import a2.plotting.utils_plotting
import a2.utils.constants
import matplotlib.offsetbox
import matplotlib.pyplot as plt
import numpy as np
import xarray


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../data"


def plot_histogram_2d(
    x: t.Union[np.ndarray, str],
    y: t.Union[np.ndarray, str],
    ds: xarray.Dataset | None = None,
    bins: t.Sequence | None = None,
    xlim: t.Sequence | None = None,
    ylim: t.Sequence | None = None,
    ax: a2.utils.constants.TYPE_MATPLOTLIB_AXES | None = None,
    ax_colorbar: a2.utils.constants.TYPE_MATPLOTLIB_AXES | None = None,
    log: t.Union[bool, t.List[bool]] = False,
    filename: t.Union[str, pathlib.Path] = None,
    fig: a2.utils.constants.TYPE_MATPLOTLIB_FIGURES | None = None,
    n_bins: t.Union[int, t.List[int]] = 60,
    norm: str | None = None,
    linear_thresh: t.Union[float, t.List[float]] = 1e-9,
    label_x: str | None = None,
    label_y: str | None = None,
    label_colorbar: str | None = None,
    font_size: int = 12,
    overplot_values: bool = False,
    cmap: str = "plasma",
    overplot_round_base: int | None = None,
    overplot_color: str = "gray",
    return_matrix: bool = False,
    aspect: str = "auto",
    vmin: float | None = None,
    vmax: float | None = None,
    marginal_x: str = "histogram",
    marginal_x_label_x: bool = False,
    marginal_x_label_y: str = "N",
    marginal_x_show_xticks: bool = False,
    marginal_y: str = "histogram",
    marginal_y_label_x: str = "N",
    marginal_y_label_y: bool = False,
    marginal_y_show_yticks: bool = False,
    marginal_color: str | None = None,
    figure_size: t.List[float] | None = None,
    colormap: str = "viridis",
):
    return _plot_histogram_2d(
        x,
        y,
        ds=ds,
        bins=bins,
        xlim=xlim,
        ylim=ylim,
        ax=ax,
        ax_colorbar=ax_colorbar,
        log=log,
        filename=filename,
        fig=fig,
        n_bins=n_bins,
        norm=norm,
        linear_thresh=linear_thresh,
        label_x=label_x,
        label_y=label_y,
        label_colorbar=label_colorbar,
        font_size=font_size,
        overplot_values=overplot_values,
        cmap=cmap,
        overplot_round_base=overplot_round_base,
        overplot_color=overplot_color,
        return_matrix=return_matrix,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        marginal_x=marginal_x,
        marginal_x_label_x=marginal_x_label_x,
        marginal_x_label_y=marginal_x_label_y,
        marginal_x_show_xticks=marginal_x_show_xticks,
        marginal_y=marginal_y,
        marginal_y_label_x=marginal_y_label_x,
        marginal_y_label_y=marginal_y_label_y,
        marginal_y_show_yticks=marginal_y_show_yticks,
        marginal_color=marginal_color,
        figure_size=figure_size,
        colormap=colormap,
    )


def _plot_histogram_2d(
    x: t.Union[np.ndarray, str],
    y: t.Union[np.ndarray, str],
    ds: xarray.Dataset | None = None,
    bins: t.Sequence | None = None,
    xlim: t.Sequence | None = None,
    ylim: t.Sequence | None = None,
    ax: a2.utils.constants.TYPE_MATPLOTLIB_AXES | None = None,
    ax_colorbar: a2.utils.constants.TYPE_MATPLOTLIB_AXES | None = None,
    log: t.Union[bool, t.List[bool]] = False,
    filename: t.Union[str, pathlib.Path] = None,
    fig: a2.utils.constants.TYPE_MATPLOTLIB_FIGURES | None = None,
    n_bins: t.Union[int, t.List[int]] = 60,
    norm: str | None = None,
    linear_thresh: t.Union[float, t.List[float]] = 1e-9,
    label_x: str | None = None,
    label_y: str | None = None,
    label_colorbar: str | None = None,
    font_size: int = 12,
    overplot_values: bool = False,
    cmap: str = "plasma",
    overplot_round_base: int | None = None,
    overplot_color: str = "gray",
    return_matrix: bool = False,
    aspect: str = "auto",
    vmin: float | None = None,
    vmax: float | None = None,
    marginal_x: str = "histogram",
    marginal_x_label_x: bool = False,
    marginal_x_label_y: str = "N",
    marginal_x_show_xticks: bool = False,
    marginal_y: str = "histogram",
    marginal_y_label_x: str = "N",
    marginal_y_label_y: bool = False,
    marginal_y_show_yticks: bool = False,
    marginal_color: str | None = None,
    figure_size: t.List[float] | None = None,
    colormap: str = "viridis",
) -> t.Union[plt.axes, t.Tuple[plt.axes, t.Sequence]]:
    """
    plots 2d histogram

    Log types included `False`, `log`/`True`, `symlog`.
    Norm takes care of colorbar scale, so far included: `log`
    Parameters:
    ----------
    x: values binned on x-axis
    y: values binned on y-axis
    bins: if None computed based on data provided, otherwise should
          provide [x_edges, y_edges]
    xlim: limits of x-axis and bin_edges, if None determined from x
    ylim: limits of y-axis and bin_edges, if None determined from y
    ax: matplotlib axes
    log: type of bins
    filename: plot saved to file if provided
    fig: matplotlib figure
    n_bins: number of bins
    norm: scaling of colorbar
    linear_thresh: required if using log='symlog' to indicate
                   where scale turns linear
    label_x: label of x-axis
    label_y: label of y-axis
    font_size: size of font
    overplot_values: show number of samples on plot
    overplot_round_base: Round values to be overplotted to this base int
    return_matrix: Additionally return histogram values
    marginal_x: Add additional plot on top of histogram plot ("histogram"/None)
    marginal_x_label_x: Label along x-axis for plot `marginal_x`
    marginal_x_label_y: Label along y-axis for plot `marginal_x`
    marginal_x_show_xticks: Wether to show ticks along the x-axis for plot `marginal_x`
    marginal_y: Add additional plot on top of histogram plot ("histogram"/None)
    marginal_y_label_x: Label along x-axis for plot `marginal_y`
    marginal_y_label_y: Label along y-axis for plot `marginal_y`
    marginal_y_show_yticks: Wether to show ticks along the y-axis for plot `marginal_y`
    figure_size: Size of figure (height, width), default (10, 6)
    colormap: Matplotlib colormap name
    vmin: Minimum value of colormap
    vmax: Maximum value of colormap

    Returns
    -------
    axes
    """

    if marginal_x is not None and marginal_y is not None:
        fig, axes, axes_colorbar = a2.plotting.utils_plotting.create_axes_grid(
            n_cols=2,
            n_rows=2,
            figure_size=figure_size,
            skip_row_col=[[0, 1]],
            colorbar_width=0.02,
            spacing_x=0.03,
            spacing_y=0.03,
            widths_along_x=[0.2, 0.1],
            heights_along_y=[0.1, 0.2],
            colorbar_include_row_col=[[1, 1]],
            spacing_colorbar=0.03,
        )
        ax = axes[1][0]
        ax_colorbar = axes_colorbar[1][1]
    else:
        fig, ax = a2.plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)

    def get_label(label):
        if label is not None and label:
            return label
        else:
            return ""

    if isinstance(x, str):
        if ds is not None:
            if label_x is None:
                label_x = x
            x = ds[x].values
        else:
            ValueError(f"{ds=}, if {x=} given as string, dataset required")
    if isinstance(y, str):
        if ds is not None:
            if label_y is None:
                label_y = y
            y = ds[y].values
        else:
            ValueError(f"{ds=}, if {y=} given as string, dataset required")

    x, y = _get_xy_values(x, y)

    xlim, ylim, log, n_bins, linear_thresh = _prepare_parameters(xlim, ylim, log, n_bins, linear_thresh)

    if bins is None:
        bin_edges_x = get_bin_edges(
            data=x,
            n_bins=n_bins[0],
            linear_thresh=linear_thresh[0],
            log=log[0],
            vmin=xlim[0],
            vmax=xlim[1],
        )
        bin_edges_y = get_bin_edges(
            data=y,
            n_bins=n_bins[1],
            linear_thresh=linear_thresh[1],
            log=log[1],
            vmin=ylim[0],
            vmax=ylim[1],
        )
    else:
        bin_edges_x, bin_edges_y = bins
    a2.utils.checks.validate_array(bin_edges_x)  # type: ignore
    a2.utils.checks.validate_array(bin_edges_y)  # type: ignore

    norm_object = a2.plotting.utils_plotting.get_norm(norm, vmin=vmin, vmax=vmax)
    H, bin_edges_x, bin_edges_y = np.histogram2d(x, y, bins=[np.array(bin_edges_x), np.array(bin_edges_y)])
    H_plot = H.T
    X, Y = np.meshgrid(bin_edges_x, bin_edges_y)
    plot = ax.pcolormesh(X, Y, H_plot, norm=norm_object, cmap=colormap)
    if overplot_values:
        a2.plotting.utils_plotting.overplot_values(
            H, ax, len(bin_edges_x) - 1, len(bin_edges_y) - 1, color=overplot_color, round_to_base=overplot_round_base
        )
    colorbar = plt.colorbar(plot, cax=ax_colorbar)
    ax_colorbar = colorbar.ax
    if label_colorbar is not None:
        ax_colorbar.set_ylabel(label_colorbar)

    if xlim[0] is None:
        xlim[0] = bin_edges_x[0]
    if xlim[1] is None:
        xlim[1] = bin_edges_x[-1]
    if ylim[0] is None:
        ylim[0] = bin_edges_y[0]
    if ylim[1] is None:
        ylim[1] = bin_edges_y[-1]

    a2.plotting.utils_plotting.set_x_log(ax, log[0], linear_thresh=linear_thresh[0])
    a2.plotting.utils_plotting.set_y_log(ax, log[1], linear_thresh=linear_thresh[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if label_x is not None:
        ax.set_xlabel(label_x)
    if label_y is not None:
        ax.set_ylabel(label_y)

    if marginal_x == "histogram":
        plot_histogram(
            x,
            bin_edges=bin_edges_x,
            xlim=xlim,
            ylim=None,
            ax=axes[0][0],
            log=log[0],
            fig=fig,
            n_bins=n_bins[0],
            linear_thresh=linear_thresh[0],
            label_x=get_label(marginal_x_label_x),
            label_y=get_label(marginal_x_label_y),
            font_size=font_size,
            return_plot=False,
            color=marginal_color,
        )
        if not marginal_x_show_xticks:
            a2.plotting.utils_plotting.remove_tick_labels(axes[0][0], "x")
    if marginal_y == "histogram":
        plot_histogram(
            y,
            bin_edges=bin_edges_y,
            xlim=None,
            ylim=ylim,
            ax=axes[1][1],
            log=log[1],
            fig=fig,
            n_bins=n_bins[1],
            linear_thresh=linear_thresh[1],
            label_x=get_label(marginal_y_label_x),
            label_y=get_label(marginal_y_label_y),
            font_size=font_size,
            vertical=True,
            color=marginal_color,
        )
        if not marginal_y_show_yticks:
            a2.plotting.utils_plotting.remove_tick_labels(axes[1][1], "y")
    if fig is not None and marginal_x is None and marginal_y is None:
        fig.tight_layout()
    a2.plotting.utils_plotting.save_figure(fig, filename)
    if return_matrix:
        return ax, H
    return ax


def _prepare_parameters(xlim, ylim, log, n_bins, linear_thresh):
    log = a2.plotting.utils_plotting.to_list(log)
    n_bins = a2.plotting.utils_plotting.to_list(n_bins)
    xlim = a2.plotting.utils_plotting.to_list(xlim)
    ylim = a2.plotting.utils_plotting.to_list(ylim)
    n_bins = [i + 1 if i is not None else i for i in n_bins]  # using bin edges later, where n_edges = n_bins + 1
    linear_thresh = a2.plotting.utils_plotting.to_list(linear_thresh)
    return xlim, ylim, log, n_bins, linear_thresh


def _get_xy_values(x, y):
    x = np.ndarray.flatten(x)  # type: ignore
    a2.utils.checks.validate_array(x)
    y = np.ndarray.flatten(y)  # type: ignore
    a2.utils.checks.validate_array(y)
    if x.shape != y.shape:
        raise Exception(f"x and y need to be of same shape: {np.shape(x)} != {np.shape(y)}")
    return x, y


def annotate_histogram(
    ax: plt.axes, plot, labels: list, fontproperties=None, as_label: str | None = None, fontsize: int = 8
):
    """
    annotate labels to individual bar of histogram plot
    To download emoji images refer to `download_emoji` in
    notebooks/dataset/visualize_raw_tweets.ipynb

    Parameters:
    ----------
    ax: matplotlib axes
    plot: matplotlib plot object
    labels: List of labels for individual bars
    fontproperties: matplotlib fontproperties
    as_label: Annotate the labels of an axis instead of the bars directly
    font_size: size of font

    Returns
    -------
    """
    xycoords = "data"
    zoom = 0.5
    verticalalignment = "bottom"
    for num_label, (rect1, label) in enumerate(zip(plot, labels)):
        height = rect1.get_height()
        x_y = (rect1.get_x() + rect1.get_width() / 2, height + 5)
        x_y = [int(i) for i in x_y]
        x_y_images = x_y
        if as_label == "x":
            xycoords = "axes fraction"
            zoom = 0.3
            spacing_labels = 1 / len(labels)
            x_y = [spacing_labels / 2 + num_label * spacing_labels, -0.03]
            x_y_images = x_y[:]
            ax.tick_params(labelbottom=False)
            verticalalignment = "top"
            x_y_images[1] -= 0.05
        if a2.dataset.emojis.is_emoji(label):
            png = a2.dataset.emojis.get_emoji_filename(label)
            try:
                emoj = plt.imread(f"{DATA_FOLDER.__str__()}/emoji/emoji_images/{png}")
            except FileNotFoundError:
                png_alternative = png.replace("_fe0f", "")
                emoj = plt.imread(f"{DATA_FOLDER.__str__()}/emoji/emoji_images/{png_alternative}")
            imagebox = matplotlib.offsetbox.OffsetImage(emoj, zoom=zoom)
            ab = matplotlib.offsetbox.AnnotationBbox(imagebox, x_y_images, frameon=False, xycoords=xycoords)
            ax.add_artist(ab)
        else:
            ax.annotate(
                label,
                x_y,
                ha="center",
                va=verticalalignment,
                fontsize=fontsize,
                fontproperties=fontproperties,
                rotation=90,
                xycoords=xycoords,
            )


def plot_histogram(
    x: np.ndarray,
    ds=None,
    bin_edges=None,
    xlim: t.Optional[t.Sequence] = None,
    ylim: t.Optional[t.Sequence] = None,
    ax: t.Optional[plt.axes] = None,
    log: t.Union[bool, t.List[bool]] = False,
    filename: t.Union[str, pathlib.Path] = None,
    fig: t.Optional[plt.figure] = None,
    n_bins: int = 60,
    linear_thresh: t.Optional[float] = None,
    label_x: t.Optional[str] = None,
    label_y: t.Optional[str] = None,
    font_size: int = 12,
    label: str = None,
    return_plot: bool = False,
    vertical: bool = False,
    alpha: float = 1,
    annotatations_bars: t.Optional[list] = None,
    color: str | None = None,
    min_counts: int = 0,
) -> t.Union[plt.axes, t.Tuple[plt.figure, plt.axes]]:
    """
    plots 1d histogram

    Log types included `False`, `log`/`True`, `symlog`. Norm takes care of
    colorbar scale, so far included: `log`
    Parameters:
    ----------
    x: values binned on x-axis
    bin_edges: if None computed based on data provided, otherwise should
               provide [x_edges, y_edges]
    xlim: limits of x-axis and bin_edges, if None determined from x
    ylim: limits of y-axis and bin_edges, if None determined from y
    ax: matplotlib axes
    log: type of bins
    filename: plot saved to file if provided
    fig: matplotlib figure
    n_bins: number of bins
    norm: scaling of colorbar
    linear_thresh: required if using log='symlog' to indicate where
                   scale turns linear
    label_x: label of x-axis
    label_y: label of y-axis
    font_size: size of font
    alpha: alpha value of plot
    label: Label of plot
    return_plot: return axes and bar plot object

    Returns
    -------
    axes
    """
    if all([isinstance(i, str) for i in x]):
        labels_bars, x, counts = np.unique(x, return_inverse=True, return_counts=True)
        n_bins = len(labels_bars)
        annotatations_bars = [_la if c >= min_counts else "" for _la, c in zip(labels_bars, counts)]
        xlim = [0, np.max(x)]
    log = a2.plotting.utils_plotting.to_list(log)
    xlim = a2.plotting.utils_plotting.to_list(xlim)
    ylim = a2.plotting.utils_plotting.to_list(ylim)

    if ds is not None and isinstance(x, str):
        if label_x is None:
            label_x = x
        x = ds[x].values
    x = flatten_array(x)

    fig, ax = a2.plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)
    if min_counts > 0:
        ax.axhline(min_counts)

    if bin_edges is None:
        bin_edges, linear_thresh = get_bin_edges(
            data=x,
            n_bins=n_bins,
            linear_thresh=linear_thresh,
            log=log[0],
            return_linear_thresh=True,
            vmin=xlim[0],
            vmax=xlim[1],
        )
    (
        hist,
        bin_edges,
    ) = np.histogram(x, bins=bin_edges)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    plot = plot_bar(
        bin_centers,
        hist,
        width_bars=np.diff(bin_edges),
        xlim=xlim,
        ylim=ylim,
        ax=ax,
        log=log,
        linear_thresh=linear_thresh,
        label_x=label_x,
        label_y=label_y,
        vertical=vertical,
        alpha=alpha,
        color=color,
    )
    if annotatations_bars is not None:
        annotate_histogram(ax, plot, labels=annotatations_bars, as_label="x", fontsize=font_size)
    fig.tight_layout()
    a2.plotting.utils_plotting.save_figure(fig, filename)
    print(f"{log=}")
    if return_plot:
        return ax, plot
    return ax


def plot_bar(
    bin_centers: np.ndarray,
    hist: np.ndarray,
    width_bars: np.ndarray | None = None,
    xlim: t.Optional[t.Sequence] = None,
    ylim: t.Optional[t.Sequence] = None,
    ax: t.Optional[plt.axes] = None,
    fig: a2.utils.constants.TYPE_MATPLOTLIB_FIGURES | None = None,
    log: t.Union[bool, t.List[bool]] = False,
    linear_thresh: t.Optional[float] = None,
    label_x: t.Optional[str] = None,
    label_y: t.Optional[str] = None,
    vertical: bool = False,
    alpha: float = 1,
    font_size: int = 8,
    replace_x_labels_at: t.Sequence | None = None,
    replace_x_labels_with: t.Sequence | None = None,
    replace_y_labels_at: t.Sequence | None = None,
    replace_y_labels_with: t.Sequence | None = None,
    color: str | None = None,
):
    """
    plots 1d bar plot

    Parameters:
    ----------
    bin_centers: Center position of bars
    hist: Bar values
    width_bars: Bar widths
    xlim: limits of x-axis and bin_edges, if None determined from x
    ylim: limits of y-axis and bin_edges, if None determined from y
    ax: matplotlib axes
    fig: matplotlib figure
    log: type of bins
    linear_thresh: required if using log='symlog' to indicate where
    label_x: label of x-axis
    label_y: label of y-axis
    Vertical: Plot bar plot along vertical axis instead of default horizontal
    alpha: Alpha value of bars
    font_size: Size of font
    replace_x_labels_at: Tick values of x-axis to be replaced
    replace_x_labels_with: Replacement values for tick values of x-axis
    replace_y_labels_at: Tick values of y-axis to be replaced
    replace_y_labels_with: Replacement values for tick values of y-axis
    color: Color of bars

    Returns
    -------
    plot
    """
    log = a2.plotting.utils_plotting.to_list(log)
    xlim = a2.plotting.utils_plotting.to_list(xlim)
    ylim = a2.plotting.utils_plotting.to_list(ylim)

    if ax is None:
        fig, ax = a2.plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)
    if vertical:
        plot = ax.barh(bin_centers, hist, height=width_bars, edgecolor="black", alpha=alpha, color=color)
    else:
        plot = ax.bar(bin_centers, hist, width=width_bars, edgecolor="black", alpha=alpha, color=color)

    a2.plotting.utils_plotting.set_x_log(ax, log[0], linear_thresh=linear_thresh)
    a2.plotting.utils_plotting.set_y_log(ax, log[1], linear_thresh=linear_thresh)
    a2.plotting.utils_plotting.set_axis_tick_labels(ax, replace_x_labels_at, replace_x_labels_with, axis="x")
    a2.plotting.utils_plotting.set_axis_tick_labels(ax, replace_y_labels_at, replace_y_labels_with, axis="y")
    if label_x is not None:
        ax.set_xlabel(label_x)
    if label_y is not None:
        ax.set_ylabel(label_y)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return plot


def flatten_array(x):
    x = np.asarray(x)
    x = np.ndarray.flatten(x)
    return x


def get_bin_edges(
    vmin: t.Optional[float] = None,
    vmax: t.Optional[float] = None,
    linear_thresh: t.Optional[float] = None,
    n_bins: int = 60,
    data: t.Optional[np.ndarray] = None,
    log: t.Union[str, bool] = False,
    return_linear_thresh: bool = False,
) -> t.Union[t.Tuple[np.ndarray, t.Optional[float]], np.ndarray]:
    """
    returns bin edges for plots

    Log types included `False`, `log`/`True`, `symlog`
    Parameters:
    ----------
    vmin: minimum value of data
    vmax: maximum value of data
    linear_thresh: threshold below which bins are linear to include zero values
    n_bins: number of bins for logarithmic part of bins
    data: if provided used to compute `vmin` and `vmax`
    log: type of bins

    Returns
    -------
    bin edges
    """
    if data is not None and vmin is None:
        vmin = data.min()
    if data is not None and vmax is None:
        vmax = data.max()
    if vmin is None or vmax is None:
        raise Exception(f"Need to specify vmin {vmin} and {vmax} or provide data: {data}!")
    if not log:
        bins = np.linspace(vmin, vmax, n_bins)
    elif log == "symlog":
        if linear_thresh is None:
            abs_max = abs(vmax)
            abs_min = abs(vmin)
            linear_thresh = abs_min if abs_min < abs_max or abs_min == 0 else abs_max if abs_max != 0 else abs_min
            logging.info(f"Setting: linear_thresh: {linear_thresh} with vmin: {vmin}" " and vmax: {vmax}!")
        bins = _get_bin_edges_symlog(vmin, vmax, linear_thresh, n_bins=n_bins)
    else:
        bins = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n_bins)

    if return_linear_thresh:
        return bins, linear_thresh
    else:
        return bins


def _get_bin_edges_symlog(
    vmin: float,
    vmax: float,
    linear_thresh: float,
    n_bins: int = 60,
    n_bins_linear: int = 10,
) -> np.ndarray:
    """
    returns symmetrical logarithmic bins

    Bins have same absolute vmin, vmax if vmin is negative
    Parameters:
    ----------
    vmin: minimum value of data
    vmax: maximum value of data
    linear_thresh: threshold below which bins are linear to include zero values
    n_bins: number of bins for logarithmic part of bins
    n_bins_linear: number of bins for linear part of bins

    Returns
    -------
    symmetrical bin edges
    """
    if isinstance(vmin, np.datetime64) or vmin > 0:
        bins = 10 ** np.linspace(np.log10(vmin), np.log10(vmax), n_bins)
    elif vmin == 0:
        bins = np.hstack(
            (
                np.linspace(0, linear_thresh, n_bins_linear),
                10 ** np.linspace(np.log10(linear_thresh), np.log10(vmax)),
            )
        )
    else:
        bins = np.hstack(
            (
                -(
                    10
                    ** np.linspace(
                        np.log10(vmax),
                        np.log10(linear_thresh),
                        n_bins // 2,
                        endpoint=False,
                    )
                ),
                np.linspace(-linear_thresh, linear_thresh, n_bins_linear, endpoint=False),
                10 ** np.linspace(np.log10(linear_thresh), np.log10(vmax), n_bins // 2),
            )
        )
    return bins
