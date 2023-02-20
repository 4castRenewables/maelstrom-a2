import logging
import pathlib
import typing as t

import a2.dataset
import a2.plotting.utils_plotting
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_2d(
    x: np.ndarray,
    y: np.ndarray,
    bins: t.Optional[t.Sequence] = None,
    xlim: t.Optional[t.Sequence] = None,
    ylim: t.Optional[t.Sequence] = None,
    ax: t.Optional[plt.axes] = None,
    log: t.Union[bool, t.List[bool]] = False,
    filename: t.Union[str, pathlib.Path] = None,
    fig: t.Optional[plt.figure] = None,
    n_bins: t.Union[int, t.List[int]] = 60,
    norm: t.Optional[str] = None,
    linear_thresh: t.Union[float, t.List[float]] = 1e-9,
    label_x: t.Optional[str] = None,
    label_y: t.Optional[str] = None,
    label_colorbar: t.Optional[str] = None,
    font_size: int = 12,
    overplot_values: bool = False,
    return_matrix: bool = False,
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

    Returns
    -------
    axes
    """
    x = x.flatten()
    y = y.flatten()

    if np.sum(np.isnan(x.astype(float))) > 0:
        raise ValueError("found nan-values in x")
    if np.sum(np.isnan(y.astype(float))) > 0:
        raise ValueError("found nan-values in y")
    if x.shape != y.shape:
        raise Exception(f"x and y need to be of same shape: {np.shape(x)} != {np.shape(y)}")
    fig, ax = a2.plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)
    if not isinstance(log, list):
        log = [log, log]
    if not isinstance(n_bins, list):
        n_bins = [n_bins, n_bins]
    if not isinstance(xlim, list):
        xlim = [xlim, xlim]
    if not isinstance(ylim, list):
        ylim = [ylim, ylim]
    n_bins = [x + 1 if x is not None else x for x in n_bins]  # using bin edges later, where n_edges = n_bins + 1
    if not isinstance(linear_thresh, list):
        linear_thresh = [linear_thresh, linear_thresh]
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
    norm_object = a2.plotting.utils_plotting.get_norm(norm)
    H, bin_edges_x, bin_edges_y = np.histogram2d(x, y, bins=[np.array(bin_edges_x), np.array(bin_edges_y)])
    H_plot = H.T
    X, Y = np.meshgrid(bin_edges_x, bin_edges_y)
    plot = ax.pcolormesh(X, Y, H_plot, norm=norm_object)
    if overplot_values:
        a2.plotting.utils_plotting.overplot_values(H, ax, len(bin_edges_x) - 1, len(bin_edges_y) - 1)
    colorbar = fig.colorbar(plot)
    ax_colorbar = colorbar.ax
    if label_colorbar is not None:
        ax_colorbar.set_ylabel(label_colorbar)

    a2.plotting.utils_plotting.set_x_log(ax, log[0], linear_thresh=linear_thresh[0])
    a2.plotting.utils_plotting.set_y_log(ax, log[1], linear_thresh=linear_thresh[1])
    if label_x is not None:
        ax.set_xlabel(label_x)
    if label_y is not None:
        ax.set_ylabel(label_y)
    fig.tight_layout()
    a2.plotting.utils_plotting.save_figure(fig, filename)
    if return_matrix:
        return ax, H
    return ax


def plot_histogram(
    x: np.ndarray,
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
    return_plot: bool = False,
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
    return_plot: return axes and bar plot object

    Returns
    -------
    axes
    """
    x = x.flatten()
    if not isinstance(log, list):
        log = [log, log]
    if not isinstance(xlim, list):
        xlim = [xlim, xlim]
    if not isinstance(ylim, list):
        ylim = [ylim, ylim]

    fig, ax = a2.plotting.utils_plotting.create_figure_axes(fig=fig, ax=ax, font_size=font_size)
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
    plot = ax.bar(bin_centers, hist, width=np.diff(bin_edges), edgecolor="black")

    a2.plotting.utils_plotting.set_x_log(ax, log[0], linear_thresh=linear_thresh)
    a2.plotting.utils_plotting.set_y_log(ax, log[1], linear_thresh=linear_thresh)

    if label_x is not None:
        ax.set_xlabel(label_x)
    if label_y is not None:
        ax.set_ylabel(label_y)
    ax.set_ylim(ylim)
    fig.tight_layout()
    a2.plotting.utils_plotting.save_figure(fig, filename)

    if return_plot:
        return ax, plot
    return ax


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
