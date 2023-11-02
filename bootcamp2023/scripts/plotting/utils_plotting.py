import logging
import pathlib
import typing as t

import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt


def set_x_log(ax: plt.axes, log: bool = False, linear_thresh: t.Optional[float] = None) -> plt.axes:
    """
    Sets scale of x-axis

    Parameters:
    ----------
    ax: Matplotlib axes
    log: Type of bins. Log types included `False`, `log`/`True`, `symlog`
    linear_thresh: Threshold below which bins are linear to include zero values

    Returns
    -------
    axes
    """
    if log == "symlog":
        if linear_thresh is None:
            raise ValueError(f"If log=='symlog', setting linear_thresh: " f"{linear_thresh} required!")
        ax.set_xscale("symlog", linthresh=linear_thresh)
    elif log:
        ax.set_xscale("log")
    return ax


def set_y_log(ax: plt.axes, log: bool = False, linear_thresh: t.Optional[float] = None) -> plt.axes:
    """
    Sets scale of y-axis

    Parameters:
    ----------
    ax: Matplotlib axes
    log: Type of bins. Log types included `False`, `log`/`True`, `symlog`
    linear_thresh: Threshold below which bins are linear to include zero values

    Returns
    -------
    axes
    """
    if log == "symlog":
        if linear_thresh is None:
            raise ValueError("If log=='symlog', " f"setting linear_thresh: {linear_thresh} required!")
        ax.set_yscale("symlog", linthresh=linear_thresh)
    elif log:
        ax.set_yscale("log")
    return ax


def get_norm(
    norm: t.Optional[str] = None,
    vmin: t.Optional[float] = None,
    vmax: t.Optional[float] = None,
) -> t.Union[matplotlib.colors.LogNorm, matplotlib.colors.Normalize]:
    """
    Returns matplotlib norm used for normalizing matplotlib's colorbars

    Parameters:
    ----------
    norm: Normalization type: `log` or `linear`/None
    vmin: Minimum
    vmax: Maximum

    Returns
    -------
    Norm object from matplotlib.colors
    """
    if norm == "log":
        return matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm == "linear" or norm is None:
        return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise Exception(f"Norm: {norm} unknown!")


def create_figure_axes(
    fig: t.Optional[plt.figure] = None,
    ax: t.Optional[plt.axes] = None,
    figure_size: t.Sequence = None,
    font_size: int = 10,
) -> t.Tuple[plt.figure, plt.axes]:
    """
    Creates figure with axes and sets font size

    Parameters:
    ----------
    fig: Figure if available
    ax: Axes if available
    figure_size: Size of figure (height, width), default (10, 6)
    font_size: Font size

    Returns
    -------
    matplotlib figure and axes
    """
    if figure_size is None:
        figure_size = (10, 6)
    font = {"family": "DejaVu Sans", "weight": "normal", "size": font_size}

    matplotlib.rc("font", **font)
    if fig is None and ax is None:
        fig = plt.figure(figsize=figure_size)
        ax = plt.gca()
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    return fig, ax


def overplot_values(H, ax, size_x, size_y, color="black"):
    """
    Overplot values on plot based on 2d matrix whose values are
    plotted in equidistant intervals

    Parameters:
    ----------
    H: Matrix (2d) whose values will be overplot
    ax: Matplotlib axes
    size_x: Size of figure (height, width)
    size_y: Font size

    Returns
    -------
    matplotlib figure and axes
    """
    x_start = 0
    x_end = 1
    y_start = 0
    y_end = 1
    jump_x = (x_end - x_start) / (2.0 * size_x)
    jump_y = (y_end - y_start) / (2.0 * size_y)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size_x, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size_y, endpoint=False)
    for x_index, x in enumerate(x_positions):
        for y_index, y in enumerate(y_positions):
            label = H[x_index, y_index]
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(
                text_x,
                text_y,
                label,
                color=color,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )


def set_axis_tick_labels(ax: plt.axes, values: t.Sequence[float], labels: t.Sequence, axis: str = "x") -> plt.axes:
    """
    Set new tick labels for given values

    Parameters:
    ----------
    ax: Matplotlib axes
    values: Values for corresponding new labels
    labels: New labels
    axis: Which axis to set, i.e. 'x' or 'y'

    Returns
    -------
    axes
    """
    if axis == "x":
        ax.set_xticks(values)
        ax.set_xticklabels(labels)
    elif axis == "y":
        ax.set_yticks(values)
        ax.set_yticklabels(labels)
    return ax


def save_figure(fig: plt.figure, filename: t.Union[str, pathlib.Path] = None) -> None:
    if filename is not None:
        logging.info(f"... saving {filename}")
        fig.savefig(filename, bbox_inches="tight")


def _get_values_from_bar_object(
    bar_object: matplotlib.container.BarContainer,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Debug function to obtain plotted values from a bar plot object

    Returns X, Y and H values of original plot
    Parameters:
    ----------
    bar_object: Bar plot object

    Returns
    -------
    X, Y, H
    """
    X = []
    Y = []
    H = []
    for b in bar_object:
        x, y = b.get_xy()
        X.append(x)
        Y.append(y)
        H.append(b.get_height())
    return np.array(X), np.array(Y), np.array(H)


def _get_values_from_pcolormesh_object(
    pc_object: matplotlib.collections.QuadMesh,
) -> np.ndarray:
    """
    Debug function to obtain plotted values from pcolormesh object

    Returns flattened H values
    Parameters:
    ----------
    pc_object: Pcolormesh plot object

    Returns
    -------
    Flattened matrix values
    """
    return pc_object.get_array().data
