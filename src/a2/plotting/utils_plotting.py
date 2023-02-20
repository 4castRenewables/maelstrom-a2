import collections
import logging
import os
import pathlib
import typing as t

import a2.utils.utils
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


def _create_figure(figure_size):
    return plt.figure(figsize=figure_size)


def set_font(font_size=10):
    font = {"family": "DejaVu Sans", "weight": "normal", "size": font_size}
    matplotlib.rc("font", **font)


def create_figure_axes(
    fig: t.Optional[plt.figure] = None,
    ax: t.Optional[plt.axes] = None,
    figure_size: t.Sequence = None,
    font_size: t.Optional[int] = 10,
    aspect: str = "auto",
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
    if font_size is not None:
        set_font(font_size=font_size)
    if figure_size is None:
        figure_size = (10, 6)
    if fig is None and ax is None:
        fig = plt.figure(figsize=figure_size)
        ax = plt.gca()
        ax.set_aspect(aspect)
    if ax is None:
        ax = plt.gca()
        ax.set_aspect(aspect)
    if fig is None:
        fig = ax.get_figure()
    return fig, ax


def create_axes_grid(
    n_cols: int,
    n_rows: int,
    figure_size: t.Optional[t.List[float]] = None,
    widths_along_x: t.Optional[t.List] = None,
    heights_along_y: t.Optional[t.List] = None,
    top: float = 0.05,
    bottom: float = 0.05,
    right: float = 0.05,
    left: float = 0.05,
    spacing_x: float = 0.05,
    spacing_y: float = 0.05,
    spacing_colorbar: float = 0.04,
    colorbar_width: float = 0.07,
    colorbar_skip_row_col: t.Optional[t.List] = None,
    colorbar_include_row_col: t.Optional[t.List] = None,
    colorbar_off: bool = True,
    skip_cols: t.Optional[t.List] = None,
    skip_rows: t.Optional[t.List] = None,
    skip_row_col: t.Optional[t.List] = None,
    unravel: bool = False,
) -> t.Tuple[plt.figure, np.ndarray, np.ndarray]:
    """
    Create figure with grid of axes. Units are scaled to normalized figure size (max value: 1)
    unless specified otherwise.
    See also https://docs.google.com/presentation/d/1Ec-000rszefjCsv_sgUO62eGyT0-YbYzbk1aLQkU2gM/edit?usp=sharing
    Parameters:
    ----------
    n_cols: Number of columns
    n_rows: Number of rows
    figure_size: Size of figure
    widths_along_x: Normalized width ratios along horizontal direction
    heights_along_y: Normalized height ratios along vertical direction
    top: Offset of axes grid from the top
    bottom: Offset of axes grid from the bottom
    right: Offset of axes grid from the right
    left: Offset of axes grid from the left
    spacing_x: Spacing between axes along horizontal direction
    spacing_y: Spacing between axes along vertical direction
    spacing_colorbar: Spacing between axes and colorbar axes
    colorbar_width: Width of the colorbars
    colorbar_skip_row_col: (row, col) pairs of colorbars to be skipped
    colorbar_include_row_col: (row, col) pairs of colorbars to be plotted
    colorbar_off: No colorbars plotted
    skip_cols: Leave these columns blank
    skip_rows: Leave these rows blank
    skip_row_col: (row, col) pairs of colorbars for which no axes plotted

    Returns
    -------
    Figure, List of axes, List of colorbar axes
    """
    if figure_size is None:
        figure_size = [10.0, 6.0]
    if colorbar_off and colorbar_include_row_col is None and colorbar_skip_row_col is None:
        colorbar_skip_row_col = [[j, i] for i in range(n_cols) for j in range(n_rows)]
    fig = _create_figure(figure_size)
    a2.utils.utils.assert_same_type(skip_row_col, [[]])
    a2.utils.utils.assert_same_type(colorbar_skip_row_col, [[]])
    a2.utils.utils.assert_same_type(colorbar_include_row_col, [[]])
    a2.utils.utils.assert_shape(widths_along_x, (n_cols,), "widths_along_x")
    a2.utils.utils.assert_shape(heights_along_y, (n_rows,), "heights_along_y")

    if skip_rows is None:
        skip_rows = []
    if skip_cols is None:
        skip_cols = []
    if skip_row_col is None:
        skip_row_col = []

    height_total = 1 - (n_rows - 1) * spacing_y - top - bottom
    if heights_along_y is not None:
        heights_along_y = [x / sum(heights_along_y) * height_total for x in heights_along_y]
        axes_heights = np.array([[x for i in range(n_cols)] for x in heights_along_y])
    else:
        height = height_total / n_rows
        axes_heights = np.array([[height for i in range(n_cols)] for j in range(n_rows)])

    colorbar_heights = np.zeros((n_rows, n_cols))
    colorbar_widths = np.zeros((n_rows, n_cols))
    spacings_colorbar = np.zeros((n_rows, n_cols))
    skip_col_colorbar = []
    if colorbar_include_row_col is not None:
        include_col_colorbar = [x[1] for x in colorbar_include_row_col]
        skip_col_colorbar = [x for x in range(n_cols) if x not in include_col_colorbar]
    if colorbar_skip_row_col is not None:
        skip_col = [x[1] for x in colorbar_skip_row_col]
        counts = collections.Counter(skip_col)
        for k, v in counts.items():
            if v == n_rows:
                skip_col_colorbar.append(k)
    for i_col in range(n_cols):
        for i_row in range(n_rows):
            if i_col in skip_col_colorbar:
                continue
            colorbar_heights[i_row, i_col] = axes_heights[i_row, i_col]
            colorbar_widths[i_row, i_col] = colorbar_width
            spacings_colorbar[i_row, i_col] = spacing_colorbar
    width_total = (
        1
        - (n_cols - 1) * spacing_x
        - left
        - right
        - max(sum(colorbar_widths[i, :]) for i in range(n_rows))
        - max(sum(spacings_colorbar[i, :]) for i in range(n_rows))
    )
    if widths_along_x is not None:
        widths_along_x = [x / sum(widths_along_x) * width_total for x in widths_along_x]
        axes_widths = np.array([widths_along_x for x in range(n_rows)])
    else:
        width = width_total / n_cols
        axes_widths = np.array([[width for i in range(n_cols)] for j in range(n_rows)])

    axes = [[None for i in range(n_cols)] for j in range(n_rows)]
    axes_colorbar = [[None for i in range(n_cols)] for j in range(n_rows)]
    for i_col in range(n_cols):
        if i_col in skip_cols:
            continue
        for i_row in range(n_rows):
            if i_row in skip_rows:
                continue
            if [i_row, i_col] in skip_row_col:
                continue
            axes_left, axes_bottom, axes_width, axes_height = _determine_axes_dimensions(
                n_rows,
                bottom,
                left,
                spacing_x,
                spacing_y,
                axes_heights,
                colorbar_widths,
                spacings_colorbar,
                i_col,
                i_row,
                axes_widths,
            )
            axes[i_row][i_col] = plt.axes([axes_left, axes_bottom, axes_width, axes_height])
            if colorbar_skip_row_col is not None and [i_row, i_col] in colorbar_skip_row_col:
                continue
            if colorbar_include_row_col is not None and [i_row, i_col] not in colorbar_include_row_col:
                continue
            colorbar_width, colorbar_left, colorbar_bottom, colorbar_height = _determine_colorbar_axes_dimensions(
                colorbar_heights, colorbar_widths, spacings_colorbar, i_col, i_row, axes_left, axes_bottom, axes_width
            )
            axes_colorbar[i_row][i_col] = plt.axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
    axes, axes_colorbar = np.array(axes, dtype=object), np.array(axes_colorbar, dtype=object)
    print(f"{axes=}")
    if unravel:
        axes = a2.utils.utils.flatten_list(axes)
        axes_colorbar = a2.utils.utils.flatten_list(axes_colorbar)
        if n_cols * n_rows == 1:
            axes = axes[0]
            axes_colorbar = axes_colorbar[0]
    print(f"{axes=}")
    return fig, axes, axes_colorbar


def _determine_colorbar_axes_dimensions(
    colorbar_heights, colorbar_widths, spacings_colorbar, i_col, i_row, axes_left, axes_bottom, axes_width
):
    colorbar_left = axes_left + axes_width + spacings_colorbar[i_row][i_col]
    colorbar_bottom = axes_bottom
    colorbar_width = colorbar_widths[i_row, i_col]
    colorbar_height = colorbar_heights[i_row, i_col]
    return colorbar_width, colorbar_left, colorbar_bottom, colorbar_height


def _determine_axes_dimensions(
    n_rows,
    bottom,
    left,
    spacing_x,
    spacing_y,
    axes_heights,
    colorbar_widths,
    spacings_colorbar,
    i_col,
    i_row,
    axes_widths,
):
    axes_heights_cumulative = axes_heights[slice(i_row + 1, None), i_col]
    axes_left = (
        left
        + sum(axes_widths[i_row, 0:i_col])
        + sum(colorbar_widths[i_row, 0:i_col])
        + sum(spacings_colorbar[i_row, 0:i_col])
        + i_col * spacing_x
    )
    axes_bottom = bottom + sum(axes_heights_cumulative) + (n_rows - i_row - 1) * spacing_y
    axes_width = axes_widths[i_row, i_col]
    axes_height = axes_heights[i_row, i_col]
    return axes_left, axes_bottom, axes_width, axes_height


def overplot_values(
    H: np.ndarray,
    ax: plt.axes,
    size_x: int,
    size_y: int,
    color: str = "black",
    round_to_base: int | None = None,
    font_size: int | None = None,
):
    """
    Overplot values on plot based on 2d matrix whose values are
    plotted in equidistant intervals

    Parameters:
    ----------
    H: Matrix (2d) whose values will be overplot
    ax: Matplotlib axes
    size_x: Number of elements along x-axis
    size_y: Number of elements along y-axis

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
    H_processed = H.copy()
    if round_to_base is not None:
        H_processed = np.round(H, round_to_base)
        if round_to_base < 0:
            H_processed = np.array(H_processed, int)
    for x_index, x in enumerate(x_positions):
        for y_index, y in enumerate(y_positions):
            label = H_processed[x_index, y_index]
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
                fontsize=font_size,
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
    if values is None or labels is None:
        return ax
    if axis == "x":
        ax.set_xticks(values)
        ax.set_xticklabels(labels)
    elif axis == "y":
        ax.set_yticks(values)
        ax.set_yticklabels(labels)
    return ax


def remove_tick_labels(ax: plt.axes, axis: str = "x"):
    """Remove ticks and tick labels for specified axis"""
    set_axis_tick_labels(ax=ax, values=[], labels=[], axis=axis)


def save_figure(fig: plt.figure, filename: t.Union[str, pathlib.Path] = None, dpi: int = 450) -> None:
    """Save figure to filename"""
    if filename is not None:
        logging.info(f"... saving {filename}")
        folder = os.path.split(filename.__str__())[0]
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)


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


def to_list(x, n=2):
    if not (isinstance(x, list) or isinstance(x, tuple)):
        return [x] * n
    if len(x) != n:
        raise ValueError(f"{x} doesn't have expected length {n=}")
    return x
