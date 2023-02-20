import functools
import io
import typing as t
from copy import deepcopy

import a2.plotting.utils_plotting
import a2.utils
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def parallel_plot(
    plot_fn: t.Callable,
    data: t.Iterable,
    grid_shape: tuple,
    figsize: t.Optional[t.List[float]] = None,
    aspect: str = "auto",
    kwargs=None,
    height_ratios: list = None,
    axes_backend: str = "matplotlib",
    processes: int = -1,
):
    """
    Plot figure in parallel.

    Parameters:
    ----------
    plot_fn: Plotting function that will be applied to all axes
    data: Zipped data over all axes, expected shape [number of axes]
    grid_shape: Shape of axes grid making up the figure
    figsize: Size of the figure
    aspect: Aspect ratio of the axes ("auto"/"equal")
    kwargs: Key word arguments for plotting function
    height_ratios: Height ratios of axes along the rows of the figure
    axes_backend: How axes are generated ("default"/"matplotlib")
    processes: Number of plots generated in parallel

    Returns
    -------
    Figure
    """

    if axes_backend == "default":
        fig, axes, axes_colorbar = a2.plotting.utils_plotting.create_axes_grid(
            n_cols=grid_shape[1],
            n_rows=grid_shape[0],
            figure_size=figsize,
            colorbar_width=0.02,
            spacing_x=0.03,
            spacing_y=0.03,
        )
    elif axes_backend == "matplotlib":
        fig, axes = plt.subplots(
            *grid_shape, figsize=figsize, height_ratios=height_ratios, subplot_kw={"aspect": aspect}, squeeze=False
        )
    else:
        raise ValueError(f"{axes_backend=} not available")
    if kwargs is None:
        kwargs = {}
    rastered_images = a2.utils.utils.parallelize(
        functools.partial(_parallel_plot_worker, function_to_plot=plot_fn, kwargs=kwargs),
        zip(data),
        processes=processes,
    )
    for ax, rastered in zip(axes.ravel(), rastered_images):
        ax.imshow(rastered, aspect=aspect)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(hspace=0, wspace=0)
    return fig


def _parallel_plot_worker(data, function_to_plot: t.Callable, kwargs):
    """Helper function for parallelization, where all plots of individual axes are saved in a buffer
    and plotted in the main function as an image (`plt.imshow`)"""
    fig = plt.figure()
    matplotlib.font_manager._get_font.cache_clear()  # necessary to reduce text corruption artifacts
    axes = plt.axes()

    function_to_plot(*data, fig=fig, ax=axes, **kwargs)
    pil_img = rasterize(fig)
    plt.close()

    return pil_img


def rasterize(fig: plt.figure):
    """Helper function for rasterization"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=400)
    buf.seek(0)
    pil_img = deepcopy(Image.open(buf))
    buf.close()

    return pil_img
