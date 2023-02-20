import pathlib
import typing as t

import a2.plotting.axes_utils
import a2.plotting.utils_plotting
import matplotlib.pyplot as plt
import numpy as np

try:
    import sklearn.metrics
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"{e}\n" f"Optional dependencies missing, install via" " `poetry install --extras=train`")


def plot_prediction_certainty(
    truth: np.ndarray,
    prediction_probabilities: np.ndarray,
    n_true_labels: int = 2,
    n_bins: int = 10,
    label_x: str = "True label",
    label_y: str = "Prediction probability for 'raining'",
    label_colorbar: str = "Number of Tweets",
    filename: t.Union[str, pathlib.Path] = None,
    font_size: int = 12,
    vmin=None,
    vmax=None,
    cmap="viridis",
    figure_size: t.Sequence = None,
    return_matrix: bool = False,
    ax: plt.axes = None,
    fig: plt.figure = None,
    **kwargs,
) -> t.Union[plt.Axes, t.Tuple[plt.Axes, np.ndarray]]:
    """
    Compute 2d histogram of true labels (x-axis) and
    binned prediction probabilities (y-axis)

    Parameters:
    ----------
    truth: true labels
    prediction_probabilities: probabilities for labels
    n_true_labels: number of different labels
    n_bins: number of bins for prediction probabilities
    label_x: label of x-axis
    label_y: label of y-axis
    label_colorbar: label of colorbar axis
    filename: save figure to this file, not saved if `None`
    font_size: font size of all labels of plot
    figure_size: size of figure [width, height]
    vmin: Minimum value for colorbar
    vmax: Maximum value for colorbar
    return_matrix: return 2d histogram matrix and axes

    Returns
    -------
    axes of plot
    """
    if figure_size is None:
        figure_size = [8, 8]
    ax, H = a2.plotting.histograms.plot_histogram_2d(
        truth,
        prediction_probabilities,
        n_bins=[n_true_labels, n_bins],
        label_x=label_x,
        label_colorbar=label_colorbar,
        label_y=label_y,
        overplot_values=True,
        return_matrix=True,
        filename=filename,
        fig=fig,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        font_size=font_size,
        **kwargs,
    )
    if return_matrix:
        return ax, H
    return ax


def classification_report(
    truth: t.Sequence,
    prediction: t.Sequence,
    target_names: t.Optional[t.Sequence[str]] = None,
    output_dict: bool = True,
) -> t.Union[str, t.Mapping]:
    """
    Compute classification report, returns precision, recall, f1-score
    and respective averages

    report returned as dict or string (see `output_dict`)
    Parameters:
    ----------
    truth: true labels
    prediction: predicted labels
    target_names: names of labels
    output_dict: return report as dictionary rather than string

    Returns
    -------
    report
    """
    if target_names is None:
        target_names = ["not raining", "raining"]
    report = sklearn.metrics.classification_report(
        truth,
        prediction,
        target_names=target_names,
        output_dict=output_dict,
    )
    return report


def check_prediction(
    truth: t.Sequence,
    prediction: t.Sequence,
    output_dict: bool = True,
    filename: t.Union[str, pathlib.Path] | None = None,
):
    plot_confusion_matrix(
        truth,
        prediction,
        filename=filename,
        overplot_round_base=2,
    )

    report = classification_report(
        truth,
        prediction,
        target_names=["not raining", "raining"],
        output_dict=output_dict,
    )
    return report


def plot_confusion_matrix(
    truth: t.Sequence,
    prediction: t.Sequence,
    filename: t.Union[str, pathlib.Path] | None = None,
    normalize: str = "all",
    font_size: int = 14,
    figure_size: t.Sequence | None = None,
    label_x: str | None = "Predicted label",
    label_y: str | None = "True label",
    vmin: float | None = None,
    vmax: float | None = None,
    text_color: str | None = "firebrick",
    colormap: str | None = "Blues",
    overplot_round_base: int | None = None,
    tick_labels_x: tuple | None = ("not raining", "raining"),
    tick_labels_y: tuple | None = ("raining", "not raining"),
    colorbar_label: str | None = "Fraction",
    colorbar_width: float = 0.05,
    ax: object | None = None,
    ax_colorbar: object | None = None,
):
    """
    Computes classification report and plots confusion matrix

    Parameters:
    ----------
    truth: True labels
    prediction: Predicted labels
    filename: Save figure to this file, not saved if `None`
    normalize: Normalize style of confusion matrix
    output_dict: Return classification report as dictionary
    font_size: Font size of all labels of plot
    figure_size: Size of figure [width, height], default [6, 6]

    Returns
    -------
    classification report
    """
    if figure_size is None:
        figure_size = [6, 6]
    cm = sklearn.metrics.confusion_matrix(truth, prediction, normalize=normalize)
    print(f"{cm=}")
    if ax is None:
        fig, ax, ax_colorbar = a2.plotting.utils_plotting.create_axes_grid(
            1, 1, figure_size=figure_size, unravel=True, colorbar_off=False, colorbar_width=colorbar_width
        )
    else:
        fig = ax.get_figure()
    norm = a2.plotting.utils_plotting.get_norm("linear", vmin=vmin, vmax=vmax)
    xedges = [0, 0.5, 1]
    yedges = [0, 0.5, 1]
    cm = cm[::-1, :].T
    mesh = ax.pcolormesh(xedges, yedges, cm.T, norm=norm, cmap=colormap)
    cbar = plt.colorbar(mesh, cax=ax_colorbar, orientation="vertical")
    a2.plotting.axes_utils.set_colorbar(cbar.ax, label_y=colorbar_label, fontsize=font_size)
    a2.plotting.utils_plotting.overplot_values(
        np.array(cm), ax, 2, 2, color=text_color, round_to_base=overplot_round_base, font_size=font_size
    )
    a2.plotting.utils_plotting.set_axis_tick_labels(ax, [0.25, 0.75], tick_labels_x, "x")
    a2.plotting.utils_plotting.set_axis_tick_labels(ax, [0.25, 0.75], tick_labels_y, "y")
    a2.plotting.axes_utils.set_axes(
        ax,
        xlim=[0, 1],
        ylim=[0, 1],
        fontsize=font_size,
        label_x=label_x,
        label_y=label_y,
    )

    a2.plotting.utils_plotting.save_figure(fig, filename)
    return fig


def plot_roc(
    truth: t.Sequence,
    prediction_probabilities: t.Sequence,
    filename: t.Union[str, pathlib.Path] = None,
    font_size: int = 12,
    figure_size: t.Sequence = None,
    ax: object | None = None,
    fig: object | None = None,
    label_x: str | None = "False Positive Rate",
    label_y: str | None = "True Positive Rate",
    return_rates: bool = False,
) -> t.Union[plt.Axes, t.Tuple[plt.Axes, t.Sequence, t.Sequence]]:
    """
    Plots Receiver Operating Characteristic curve

    Can optionally return true positive and false positive rate
    (if `return_rates=True`)
    Parameters:
    ----------
    truth: true labels
    prediction_probabilities: probabilities for labels
    filename: save figure to this file, not saved if `None`
    font_size: font size of all labels of plot
    figure_size: size of figure [width, height], default: [6, 6]
    return_rates: return true positive and false positive rate

    Returns
    -------
    axes of plot
    """
    if figure_size is None:
        figure_size = [6, 6]
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(
        truth, prediction_probabilities, drop_intermediate=False
    )
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    fig, ax = a2.plotting.utils_plotting.create_figure_axes(
        fig=fig, ax=ax, figure_size=figure_size, font_size=font_size
    )
    lw = 2
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        lw=lw,
        label="ROC curve (area = %.04f)" % roc_auc,
    )

    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    a2.plotting.axes_utils.set_axes(
        ax,
        xlim=[0, 1],
        ylim=[0, 1],
        fontsize=font_size,
        label_x=label_x,
        label_y=label_y,
    )

    ax.legend(loc="lower right")
    a2.plotting.utils_plotting.save_figure(fig, filename)
    if return_rates:
        return ax, true_positive_rate, false_positive_rate
    return ax
