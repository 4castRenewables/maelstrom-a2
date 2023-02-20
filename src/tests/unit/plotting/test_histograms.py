import pathlib

import a2.plotting.utils_plotting
import matplotlib.pyplot as plt
import numpy as np
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


def test_plot_histogram_symlog(test_gaussian_random_samples, test_symlog_histogram_results):
    x_expected, h_expected = test_symlog_histogram_results.T
    _, plot = a2.plotting.histograms.plot_histogram(
        test_gaussian_random_samples,
        return_plot=True,
        log=["symlog", "log"],
        linear_thresh=1e-1,
        xlim=[-1, 1],
        n_bins=20,
    )
    x, _, h = a2.plotting.utils_plotting._get_values_from_bar_object(plot)
    assert np.allclose(x, x_expected)
    assert np.allclose(h, h_expected)


def test_plot_histogram_symlog_linthresh(test_gaussian_random_samples, test_symlog_linthresh_histogram_results):
    x_expected, h_expected = test_symlog_linthresh_histogram_results.T
    _, plot = a2.plotting.histograms.plot_histogram(
        test_gaussian_random_samples,
        return_plot=True,
        log=["symlog", "symlog"],
        xlim=[-1, 1],
        n_bins=20,
    )
    x, _, h = a2.plotting.utils_plotting._get_values_from_bar_object(plot)
    assert np.allclose(x, x_expected)
    assert np.allclose(h, h_expected)


def test_plot_histogram_log(tmp_path, test_gaussian_random_samples, test_log_histogram_results):
    directory = tmp_path / "test_plot_histogram_log/"
    directory.mkdir()
    x_expected, h_expected = test_log_histogram_results.T

    _, plot = a2.plotting.histograms.plot_histogram(
        test_gaussian_random_samples,
        return_plot=True,
        log=["log", False],
        xlim=[1e-2, 1],
        n_bins=20,
        filename=directory / "test.pdf",
    )
    x, _, h = a2.plotting.utils_plotting._get_values_from_bar_object(plot)

    assert np.allclose(x, x_expected)
    assert np.allclose(h, h_expected)


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_annotate_histogram(fake_keywords):
    occurence = np.arange(len(fake_keywords))
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True, squeeze=False)
    ax = axes[0][0]
    plot = a2.plotting.histograms.plot_bar(
        bin_centers=range(len(occurence)),
        hist=occurence,
        width_bars=1,
        ax=ax,
        alpha=1,
    )
    a2.plotting.histograms.annotate_histogram(ax, plot, fake_keywords, as_label="x")
    ax.set_xlim([-0.5, len(occurence) - 0.5])
    a2.plotting.axes_utils.set_axes(ax=ax, label_y="Number of keyword occurence")
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_histogram_2d_dataset(fake_prediction_dataset):
    ax = a2.plotting.histograms.plot_histogram_2d(
        x="truth",
        y="predictions",
        ds=fake_prediction_dataset,
    )
    return ax.get_figure()
