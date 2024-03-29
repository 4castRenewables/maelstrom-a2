import pathlib

import a2.plotting.utils_plotting
import numpy as np
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


def test_plot_histogram_symlog(test_gaussian_random_samples, test_symlog_histogram_results):
    x_expected, h_expected = test_symlog_histogram_results.T
    axes = a2.plotting.histograms.plot_histogram(
        test_gaussian_random_samples,
        log=["symlog", "log"],
        symlog_linear_threshold=1e-1,
        xlim=[-1, 1],
        n_bins=20,
    )
    x, _, h = a2.plotting.utils_plotting._get_values_from_bar_object(axes.containers[0])
    assert np.allclose(x, x_expected)
    assert np.allclose(h, h_expected)


def test_plot_histogram_symlog_linthresh(test_gaussian_random_samples, test_symlog_linthresh_histogram_results):
    x_expected, h_expected = test_symlog_linthresh_histogram_results.T
    axes = a2.plotting.histograms.plot_histogram(
        test_gaussian_random_samples,
        log=["symlog", "symlog"],
        xlim=[-1, 1],
        n_bins=20,
    )
    x, _, h = a2.plotting.utils_plotting._get_values_from_bar_object(axes.containers[0])
    assert np.allclose(x, x_expected)
    assert np.allclose(h, h_expected)


def test_plot_histogram_log(tmp_path, test_gaussian_random_samples, test_log_histogram_results):
    directory = tmp_path / "test_plot_histogram_log/"
    directory.mkdir()
    x_expected, h_expected = test_log_histogram_results.T

    axes = a2.plotting.histograms.plot_histogram(
        test_gaussian_random_samples,
        log=["log", False],
        xlim=[1e-2, 1],
        n_bins=20,
        filename=directory / "test.pdf",
    )
    x, _, h = a2.plotting.utils_plotting._get_values_from_bar_object(axes.containers[0])

    assert np.allclose(x, x_expected)
    assert np.allclose(h, h_expected)


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_histogram_2d_dataset(fake_prediction_dataset):
    dic = a2.plotting.histograms.plot_histogram_2d(
        "truth",
        "predictions",
        df=fake_prediction_dataset,
    )
    ax = dic["axes_2d_histograms_10"]
    return ax.get_figure()
