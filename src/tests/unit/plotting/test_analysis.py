import pathlib

import a2.plotting
import numpy as np
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


def test_plot_prediction_certainty(fake_prediction, fake_prediction_certainties):
    truth, _, prediction_probabilities = fake_prediction
    _, H = a2.plotting.analysis.plot_prediction_certainty(truth, prediction_probabilities, return_matrix=True)
    assert np.array_equal(H, fake_prediction_certainties)


def test_classification_report(fake_prediction, fake_classification_report):
    truth, predictions, _ = fake_prediction
    report = a2.plotting.analysis.classification_report(truth, predictions)
    assert report == fake_classification_report


def test_check_prediction(fake_prediction, fake_classification_report):
    truth, predictions, _ = fake_prediction
    report = a2.plotting.analysis.check_prediction(truth, predictions, output_dict=True)
    assert report == fake_classification_report


def test_plot_roc(fake_prediction, fake_roc_rates):
    truth, _, prediction_probabilities = fake_prediction
    _, tpr, fpr = a2.plotting.analysis.plot_roc(truth, prediction_probabilities, return_rates=True)
    tpr_expected, fpr_expected = fake_roc_rates
    assert np.array_equal(tpr, tpr_expected)
    assert np.array_equal(fpr, fpr_expected)


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_confusion_matrix(fake_prediction, fake_roc_rates):
    truth, predictions, _ = fake_prediction
    fig = a2.plotting.analysis.plot_confusion_matrix(truth, predictions)
    return fig
