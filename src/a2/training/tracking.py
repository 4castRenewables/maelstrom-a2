import logging
import typing as t
import warnings

import a2.plotting
import mantik
import mlflow


def initialize_mantik():
    """
    Attemps to initialize mantik, throws exception if fails
    """
    try:
        mantik.init_tracking()
    except Exception as e:
        warnings.warn(f"{e}\nCannot initialize mantik!")


def log_metric_classification_report(truth: t.Sequence, predictions: t.Sequence, step: int = 1):
    """
    Compute f1 score and logs results to mlflow

    Parameters:
    ----------
    truth: True labels
    predictions: Predicted labels
    prediction_probabilities: Prediction probability for both labels, shape = [n_tests, 2]
    Step: Current training stop (epoch)

    Returns
    -------
    """
    classification_report = a2.plotting.analysis.check_prediction(
        truth=truth,
        prediction=predictions,
        filename="confusion_matrix.pdf",
        output_dict=True,
    )
    logging.info(classification_report)
    initialize_mantik()
    mlflow.log_metric(
        key="eval_f1_raining",
        value=classification_report["raining"]["f1-score"],
        step=step,
    )
    mlflow.log_metric(
        key="eval_f1_not_raining",
        value=classification_report["not raining"]["f1-score"],
        step=step,
    )
    mlflow.log_metric(
        key="weighted average f1-score",
        value=classification_report["weighted avg"]["f1-score"],
        step=step,
    )
    mlflow.log_metric(
        key="macro average f1-score",
        value=classification_report["macro avg"]["f1-score"],
        step=step,
    )
    mlflow.log_artifact("confusion_matrix.pdf")
