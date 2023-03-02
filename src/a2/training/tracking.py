import contextlib
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


def catch_mantik_exceptions(func):
    def wrapper_do_twice(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except mlflow.exceptions.MlflowException as e:
            logging.info(f"Ignoring mlflow exception:\n{e}")
        else:
            func(*args, **kwargs)

    return wrapper_do_twice


class Tracker:
    def __init__(self) -> None:
        initialize_mantik()

    @catch_mantik_exceptions
    def log_param(self, name, value):
        mlflow.log_param(name, value)

    @catch_mantik_exceptions
    def log_params(self, params, **kwargs):
        mlflow.log_params(params)

    @contextlib.contextmanager
    def start_run(self, *args, **kwargs):
        yield mlflow.start_run(*args, **kwargs)

    def end_run(self, *args, **kwargs):
        mlflow.end_run(*args, **kwargs)

    @catch_mantik_exceptions
    def log_artifact(self, *args, **kwargs):
        mlflow.log_artifact(*args, **kwargs)


def log_metric_classification_report(tracker: Tracker, truth: t.Sequence, predictions: t.Sequence, step: int = 1):
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
    initialize_mantik()
    classification_report = a2.plotting.analysis.check_prediction(
        truth=truth,
        prediction=predictions,
        filename="confusion_matrix.pdf",
        output_dict=True,
    )
    logging.info(classification_report)
    log_classification_report(tracker, classification_report, step)
    tracker.log_artifact("confusion_matrix.pdf")


def log_classification_report(tracker, classification_report, step):
    initialize_mantik()
    tracker.log_metric(
        key="eval_f1_raining",
        value=classification_report["raining"]["f1-score"],
        step=step,
    )
    tracker.log_metric(
        key="eval_f1_not_raining",
        value=classification_report["not raining"]["f1-score"],
        step=step,
    )
    tracker.log_metric(
        key="weighted average f1-score",
        value=classification_report["weighted avg"]["f1-score"],
        step=step,
    )
    tracker.log_metric(
        key="macro average f1-score",
        value=classification_report["macro avg"]["f1-score"],
        step=step,
    )
