import contextlib
import logging
import pathlib
import typing as t
import warnings

import a2.plotting.analysis
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
    def wrapper_catch_exception(*args, **kwargs):
        result = None
        try:
            if not args[0].ignore:
                result = func(*args, **kwargs)
        except mlflow.exceptions.MlflowException as e:
            logging.info(f"Ignoring mlflow exception:\n{e}")
        except Exception as exc:
            raise ValueError("Tracking error!") from exc
        return result

    return wrapper_catch_exception


class Tracker:
    def __init__(self, ignore=False) -> None:
        self.ignore = ignore
        if not self.ignore:
            initialize_mantik()

    @catch_mantik_exceptions
    def log_param(self, name, value):
        mlflow.log_param(name, value)

    @catch_mantik_exceptions
    def log_params(self, params, **kwargs):
        mlflow.log_params(params)

    @catch_mantik_exceptions
    def log_metric(self, *args, **kwargs):
        mlflow.log_metric(*args, **kwargs)

    @catch_mantik_exceptions
    def log_metrics(self, *args, **kwargs):
        mlflow.log_metrics(*args, **kwargs)

    @contextlib.contextmanager
    def start_run(self, *args, **kwargs):
        if not self.ignore:
            yield mlflow.start_run(*args, **kwargs)
        else:
            yield None

    @catch_mantik_exceptions
    def end_run(self, *args, **kwargs):
        mlflow.end_run(*args, **kwargs)

    @catch_mantik_exceptions
    def log_artifact(self, *args, **kwargs):
        mlflow.log_artifact(*args, **kwargs)

    @catch_mantik_exceptions
    def active_run(self, *args, **kwargs):
        return mlflow.active_run(*args, **kwargs)

    @catch_mantik_exceptions
    def local_file_uri_to_path(self, *args, **kwargs):
        return mlflow.utils.file_utils.local_file_uri_to_path(*args, **kwargs)

    # @catch_mantik_exceptions
    def set_tracking_uri(self, *args, **kwargs):
        mlflow.set_tracking_uri(*args, **kwargs)

    @catch_mantik_exceptions
    def get_tracking_uri(self, *args, **kwargs):
        return mlflow.get_tracking_uri(*args, **kwargs)

    @catch_mantik_exceptions
    def create_experiment(self, name, **kwargs):
        experiment_id = self.get_experiment_by_name(name)
        print(f"{experiment_id=}")
        logging.info(f"{not experiment_id=}")
        if not experiment_id:
            experiment_id = mlflow.create_experiment(name, **kwargs)
            logging.info(f"create new: {experiment_id}")
        if experiment_id is not None and not isinstance(experiment_id, str):
            experiment_id = experiment_id.experiment_id
        return experiment_id

    @catch_mantik_exceptions
    def get_experiment_by_name(self, *args, **kwargs):
        return mlflow.get_experiment_by_name(*args, **kwargs)


def log_metric_classification_report(
    tracker: Tracker,
    truth: t.Sequence,
    predictions: t.Sequence,
    step: int | None = 1,
    label: str = "raining",
    filename_confusion_matrix: pathlib.Path | str = "confusion_matrix.pdf",
    font_size: int = 14,
):
    """
    Compute f1 score and logs results to mlflow

    Parameters:
    ----------
    truth: True labels
    predictions: Predicted labels
    prediction_probabilities: Prediction probability for both labels, shape = [n_tests, 2]
    Step: Current training stop (epoch)
    label: Label name of the classification

    Returns
    -------
    """
    initialize_mantik()
    classification_report = a2.plotting.analysis.check_prediction(
        truth=truth,
        prediction=predictions,
        filename=filename_confusion_matrix,
        output_dict=True,
        label=label,
        font_size=font_size,
    )
    logging.info(classification_report)
    log_classification_report(tracker, classification_report, step, label)
    tracker.log_artifact(filename_confusion_matrix)


def log_classification_report(tracker, classification_report, step, label):
    initialize_mantik()
    tracker.log_metric(
        key=f"eval_f1_{label}",
        value=classification_report[label]["f1-score"],
        step=step,
    )
    tracker.log_metric(
        key=f"eval_f1_not_{label}",
        value=classification_report[f"not {label}"]["f1-score"],
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
