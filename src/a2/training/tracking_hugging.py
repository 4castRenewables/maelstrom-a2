import logging

import transformers

from . import tracking


class LogCallback(transformers.TrainerCallback):
    def __init__(self, tracker: tracking.Tracker | None = None, log_to_stdout=True):
        self.tracker = tracker
        self.log_to_stdout = log_to_stdout
        self.steps_logged = {"train": [], "evaluate": []}

    def _log_metrics(self, state, which="train", prefix=""):
        logs = state.log_history
        evaluate_logs = [_l for _l in logs if "eval_loss" in _l]
        training_logs = [_l for _l in logs if "loss" in _l]
        if which == "train":
            log = training_logs
        elif which == "evaluate":
            log = evaluate_logs
        else:
            raise ValueError(f"Metric type {which=} not available (yet)!")
        if log:
            current_log = log[-1]
            if self.log_to_stdout:
                logging.info(f"LogCallback: {prefix} {current_log}")
            step = current_log["step"]
            if step in self.steps_logged:
                # do not log same step twice: may occur as `on_log` also called during evaluation
                return
            if self.tracker is not None:
                self.tracker.log_metrics({k: v for k, v in current_log.items() if k != "step"}, step=step)
            self.steps_logged[which].append(step)

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self._log_metrics(state, which="evaluate", prefix="on_evaluate: ")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Using this callback as `on_step_end` doesn't include the updated loss after step"""
        if state.is_local_process_zero:
            self._log_metrics(state, which="train", prefix="on_log: ")
