import dataclasses
import logging
import resource
import time

import numpy as np


@dataclasses.dataclass(frozen=True)
class TimeType:
    EPOCH: str = "EPOCH"
    BATCH: str = "BATCH"
    FORWARD: str = "FORWARD"
    BACKWARD: str = "BACKWARD"
    IO: str = "IO"
    TRAINING: str = "TRAINING"
    RUN: str = "RUN"
    SAVING_MODEL: str = "SAVING_MODEL"


MAX_LENGTH_TIME_TYPES = max([len(x) for x in TimeType().__dict__.values()])


def current_time():
    return time.time()


class Timer:
    def __init__(self, print_all_single_time_stats=True, debug=True):
        self.times_archive = {}
        self.times_running = {}
        self.TimeType = TimeType()
        self.debug = debug
        self.print_all_single_time_stats = print_all_single_time_stats

    def start(self, time_type, gpu=None):
        if time_type in self.TimeType.__dict__.keys():
            if time_type in self.times_running:
                logging.warning(f"Overwriting unfinished timing of {time_type}!")
            self.times_running[time_type] = current_time()

    def _archive_timing(self, time_type, duration):
        if time_type not in self.times_archive:
            self.times_archive[time_type] = []
        self.times_archive[time_type].extend([duration])

    def end(self, time_type, gpu=None):
        if time_type in self.times_running:
            duration = current_time() - self.times_running.pop(time_type)
            self._archive_timing(time_type, duration=duration)
            if self.print_all_single_time_stats:
                logging.info(f"{time_type:<10}: took {duration}")
        else:
            logging.warning(f"Attempting to finish timer type {time_type}, which was never started!")

    def complete_all(self):
        # for _type in self.times_running.copy().keys():
        #     self.end(_type)
        pass

    def print_all_time_stats(self):
        for _type, times in self.times_archive.items():
            print(
                f"{_type:<15}; "
                f"min: {min(times):.04e}, "
                f"max: {max(times):.04e}, "
                f"mean: {np.mean(times):.04e}, "
                f"median: {np.median(times):.04e}, "
                f"counts: {len(times):>5}"
            )
            if self.debug:
                for _type, elapsed_time in self.times_running.items():
                    logging.info(f"{_type} still running for {elapsed_time}s.")


def get_max_memory_usage():
    """In bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000
