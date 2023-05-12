import dataclasses
import logging
import pprint
import resource
import time

import numpy as np
import torch


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
    EVALUATION: str = "EVALUATION"


MAX_LENGTH_TIME_TYPES = max([len(x) for x in TimeType().__dict__.values()])


def current_time():
    return time.time()


class Timer:
    """Interface has analagous implementation to deep500 timer. Therefore not all variables used"""

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
    """Max RAM memory used in bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000


class CudaMemoryMonitor:
    def __init__(self) -> None:
        self.init_cuda()

    def init_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.init()
        else:
            logging.info("Cuda initialization failed, as cuda is not available.")

    def reset_cuda_memory_monitoring(self):
        for i_cuda in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i_cuda)

    def get_cuda_memory_usage(self, log_message, style="verbose"):
        logging.info("CUDA memory logging....\n")
        for i_cuda in range(torch.cuda.device_count()):
            logging.info(f"{log_message}: Cuda device {i_cuda} report:\n")
            if style == "verbose":
                pprint.pprint(torch.cuda.memory_stats(i_cuda))
                logging.info(f"Report done! for Cuda device {i_cuda}\n")
            logging.info(f"Gpu memory currently allocated: {torch.cuda.memory_allocated(i_cuda)/1e9} GB.")
            logging.info(f"Gpu max memory allocated: {torch.cuda.max_memory_allocated(i_cuda)/1e9} GB.")
            logging.info(f"Gpu memory currently reserved: {torch.cuda.memory_reserved(i_cuda)/1e9} GB.")
            logging.info(f"Gpu max memory reserved: {torch.cuda.max_memory_reserved(i_cuda)/1e9} GB.")
