import time

import logging
import pandas as pd
from pynvml import *  # noqa
from multiprocessing import Process, Queue, Event

logger = logging.getLogger(__name__)


def power_loop(queue, event, interval):
    nvmlInit()  # noqa
    device_count = nvmlDeviceGetCount()  # noqa
    device_list = [nvmlDeviceGetHandleByIndex(idx) for idx in range(device_count)]  # noqa
    power_value_dict = {idx: [] for idx in range(device_count)}
    power_value_dict["timestamps"] = []
    last_timestamp = time.time()

    while not event.is_set():
        for idx, handle in enumerate(device_list):
            power = nvmlDeviceGetPowerUsage(handle)  # noqa
            power_value_dict[idx].append(power * 1e-3)
        timestamp = time.time()
        power_value_dict["timestamps"].append(timestamp)
        wait_for = max(0, 1e-3 * interval - (timestamp - last_timestamp))
        time.sleep(wait_for)
        last_timestamp = timestamp
    queue.put(power_value_dict)


class GetPower(object):
    def __enter__(self):
        self.end_event = Event()
        self.power_queue = Queue()

        interval = 100  # ms
        self.smip = Process(target=power_loop, args=(self.power_queue, self.end_event, interval))
        self.smip.start()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.set()
        power_value_dict = self.power_queue.get()
        self.smip.join()

        self.df = pd.DataFrame(power_value_dict)

    def energy(self):
        _energy = []
        energy_df = (
            self.df.loc[:, self.df.columns != "timestamps"]
            .astype(float)
            .multiply(self.df["timestamps"].diff(), axis="index")
            / 3600
        )
        _energy = energy_df[1:].sum(axis=0).values.tolist()
        return _energy


def save_energy_to_file(measured_scope, folder, file_id):
    f = open(folder + f"EnergyFile-NVDA-{file_id}", "w")
    logger.info("Energy data:")
    logger.info(measured_scope.df)
    measured_scope.df.to_csv(folder + "EnergyFile-NVDA-{args.id}.csv")
    logger.info("Energy-per-GPU-list:")
    energy_int = measured_scope.energy()
    logger.info(f"integrated: {energy_int}")
    f.write(f"integrated: {energy_int}")
    f.close()


# if __name__ == "__main__":
#     with GetPower() as measured_scope:
#         logger.info("Measuring Energy during main() call")
#         try:
#             main(args)
#         except Exception as exc:
#             import traceback

#             logger.info(f"Errors occured during training: {exc}")
#             logger.info(f"Traceback: {traceback.format_exc()}")
#     logger.info("Energy data:")

#     logger.info(measured_scope.df)
#     logger.info("Energy-per-GPU-list:")
#     energy_int = measured_scope.energy()
#     logger.info(f"integrated: {energy_int}")
