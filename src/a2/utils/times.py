import numpy as np
import pandas as pd


def time_breakdown(time: np.datetime64):
    """Breakdown time in individual components, i.e. year, month, day, hour, minute"""
    time_pandas = pd.to_datetime(time)
    year = time_pandas.year
    month = time_pandas.month
    day = time_pandas.day
    hour = time_pandas.hour
    minute = time_pandas.minute
    return year, month, day, hour, minute
