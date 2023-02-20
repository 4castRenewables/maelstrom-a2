import pathlib

import a2.plotting
import a2.utils
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_timeseries_raw_data(fake_timeseries_data_raw):
    times, values = fake_timeseries_data_raw
    fig = a2.plotting.timeseries.plot_timeseries(
        time=times, y=values, xlim=None, ylim=[0, 10], label_x="x label", label_y="y label", label="my plot"
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_plot_timeseries_ds(fake_timeseries_data_ds):
    fig = a2.plotting.timeseries.plot_timeseries(
        time="times",
        y="values",
        ds=fake_timeseries_data_ds,
        xlim=None,
        ylim=None,
        label_x=None,
        label_y=None,
        label="my plot",
    )
    return fig
