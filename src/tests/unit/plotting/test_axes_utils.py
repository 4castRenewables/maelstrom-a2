import pathlib

import a2.plotting.axes_utils
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_set_axes():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3])
    a2.plotting.axes_utils.set_axes(
        ax,
        xlim=[0, 4],
        ylim=[0, 4],
        fontsize=8,
        title="test",
        label_x="label x",
        label_y="label y",
        labelrotation_x=45,
        labelrotation_y=45,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_set_colorbar():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = plt.subplot()
    im = ax.imshow(np.arange(100).reshape((10, 10)))
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(im, cax=cax)
    a2.plotting.axes_utils.set_colorbar(colorbar.ax, label_y="label y colorbar", fontsize=8)
    return fig
