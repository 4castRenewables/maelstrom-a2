import pathlib

import a2.plotting
import a2.utils
import pytest


FILE_LOC = pathlib.Path(__file__).parent
DATA_FOLDER = FILE_LOC / "../../data"
BASELINE_DIR = DATA_FOLDER / "baseline/"


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_create_axes_grid():
    fig, axes, axes_colorbar = a2.plotting.utils_plotting.create_axes_grid(
        n_columns=2,
        n_rows=3,
        colorbar_skip_row_col=[[2, 1], [1, 0]],
        top=0.05,
        bottom=0.05,
        right=0.02,
        left=0.05,
        spacing_x=0.05,
        spacing_y=0.05,
        spacing_colorbar=0.02,
        colorbar_width=0.07,
    )
    return fig


@pytest.mark.mpl_image_compare(BASELINE_DIR=BASELINE_DIR, tolerance=0)
def test_create_axes_grid_no_colorbar():
    fig, axes, axes_colorbar = a2.plotting.utils_plotting.create_axes_grid(
        n_columns=2,
        n_rows=3,
        colorbar_off=True,
        top=0.05,
        bottom=0.05,
        right=0.02,
        left=0.05,
        spacing_x=0.05,
        spacing_y=0.05,
        spacing_colorbar=0.02,
        colorbar_width=0.07,
    )
    return fig
