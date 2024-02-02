import pathlib
from typing import Literal
from typing import TypeAlias

import matplotlib.axes
import matplotlib.figure
import pandas as pd


UK_LONGITUDE_LIMIT = [-9, 3]
UK_LATITUDE_LIMIT = [49, 61]
PATH_DAPCEDA = pathlib.Path("/home/kristian/Projects/a2/notebooks/dataset/dap.ceda.ac.uk")
TIME_TYPE_PANDAS = pd.Timestamp
TYPE_MATPLOTLIB_AXES: TypeAlias = matplotlib.axes.Axes
TYPE_MATPLOTLIB_FIGURES: TypeAlias = matplotlib.figure.Figure
TYPE_MATPLOTLIB_COLORBAR: TypeAlias = matplotlib.colorbar.Colorbar
TYPE_DATASET_BACKEND: Literal = ["xarray", "pandas"]
AXES_STYLE_TYPE: TypeAlias = Literal["axes_single", "axes_grid"]
DEFAULT_COLORMAP: str = "viridis"
DEFAULT_FONTSIZE_LARGE: float = 20
DEFAULT_FONTSIZE_SMALL: float = 12
DEFAULT_FIGURE_SIZE: tuple[float, float] = (10.0, 6.0)
