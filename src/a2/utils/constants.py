import pathlib

import matplotlib.axes
import matplotlib.figure
import pandas as pd


UK_LONGITUDE_LIMIT = [-9, 3]
UK_LATITUDE_LIMIT = [49, 61]
PATH_DAPCEDA = pathlib.Path("/home/kristian/Projects/a2/notebooks/dataset/dap.ceda.ac.uk")
TIME_TYPE_PANDAS = pd.Timestamp
TYPE_MATPLOTLIB_FIGURES = matplotlib.figure.Figure
TYPE_MATPLOTLIB_AXES = matplotlib.axes.Axes
