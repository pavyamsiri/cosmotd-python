# Standard modules
from typing import NamedTuple, Tuple


class PlotterSettings(NamedTuple):
    """The configurable settings of a plotter.

    Attributes
    ----------
    title : str
        the title of the plot.
    nrows : int
        the number of rows to plot.
    ncols : int
        the number of columns to plot.
    figsize : Tuple[int, int]
        the size of the figure to plot to in pixels.
    """

    title: str
    nrows: int
    ncols: int
    figsize: Tuple[int, int]


class ImageSettings(NamedTuple):
    """The plotting parameters used when plotting 2D arrays.

    Attributes
    ----------
    vmin : float
        the minimum value covered by color maps.
    vmax : float
        the maximum value covered by color maps.
    cmap : str
        the color map to be used.
    """

    vmin: float
    vmax: float
    cmap: str


class PlotSettings(NamedTuple):
    """The plotting parameters used when plotting a line plot.

    Attributes
    ----------
    color : str
        the color of the line as an RGB hex value.
    linestyle : str
        the style of line.
    """
    color: str
    linestyle: str