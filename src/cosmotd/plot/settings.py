# Standard modules
from typing import NamedTuple, Tuple


class PlotterConfig(NamedTuple):
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


class ImageConfig(NamedTuple):
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


class LineConfig(NamedTuple):
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


class ScatterConfig(NamedTuple):
    """The plotting parameters used when plotting a scatter plot.

    Attributes
    ----------
    facecolors : list[str] | str
        the face color of the scatter point markers. Either a sequence of colors or a string can be passed.
    edgecolors : list[str] | str
        the edge color of the scatter point markers. Either a sequence of colors or a string can be passed. If given `"face"`
        the edge color will be the same as the face color, and if `"none"` is given, no edges are drawn.
    marker : str
        the marker used for all scatter points.
    linewidths : list[float] | float
        the width(s) of the scatter points.
    """

    facecolors: list[str] | str
    edgecolors: list[str] | str
    marker: str
    linewidths: list[float] | float
