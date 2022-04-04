# Standard modules
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple, TypeAlias

# External modules
import matplotlib.axes
import matplotlib.colorbar
import matplotlib.figure
import numpy as np


"""Type Aliases"""
# Matplotlib figures
MatplotlibFigure: TypeAlias = matplotlib.figure.Figure
# Matplotlib axes
MatplotlibAxes: TypeAlias = matplotlib.axes.Axes
# Matplotlib objects that associate with axes
# NOTE: Currently only supports colorbars but more can be added potentially
MatplotlibSubAxes: TypeAlias = matplotlib.colorbar.Colorbar


"""Constants"""
DPI: int = 100
PLOT_CACHE: str = "data/plot_cache"
VIDEO_CACHE: str = "data/video_cache"


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


class Plotter(ABC):
    """An abstract class that serves as an interface for different plotting backends."""

    @abstractmethod
    def __init__(self, settings: PlotterSettings):
        """Constructs a plotter configured by the given settings.

        Parameters
        ----------
        settings : PlotterSettings
            settings used to configure plotting.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clear the plotting canvas."""
        pass

    @abstractmethod
    def draw_image(self, data: np.ndarray, axis_index: int, plot_args: ImageSettings):
        """Draws an image from a 2D array.

        Parameters
        ----------
        data : np.ndarray
            the 2D array to draw.
        axis_index : int
            the index of the primary axis to draw the image to.
        plot_args : ImageSettings
            the parameters to use when drawing.
        """
        pass

    @abstractmethod
    def draw_plot(
        self, x: np.ndarray, y: np.ndarray, axis_index: int, plot_args: PlotSettings
    ):
        """Plot `y` against `x`.

        Parameters
        ----------
        x : np.ndarray
            the data along the x-axis.
        y : np.ndarray
            the data along the y-axis.
        axis_index : int
            the index of the primary axis to draw the image to.
        plot_args : PlotSettings
            the parameters to use when drawing.
        """
        pass

    @abstractmethod
    def set_title(self, title: str, axis_index: int):
        """Sets the title of a plot.

        Parameters
        ----------
        title : str
            the title to set.
        axis_index : int
            the index of the primary axis to set the title of.
        """
        pass

    @abstractmethod
    def set_axes_labels(self, xlabel: str, ylabel: str, axis_index: int):
        """Sets the labels of the x and y axes.

        Parameters
        ----------
        xlabel : str
            the label of the x axis.
        ylabel : str
            the label of the y axis.
        axis_index : int
            the index of the primary axis to set the axes labels of.
        """
        pass

    @abstractmethod
    def set_axes_limits(
        self,
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float],
        y_max: Optional[float],
        axis_index: int,
    ):
        """Sets the labels of the x and y axes.

        Parameters
        ----------
        x_min : Optional[float]
            the minimum x value.
        x_max : Optional[float]
            the maximum x value.
        y_min : Optional[float]
            the minimum y value.
        y_max : Optional[float]
            the maximum y value.
        axis_index : int
            the index of the primary axis to set the axes labels of.
        """
        pass

    @abstractmethod
    def flush(self):
        """Draws all elements."""
        pass

    @abstractmethod
    def close(self):
        """Closes the plotter and any resources it may be using."""
        pass


class MockPlotter(Plotter):
    """An abstract class that serves as an interface for different plotting backends."""

    def __init__(self, _settings: PlotterSettings):
        pass

    def reset(self):
        pass

    def draw_image(
        self, _data: np.ndarray, _axis_index: int, _plot_args: ImageSettings
    ):
        pass

    def draw_plot(
        self, _x: np.ndarray, _y: np.ndarray, _axis_index: int, _plot_args: PlotSettings
    ):
        pass

    def set_title(self, _title: str, _axis_index: int):
        pass

    def set_axes_labels(self, _xlabel: str, _ylabel: str, _axis_index: int):
        pass

    def set_axes_limits(
        self,
        _x_min: Optional[float],
        _x_max: Optional[float],
        _y_min: Optional[float],
        _y_max: Optional[float],
        _axis_index: int,
    ):
        pass

    def flush(self):
        pass

    def close(self):
        pass
