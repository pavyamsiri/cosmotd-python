# Standard modules
from abc import ABC, abstractmethod
from typing import Optional

# External modules
import numpy as np

# Internal modules
from .settings import PlotterConfig, LineConfig, ImageConfig, ScatterConfig


"""Constants"""
DPI: int = 100
PLOT_CACHE: str = "data/plot_cache"
VIDEO_CACHE: str = "data/video_cache"


class Plotter(ABC):
    """An abstract class that serves as an interface for different plotting backends."""

    @abstractmethod
    def __init__(self, settings: PlotterConfig):
        """Constructs a plotter configured by the given settings.

        Parameters
        ----------
        settings : PlotterConfig
            settings used to configure plotting.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clear the plotting canvas."""
        pass

    @abstractmethod
    def flush(self):
        """Draws all elements."""
        pass

    @abstractmethod
    def close(self):
        """Closes the plotter and any resources it may be using."""
        pass

    # Plotting functions

    @abstractmethod
    def draw_image(
        self,
        data: np.ndarray,
        axis_index: int,
        image_index: int,
        image_config: ImageConfig,
    ):
        """Draws an image from a 2D array.

        Parameters
        ----------
        data : np.ndarray
            the 2D array to draw.
        axis_index : int
            the index of the primary axis to draw the image to.
        image_index : int
            the index of the image to draw to.
        image_config : ImageConfig
            the parameters to use when drawing.
        """
        pass

    @abstractmethod
    def draw_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axis_index: int,
        line_index: int,
        line_config: LineConfig,
    ):
        """Plot `y` against `x`.

        Parameters
        ----------
        x : np.ndarray
            the data along the x-axis.
        y : np.ndarray
            the data along the y-axis.
        axis_index : int
            the index of the primary axis to draw the scatter plot to.
        line_index : int
            the index of the line to draw to.
        line_config : LineConfig
            the parameters to use when drawing.
        """
        pass

    @abstractmethod
    def draw_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axis_index: int,
        scatter_index: int,
        scatter_config: ScatterConfig,
    ):
        """Plot `y` against `x` as a scatter point.

        Parameters
        ----------
        x : np.ndarray
            the data along the x-axis.
        y : np.ndarray
            the data along the y-axis.
        axis_index : int
            the index of the primary axis to draw the line to.
        scatter_index : int
            the index of the scatter plot to draw to.
        scatter_config : LineConfig
            the parameters to use when drawing the scatter plot.
        """
        pass

    # Axes-Specific Functions

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
    def set_legend(self, legend: list[str], axis_index: int):
        """Sets the legend of a plot.

        Parameters
        ----------
        legend : list[str]
            the legend.
        axis_index : int
            the index of the primary axis to set the legend of.
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
        """Sets the limits of the x and y axes.

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
            the index of the primary axis to set the axes limits of.
        """
        pass


class MockPlotter(Plotter):
    """An abstract class that serves as an interface for different plotting backends."""

    def __init__(self, settings: PlotterConfig):
        pass

    def reset(self):
        pass

    def flush(self):
        pass

    def close(self):
        pass

    def draw_image(
        self,
        data: np.ndarray,
        axis_index: int,
        image_index: int,
        image_config: ImageConfig,
    ):
        pass

    def draw_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axis_index: int,
        line_index: int,
        line_config: LineConfig,
    ):
        pass

    def draw_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axis_index: int,
        scatter_index: int,
        scatter_config: ScatterConfig,
    ):
        pass

    def set_title(self, title: str, axis_index: int):
        pass

    def set_legend(self, legend: list[str], axis_index: int):
        pass

    def set_axes_labels(self, xlabel: str, ylabel: str, axis_index: int):
        pass

    def set_axes_limits(
        self,
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float],
        y_max: Optional[float],
        axis_index: int,
    ):
        pass
