# External modules
from typing import Callable, ParamSpec
import numpy as np
from numpy import typing as npt

# Internal modules
from .settings import PlotterConfig, LineConfig, ImageConfig, ScatterConfig


"""Constants"""
DPI: int = 100
PLOT_CACHE: str = "data/plot_cache"
VIDEO_CACHE: str = "data/video_cache"


class Plotter:
    """An interface class that serves as an interface for different plotting backends. It can be used on its own if plotting is
    to be turned off."""

    def __init__(
        self, settings: PlotterConfig, progress_callback: Callable[[int], bool | None]
    ):
        """Constructs a plotter configured by the given settings.

        Parameters
        ----------
        settings : PlotterConfig
            settings used to configure plotting.
        progress_callback : Callable[[int], bool | None]
            the callback used to update progress bars.
        """
        # Store progress callback
        self._progress_callback = progress_callback

    def reset(self):
        """Clear the plotting canvas."""
        pass

    def flush(self):
        """Draws all elements."""
        # Increase progress
        self._progress_callback(1)

    def close(self):
        """Closes the plotter and any resources it may be using."""
        pass

    # Plotting functions

    def draw_image(
        self,
        data: npt.NDArray[np.float32],
        extents: tuple[float, float, float, float],
        axis_index: int,
        image_index: int,
        image_config: ImageConfig,
    ):
        """Draws an image from a 2D array.

        Parameters
        ----------
        data : npt.NDArray[np.float32]
            the 2D array to draw.
        extents : tuple[float, float, float, float]
            the image extents.
        axis_index : int
            the index of the primary axis to draw the image to.
        image_index : int
            the index of the image to draw to.
        image_config : ImageConfig
            the parameters to use when drawing.
        """
        pass

    def draw_plot(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        axis_index: int,
        line_index: int,
        line_config: LineConfig,
    ):
        """Plot `y` against `x`.

        Parameters
        ----------
        x : npt.NDArray[np.float32]
            the data along the x-axis.
        y : npt.NDArray[np.float32]
            the data along the y-axis.
        axis_index : int
            the index of the primary axis to draw the scatter plot to.
        line_index : int
            the index of the line to draw to.
        line_config : LineConfig
            the parameters to use when drawing.
        """
        pass

    def draw_scatter(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        axis_index: int,
        scatter_index: int,
        scatter_config: ScatterConfig,
    ):
        """Plot `y` against `x` as a scatter point.

        Parameters
        ----------
        x : npt.NDArray[np.float32]
            the data along the x-axis.
        y : npt.NDArray[np.float32]
            the data along the y-axis.
        axis_index : int
            the index of the primary axis to draw the line to.
        scatter_index : int
            the index of the scatter plot to draw to.
        scatter_config : LineConfig
            the parameters to use when drawing the scatter plot.
        """
        pass

    # Axes Setter Functions

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

    def set_axes_limits(
        self,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
        axis_index: int,
    ):
        """Sets the limits of the x and y axes.

        Parameters
        ----------
        x_min : float | None
            the minimum x value.
        x_max : float | None
            the maximum x value.
        y_min : float | None
            the minimum y value.
        y_max : float | None
            the maximum y value.
        axis_index : int
            the index of the primary axis to set the axes limits of.
        """
        pass

    # Miscellaneous Axes Functions

    def set_autoscale(self, enable: bool, axis: str, axis_index: int):
        """Turns on or off autoscaling for an axis.

        Parameters
        ----------
        enable : bool
            if `True` will turn on autoscale and if `False` will turn off autoscale.
        axis : str
            the axis to operate on. Allowed choices are "both", "x" and "y".
        axis_index : int
            the index of the set of axes to operate on.
        """
        pass

    def set_x_scale(self, scale: str, axis_index: int):
        """Sets the x-axis scale.

        Parameters
        ----------
        scale : str
            the scale to set to. Allowed choices are "linear", "log", "symlog", "logit", etc.
        axis_index : int
            the index of the set of axes to operate on.
        """
        pass

    def set_y_scale(self, scale: str, axis_index: int):
        """Sets the y-axis scale.

        Parameters
        ----------
        scale : str
            the scale to set to. Allowed choices are "linear", "log", "symlog", "logit", etc.
        axis_index : int
            the index of the set of axes to operate on.
        """
        pass

    def remove_axis_ticks(self, axis: str, axis_index: int):
        """Removes axis ticks for an axis.

        Parameters
        ----------
        axis : str
            the axis to operate on. Allowed choices are "both", "x" and "y".
        axis_index : int
            the index of the set of axes to operate on.
        """
        pass
