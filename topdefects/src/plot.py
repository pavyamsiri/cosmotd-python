"""This module handles all the plotting backends for the simulations."""

# Standard modules
from abc import ABC, abstractmethod
import glob
import os
from typing import NamedTuple, Tuple, TypeAlias, Optional


# External modules
import ffmpeg
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


# Constants
DPI: int = 100
PLOT_CACHE: str = "data/plot_cache"
VIDEO_CACHE: str = "data/video_cache"

# Type aliases
# Matplotlib figures
MatplotlibFigure: TypeAlias = matplotlib.figure.Figure
# Matplotlib axes
MatplotlibAxes: TypeAlias = matplotlib.axes.Axes
# Matplotlib objects that associate with axes
# NOTE: Currently only supports colorbars but more can be added potentially
MatplotlibSubAxes: TypeAlias = matplotlib.colorbar.Colorbar


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
    vmin : float
        the minimum value covered by color maps.
    vmax : float
        the maximum value covered by color maps.
    cmap : str
        the color map to be used.
    """

    pass


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

    def __init__(self, settings: PlotterSettings):
        """Constructs a plotter configured by the given settings.

        Parameters
        ----------
        settings : PlotterSettings
            settings used to configure plotting.
        """
        pass

    def reset(self):
        """Clear the plotting canvas."""
        pass

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

    def flush(self):
        """Draws all elements."""
        pass

    def close(self):
        """Closes the plotter and any resources it may be using."""
        pass


class MplNotebookPlotter(Plotter):
    """This plotter uses `matplotlib` as its plotting backend and is designed to be used for Jupyter notebooks.
    Plots are displayed upon flush.

    Attributes
    ----------
    fig : MatplotlibFigure
        the figure to draw the plots to.
    axes : list[MatplotlibAxes]
        a list of primary axes to draw plots to. Each axis corresponds to a subplot.
    sub_axes : list[dict[str, MatplotlibSubAxes]]
        a list of dictionaries of plot elements associated with each primary axis. The index of the list relates a dictionary
        with an axis at the same index in the list `axes`. Each dictionary is keyed by the name of the element and its value
        is the corresponding element.
    """

    def __init__(self, settings: PlotterSettings):
        """Constructs a plotter configured by the given settings.

        Parameters
        ----------
        settings : PlotterSettings
            settings used to configure plotting.
        """
        # Set backend for Jupyter notebook usage
        matplotlib.use("nbAgg")
        # Create figure according to the given size
        dpi = DPI
        figsize = (
            settings.figsize[0] / dpi,
            settings.figsize[1] / dpi,
        )
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._fig.suptitle(settings.title)
        # Initialise axes and subaxes
        self._axes = []
        self._sub_axes = []
        num_plots = settings.nrows * settings.ncols
        for i in range(num_plots):
            # Create a primary axis
            self._axes.append(
                self._fig.add_subplot(settings.nrows, settings.ncols, i + 1)
            )
            # Create a list of sub axes which are auxiliary axes associated with each primary axis
            self._sub_axes.append({})

    def reset(self):
        """Clears the plotting canvas."""
        # Clear all primary axes
        for axis in self._axes:
            axis.clear()
        # Clear all sub axes
        for axis in self._sub_axes:
            for axis_type in axis:
                # NOTE: This might break because I added this part to remove colorbars. Other objects might not be removed with
                # a method called `remove` and so we might need to check which method we need to use.
                axis[axis_type].remove()

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
        # Display a 2D array
        img = self._axes[axis_index - 1].imshow(
            data, vmin=plot_args.vmin, vmax=plot_args.vmax, cmap=plot_args.cmap
        )
        # Create a colorbar
        self._sub_axes[axis_index - 1]["colorbar"] = self._fig.colorbar(
            img, ax=self._axes[axis_index - 1], fraction=0.046, pad=0.04
        )
        # Invert y-axis so (0, 0) is in the bottom left
        self._axes[axis_index - 1].invert_yaxis()

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
        self._axes[axis_index - 1].plot(x, y)

    def set_title(self, title: str, axis_index: int):
        """Sets the title of a plot.

        Parameters
        ----------
        title : str
            the title to set.
        axis_index : int
            the index of the primary axis to set the title of.
        """
        # Set title
        self._axes[axis_index - 1].set_title(title)

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
        # Set x and y labels
        self._axes[axis_index - 1].set_xlabel(xlabel)
        self._axes[axis_index - 1].set_ylabel(ylabel)

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
        self._axes[axis_index - 1].set_xlim(x_min, x_max)
        self._axes[axis_index - 1].set_ylim(y_min, y_max)

    def flush(self):
        """Draws all elements."""
        plt.tight_layout()
        self._fig.canvas.draw()

    def close(self):
        """Closes the figure."""
        plt.close(self._fig)

    def __del__(self):
        """Destructor that ensures the figure gets closed if the `close` did not get called."""
        plt.close(self._fig)


class MplPngPlotter(Plotter):
    """This plotter uses `matplotlib` as its plotting backend.
    Plots are saved as a png upon flush in a folder called `plot_cache`.

    Attributes
    ----------
    dpi : int
        the number of dots per inch.
    figsize : Tuple[float, float]
        the size of the figure in inches.
    nrows : int
        the number of rows.
    ncols : int
        the number of columns.
    num_plots : int
        the number of plots.
    fig : Optional[MatplotlibFigure]
        the figure to draw the plots to.
    axes : Optional[list[MatplotlibAxes]]
        a list of primary axes to draw plots to. Each axis corresponds to a subplot.
    title : str
        the title of the plot.
    count : int
        the frame count.
    """

    def __init__(self, settings: PlotterSettings):
        """Constructs a plotter configured by the given settings.

        Parameters
        ----------
        settings : PlotterSettings
            settings used to configure plotting.
        """
        # Set backend to a rasterizer to optimise for pngs.
        matplotlib.use("agg")
        # Save plot configuration
        self._dpi = DPI
        self._figsize = (
            settings.figsize[0] / self._dpi,
            settings.figsize[1] / self._dpi,
        )
        self._nrows = settings.nrows
        self._ncols = settings.ncols
        self._num_plots = self._nrows * self._ncols
        # Declare figure and axes as attributes
        self._fig = None
        self._axes = None
        # Plot title
        self._title = settings.title
        # Frame count
        self._count = 0

    def reset(self):
        """Initialises a new figure and axes."""
        # NOTE: This does not 'reset' the canvas as all figures are closed upon flush and so they have been cleared already.
        # Create a figure
        self._fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        self._fig.suptitle(self._title, fontsize="x-large")
        # Initialise primary axes
        self._axes = []
        for i in range(self._num_plots):
            self._axes.append(self._fig.add_subplot(self._nrows, self._ncols, i + 1))

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
        # Display a 2D array
        img = self._axes[axis_index - 1].imshow(
            data, vmin=plot_args.vmin, vmax=plot_args.vmax, cmap=plot_args.cmap
        )
        # Create a colorbar
        self._fig.colorbar(img, ax=self._axes[axis_index - 1], fraction=0.046, pad=0.04)
        # Invert y-axis so (0, 0) is in the bottom left
        self._axes[axis_index - 1].invert_yaxis()

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
        self._axes[axis_index - 1].plot(x, y)

    def set_title(self, title: str, axis_index: int):
        """Sets the title of a plot.

        Parameters
        ----------
        title : str
            the title to set.
        axis_index : int
            the index of the primary axis to set the title of.
        """
        # Set title
        self._axes[axis_index - 1].set_title(title)

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
        # Set x and y labels
        self._axes[axis_index - 1].set_xlabel(xlabel)
        self._axes[axis_index - 1].set_ylabel(ylabel)

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
        self._axes[axis_index - 1].set_xlim(x_min, x_max)
        self._axes[axis_index - 1].set_ylim(y_min, y_max)

    def flush(self):
        """Saves the figure as a png in a cache."""
        plt.tight_layout()
        # Construct file name
        src_folder = os.path.dirname(os.path.realpath(__file__))
        file_name = f"{src_folder}/../{PLOT_CACHE}/frame_{self._count}.png"

        # Save figure as png
        plt.savefig(
            file_name,
            facecolor="white",
            transparent=False,
        )
        # Increment frame count
        self._count += 1
        # Close figure
        plt.close(self._fig)

    def close(self):
        """Does not need to close figure as all figures should be closed upon flush."""
        pass


class MplMp4Plotter(Plotter):
    """This plotter uses `matplotlib` as its plotting backend.
    Plots are saved as a png upon flush in a folder called `plot_cache` and then ffmpeg is used to create
    a video from that png sequence and saved in a folder called `video_cache`.

    Attributes
    ----------
    dpi : int
        the number of dots per inch.
    figsize : Tuple[float, float]
        the size of the figure in inches.
    nrows : int
        the number of rows.
    ncols : int
        the number of columns.
    num_plots : int
        the number of plots.
    fig : Optional[MatplotlibFigure]
        the figure to draw the plots to.
    axes : Optional[list[MatplotlibAxes]]
        a list of primary axes to draw plots to. Each axis corresponds to a subplot.
    title : str
        the title of the plot.
    count : int
        the frame count.
    """

    def __init__(self, settings: PlotterSettings):
        # Set backend to a rasterizer to optimise for pngs.
        matplotlib.use("agg")
        # Save plot configuration
        self._dpi = DPI
        self._figsize = (
            settings.figsize[0] / self._dpi,
            settings.figsize[1] / self._dpi,
        )
        self._nrows = settings.nrows
        self._ncols = settings.ncols
        self._num_plots = self._nrows * self._ncols
        # Declare figure and axes as attributes
        self._fig = None
        self._axes = None
        # Plot title
        self._title = settings.title
        # Count
        self._count = 0

    def reset(self):
        """Initialises a new figure and axes."""
        # Create a figure
        self._fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        self._fig.suptitle(self._title, fontsize="x-large")
        # Initialise primary axes
        self._axes = []
        for i in range(self._num_plots):
            self._axes.append(self._fig.add_subplot(self._nrows, self._ncols, i + 1))

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
        # Display a 2D array
        img = self._axes[axis_index - 1].imshow(
            data, vmin=plot_args.vmin, vmax=plot_args.vmax, cmap=plot_args.cmap
        )
        # Create a colorbar
        self._fig.colorbar(img, ax=self._axes[axis_index - 1], fraction=0.046, pad=0.04)
        # Invert y-axis so (0, 0) is in the bottom left
        self._axes[axis_index - 1].invert_yaxis()

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
        self._axes[axis_index - 1].plot(x, y)

    def set_title(self, title: str, axis_index: int):
        """Sets the title of a plot.

        Parameters
        ----------
        title : str
            the title to set.
        axis_index : int
            the index of the primary axis to set the title of.
        """
        self._axes[axis_index - 1].set_title(title)

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
        # Set x and y labels
        self._axes[axis_index - 1].set_xlabel(xlabel)
        self._axes[axis_index - 1].set_ylabel(ylabel)

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
        self._axes[axis_index - 1].set_xlim(x_min, x_max)
        self._axes[axis_index - 1].set_ylim(y_min, y_max)

    def flush(self):
        """Saves the figure as a png in a cache."""
        plt.tight_layout()
        # Construct file name
        src_folder = os.path.dirname(os.path.realpath(__file__))
        file_name = f"{src_folder}/../{PLOT_CACHE}/frame_{self._count}.png"

        # Save figure as png
        plt.savefig(
            file_name,
            facecolor="white",
            transparent=False,
        )
        # Increment frame count
        self._count += 1
        # Close figure
        plt.close(self._fig)

    def close(self):
        """Stitches the png sequence created during flushes together into an mp4."""
        # Construct file names
        src_folder = os.path.dirname(os.path.realpath(__file__))
        input_file_template = f"{src_folder}/../{PLOT_CACHE}/frame_%d.png"
        output_file = f"{src_folder}/../{VIDEO_CACHE}/simulation.mp4"
        (
            ffmpeg.input(
                input_file_template,
                framerate=20,
                y=None,
            )
            .output(output_file, crf=25, pix_fmt="yuv420p")
            .overwrite_output()
            .run()
        )
