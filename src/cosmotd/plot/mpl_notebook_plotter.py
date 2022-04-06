# Standard modules
from typing import Optional

# External modules
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# Internal modules
from .base import DPI
from .base import Plotter
from .settings import PlotSettings, PlotterSettings, ImageSettings


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
        mpl.use("nbAgg")
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
        self._axes[axis_index - 1].plot(
            x, y, color=plot_args.color, linestyle=plot_args.linestyle
        )

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

    def set_legend(self, legend: list[str], axis_index: int):
        """Sets the legend of a plot.

        Parameters
        ----------
        legend : list[str]
            the legend.
        axis_index : int
            the index of the primary axis to set the legend of.
        """
        # Set legend
        self._axes[axis_index - 1].legend(legend)

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
