# Standard modules
import glob
import os
from typing import Optional

# External modules
import ffmpeg
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# Internal modules
from .base import DPI, PLOT_CACHE, VIDEO_CACHE
from .base import PlotSettings, Plotter, PlotterSettings, ImageSettings


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
        mpl.use("agg")

        # Delete frames in plot_cache
        src_folder = os.path.dirname(os.path.realpath(__file__))
        file_name = f"{src_folder}/../../{PLOT_CACHE}/frame_*.png"
        for plot_file in glob.glob(file_name):
            os.remove(plot_file)

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
        self._axes[axis_index - 1].set_autoscale_on(False)
        self._axes[axis_index - 1].plot(x, y, color=plot_args.color, linestyle=plot_args.linestyle)

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
        file_name = f"{src_folder}/../../{PLOT_CACHE}/frame_{self._count}.png"

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
        input_file_template = f"{src_folder}/../../{PLOT_CACHE}/frame_%d.png"
        output_file = f"{src_folder}/../../{VIDEO_CACHE}/simulation.mp4"
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
