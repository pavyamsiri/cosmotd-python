# Internal modules
import glob
import multiprocessing as mp
import os
from typing import Callable, NamedTuple

# External modules
import ffmpeg
import matplotlib as mpl
import numpy as np
from numpy import typing as npt

# Internal modules
from cosmotd.plot.plotter import DPI, PLOT_FOLDER, VIDEO_FOLDER
from cosmotd.plot.plotter import Plotter
from cosmotd.plot.settings import ImageConfig, LineConfig, PlotterConfig, ScatterConfig
from cosmotd.plot.mpl_plotter.mpl_png_plotter import SUB_TO_ROOT


# TODO: Find way to be able to configure this parameter to optimise for each plotting
# Number of frames to render per process job
FRAMES_IN_FLIGHT = 1


class DrawImageCommand(NamedTuple):
    """A structure that contains the necessary information to draw a 2D array.

    Attributes
    ----------
    data : npt.NDArray[np.float32]
        the data to draw.
    extents : tuple[float, float, float, float]
        the image extents.
    axis_index : int
        the index of the primary axis to draw the image to.
    image_index : int
        the index of the image to draw to.
    image_config : ImageConfig
        the parameters to use when drawing.
    """

    data: npt.NDArray[np.float32]
    extents: tuple[float, float, float, float]
    axis_index: int
    image_index: int
    config: ImageConfig


class DrawPlotCommand(NamedTuple):
    """A structure that contains the necessary information to draw a line plot.

    Attributes
    ----------
    xdata : npt.NDArray[np.float32]
        the data along the x-axis.
    ydata : npt.NDArray[np.float32]
        the data along the y-axis.
    axis_index : int
        the index of the primary axis to draw the line to.
    line_index : int
        the index of line to draw to.
    config : LineConfig
        the parameters to use when drawing.
    """

    xdata: npt.NDArray[np.float32]
    ydata: npt.NDArray[np.float32]
    axis_index: int
    line_index: int
    config: LineConfig


class DrawScatterCommand(NamedTuple):
    """A structure that contains the necessary information to draw a scatter plot.

    Attributes
    ----------
    xdata : npt.NDArray[np.float32]
        the data along the x-axis.
    ydata : npt.NDArray[np.float32]
        the data along the y-axis.
    axis_index : int
        the index of the primary axis to draw the line to.
    scatter_index : int
        the index of the scatter plot to draw to.
    config : ScatterConfig
        the parameters to use when drawing.
    """

    xdata: npt.NDArray[np.float32]
    ydata: npt.NDArray[np.float32]
    axis_index: int
    scatter_index: int
    config: ScatterConfig


class SetTitleCommand(NamedTuple):
    """A structure that contains the necessary information to set the title of a subplot.

    Attributes
    ----------
    title : str
        the title to set.
    axis_index : int
        the index of the primary axis to set the title of.
    """

    title: str
    axis_index: int


class SetLegendCommand(NamedTuple):
    """A structure that contains the necessary information to set the legend of a subplot.

    Attributes
    ----------
    legend : list[str]
        the legend.
    axis_index : int
        the index of the primary axis to set the legend of.
    """

    legend: list[str]
    axis_index: int


class SetAxisLabelsCommand(NamedTuple):
    """A structure that contains the necessary information to set the labels of the x and y axis of a subplot.

    Attributes
    ----------
    xlabel : str
        the label of the x axis.
    ylabel : str
        the label of the y axis.
    axis_index : int
        the index of the primary axis to set the title of.
    """

    xlabel: str
    ylabel: str
    axis_index: int


class SetAxisLimitsCommand(NamedTuple):
    """A structure that contains the necessary information to set the labels of the x and y axis of a subplot.

    Attributes
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

    xmin: float | None
    xmax: float | None
    ymin: float | None
    ymax: float | None
    axis_index: int


class SetAutoscaleCommand(NamedTuple):
    """A structure that contains the necessary information to turn on/off autoscaling.

    Attributes
    ----------
    enable : bool
        if `True` will turn on autoscale and if `False` will turn off autoscale.
    axis : str
        the axis to operate on. Allowed choices are "both", "x" and "y".
    axis_index : int
        the index of the set of axes to operate on.
    """

    enable: bool
    axis: str
    axis_index: int


class SetXScaleCommand(NamedTuple):
    """A structure that contains the necessary information to set the scale of the x-axis.

    Attributes
    ----------
    scale : str
        the scale to set to. Allowed choices are "linear", "log", "symlog", "logit", etc.
    axis_index : int
        the index of the set of axes to operate on.
    """

    scale: str
    axis_index: int


class SetYScaleCommand(NamedTuple):
    """A structure that contains the necessary information to set the scale of the y-axis.

    Attributes
    ----------
    scale : str
        the scale to set to. Allowed choices are "linear", "log", "symlog", "logit", etc.
    axis_index : int
        the index of the set of axes to operate on.
    """

    scale: str
    axis_index: int


class RemoveAxisTicksCommand(NamedTuple):
    """A structure that contains the necessary information to remove axis ticks.

    Attributes
    ----------
    axis : str
        the axis to operate on. Allowed choices are "both", "x" and "y".
    axis_index : int
        the index of the set of axes to operate on.
    """

    axis: str
    axis_index: int


class SetFontSizeCommand(NamedTuple):
    """A structure that contains the necessary information to set the font size.

    Attributes
    ----------
    font_size : int
        the size of the font.
    """

    font_size: int


class EndFrameCommand(NamedTuple):
    """A structure used to signify the end of a frame. It contains the frame number of the ended frame.

    Attributes
    ----------
    frame_number : int
        the frame number of the frame to end.
    """

    frame_number: int


def plotting_job(count: int, settings: PlotterConfig, commands: list):
    """This function creates a plot according to a list of commands and then saves the plot as a png.

    Parameters
    ----------
    count : int
        the count of the current figure. This is used to name the plot when it is saved.
    settings : PlotterConfig
        settings used to configure plotting.
    commands : list
        the list of plotting commands.
    """
    # Import pyplot
    import matplotlib.pyplot as plt

    # Create figure
    dpi = DPI
    figsize = (
        settings.figsize[0] / dpi,
        settings.figsize[1] / dpi,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Add title to whole figure
    if settings.title_flag:
        fig.suptitle(settings.title)

    # Make sure that the number of plots and given settings is equal
    num_plots = settings.nrows * settings.ncols

    # Create axes and allocate lists to store image artists and line artists
    axes = []
    images = []
    lines = []
    scatters = []
    for i in range(num_plots):
        current_axis = fig.add_subplot(settings.nrows, settings.ncols, i + 1)
        # Create a primary axis
        axes.append(current_axis)
        # Create a list of images per axis
        images.append({})
        # Create a list of lines per axis
        lines.append({})
        # Create a list of scatter plots per axis
        scatters.append({})

    # Process commands
    for command in commands:
        # Draw image
        if isinstance(command, DrawImageCommand):
            # Unpack data
            data = command.data
            axis_index = command.axis_index
            image_index = command.image_index
            image_config = command.config
            # Create a new image and colorbar if it doesn't already exist
            if image_index not in images[axis_index]:
                images[axis_index][image_index] = axes[axis_index].imshow(
                    data,
                    cmap=image_config.cmap,
                    vmin=image_config.vmin,
                    vmax=image_config.vmax,
                    origin="lower",
                    extent=command.extents,
                    rasterized=True,
                )
                if image_config.colorbar_flag:
                    # Create colorbar
                    fig.colorbar(
                        images[axis_index][image_index],
                        ax=axes[axis_index],
                        fraction=0.046,
                        pad=0.04,
                        label=image_config.colorbar_label,
                    )
            # Otherwise set the data of the image to the new data
            else:
                images[axis_index][image_index].set_data(data)
        # Draw line plot
        elif isinstance(command, DrawPlotCommand):
            # Unpack data
            xdata = command.xdata
            ydata = command.ydata
            axis_index = command.axis_index
            line_index = command.line_index
            line_config = command.config
            # Create a new line if it doesn't already exist
            if line_index not in lines[axis_index]:
                lines[axis_index][line_index] = axes[axis_index].plot(
                    xdata,
                    ydata,
                    color=line_config.color,
                    linestyle=line_config.linestyle,
                )[0]
            else:
                lines[axis_index][line_index].set_data(xdata, ydata)
        # Draw scatter plot
        elif isinstance(command, DrawScatterCommand):
            # Unpack data
            xdata = command.xdata
            ydata = command.ydata
            axis_index = command.axis_index
            scatter_index = command.scatter_index
            scatter_config = command.config
            # Create a new line if it doesn't already exist
            if scatter_index not in scatters[axis_index]:
                scatters[axis_index][scatter_index] = axes[axis_index].scatter(
                    xdata,
                    ydata,
                    facecolors=scatter_config.facecolors,
                    edgecolors=scatter_config.edgecolors,
                    marker=scatter_config.marker,
                    linewidths=scatter_config.linewidths,
                )
            else:
                # Package x and y data into single 2D array
                packaged_data = np.column_stack((xdata, ydata))
                scatters[axis_index][scatter_index].set_offsets(packaged_data)
        # Set title
        elif isinstance(command, SetTitleCommand):
            axes[command.axis_index].set_title(command.title)
        # Set legend
        elif isinstance(command, SetLegendCommand):
            axes[command.axis_index].legend(command.legend)
        # Set axis labels
        elif isinstance(command, SetAxisLabelsCommand):
            axes[command.axis_index].set_xlabel(command.xlabel)
            axes[command.axis_index].set_ylabel(command.ylabel)
        # Set axis limits
        elif isinstance(command, SetAxisLimitsCommand):
            axes[command.axis_index].set_xlim(left=command.xmin, right=command.xmax)
            axes[command.axis_index].set_ylim(bottom=command.ymin, top=command.ymax)
        # Set autoscale
        elif isinstance(command, SetAutoscaleCommand):
            axes[command.axis_index].autoscale(enable=command.enable, axis=command.axis)
        # Set x-axis scale
        elif isinstance(command, SetXScaleCommand):
            axes[command.axis_index].set_xscale(command.scale)
        # Set y-axis scale
        elif isinstance(command, SetYScaleCommand):
            axes[command.axis_index].set_yscale(command.scale)
        # Remove axis ticks
        elif isinstance(command, RemoveAxisTicksCommand):
            axis = command.axis
            axis_index = command.axis_index
            if axis == "both" or axis == "x":
                axes[axis_index].get_xaxis().set_ticks([])
            if axis == "both" or axis == "y":
                axes[axis_index].get_yaxis().set_ticks([])
        # Set font size
        elif isinstance(command, SetFontSizeCommand):
            mpl.rc("font", size=command.font_size)
            mpl.rc("legend", fontsize=command.font_size)
            mpl.rc("figure", titlesize=command.font_size)
            # NOTE: This is super hacky. Not sure why setting rc doesn't work.
            for ax in axes:
                ax.tick_params(axis="both", labelsize=command.font_size)
                ax.set_xlabel(ax.get_xlabel(), fontsize=command.font_size)
                ax.set_ylabel(ax.get_ylabel(), fontsize=command.font_size)
        # End frame
        elif isinstance(command, EndFrameCommand):
            # Save as png
            plt.tight_layout()
            fig.canvas.draw()
            # Construct file name
            src_folder = os.path.dirname(os.path.realpath(__file__))
            file_name = f"{src_folder}{SUB_TO_ROOT}{PLOT_FOLDER}/frame_{command.frame_number}.png"

            # Save figure as png
            fig.savefig(
                file_name,
                facecolor="white",
                transparent=False,
            )

    plt.close(fig)


class MplMultiPlotter(Plotter):
    """This plotter uses `matplotlib` as its plotting backend.
    Plots are saved as a png upon flush in a folder called `plot_cache`, and then upon close the png sequence will be stitched
    together into a mp4 in a folder called `video_cache`.
    This plotter uses a process pool to create the plots in separate processes concurrently to speed up iteration.
    """

    def __init__(
        self, settings: PlotterConfig, progress_callback: Callable[[int], None]
    ):
        # Store progress_callback
        self._progress_callback = progress_callback

        # Set backend to a rasterizer to optimise for pngs.
        mpl.use("Agg")

        # Delete frames in plot_cache
        src_folder = os.path.dirname(os.path.realpath(__file__))
        file_name = f"{src_folder}{SUB_TO_ROOT}{PLOT_FOLDER}/frame_*.png"
        for plot_file in glob.glob(file_name):
            os.remove(plot_file)

        # Store file name to save as
        self._file_name = settings.file_name

        # Initialise frame count
        self._count = 0
        # Intialise comamnd list
        self._commands = []
        # Save configuration
        self._settings = settings
        # Initialise process pool
        self._pool = mp.Pool(max(mp.cpu_count() // 2, 1))

    def reset(self):
        pass

    def flush(self):
        # Add end frame command
        self._commands.append(EndFrameCommand(self._count))

        # Only send plotting task every `FRAMES_IN_FLIGHT` frames
        if self._count % FRAMES_IN_FLIGHT == 0 and self._count > 0:
            self._submit_command_queue()

        # plotting_job(self._count, self._settings, self._commands)

        # Increment count
        self._count += 1

    def close(self):
        # FLush remaining commands
        self._submit_command_queue()
        # Wait until plotting is complete
        self._pool.close()
        self._pool.join()
        # Create mp4
        # Construct file names
        src_folder = os.path.dirname(os.path.realpath(__file__))
        input_file_template = f"{src_folder}{SUB_TO_ROOT}{PLOT_FOLDER}/frame_%d.png"
        output_file = f"{src_folder}{SUB_TO_ROOT}{VIDEO_FOLDER}/{self._file_name}.mp4"
        (
            ffmpeg.input(
                input_file_template,
                framerate=120,
                y=None,
            )
            .output(output_file, crf=25, pix_fmt="yuv420p")
            .overwrite_output()
            .run()
        )

    def draw_image(
        self,
        data: npt.NDArray[np.float32],
        extents: tuple[float, float, float, float],
        axis_index: int,
        image_index: int,
        image_config: ImageConfig,
    ):
        self._commands.append(
            DrawImageCommand(
                data.copy(), extents, axis_index, image_index, image_config
            )
        )

    def draw_plot(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        axis_index: int,
        line_index: int,
        line_config: LineConfig,
    ):
        self._commands.append(
            DrawPlotCommand(x.copy(), y.copy(), axis_index, line_index, line_config)
        )

    def draw_scatter(
        self,
        xdata: npt.NDArray[np.float32],
        ydata: npt.NDArray[np.float32],
        axis_index: int,
        scatter_index: int,
        scatter_config: ScatterConfig,
    ):
        self._commands.append(
            DrawScatterCommand(
                xdata.copy(), ydata.copy(), axis_index, scatter_index, scatter_config
            )
        )

    def set_title(self, title: str, axis_index: int):
        self._commands.append(SetTitleCommand(title, axis_index))

    def set_legend(self, legend: list[str], axis_index: int):
        self._commands.append(SetLegendCommand(legend, axis_index))

    def set_axes_labels(self, xlabel: str, ylabel: str, axis_index: int):
        self._commands.append(SetAxisLabelsCommand(xlabel, ylabel, axis_index))

    def set_axes_limits(
        self,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
        axis_index: int,
    ):
        self._commands.append(
            SetAxisLimitsCommand(x_min, x_max, y_min, y_max, axis_index)
        )

    def set_autoscale(self, enable: bool, axis: str, axis_index: int):
        self._commands.append(SetAutoscaleCommand(enable, axis, axis_index))

    def set_x_scale(self, scale: str, axis_index: int):
        self._commands.append(SetXScaleCommand(scale, axis_index))

    def set_y_scale(self, scale: str, axis_index: int):
        self._commands.append(SetYScaleCommand(scale, axis_index))

    def remove_axis_ticks(self, axis: str, axis_index: int):
        self._commands.append(RemoveAxisTicksCommand(axis, axis_index))

    def set_font_size(self, font_size: int):
        self._commands.append(SetFontSizeCommand(font_size))

    def _submit_command_queue(self):
        """Submits the command queue to a process in the process pool to be executed."""
        # Count number of frames to be rendered
        frame_count = len(
            [None for command in self._commands if isinstance(command, EndFrameCommand)]
        )
        # Send plotting task out to another process using the pool
        self._pool.apply_async(
            plotting_job,
            args=(
                self._count,
                self._settings,
                self._commands,
            ),
            callback=lambda x: self._progress_callback(frame_count),
        )
        # Reset queue
        self._commands = []
