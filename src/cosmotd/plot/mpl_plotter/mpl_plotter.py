# External modules
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt

# Internal modules
from cosmotd.plot.plotter import DPI, Plotter
from cosmotd.plot.settings import PlotterConfig, LineConfig, ImageConfig, ScatterConfig


class MplPlotter(Plotter):
    """This plotter uses `matplotlib` as its plotting backend."""

    def __init__(self, settings: PlotterConfig):
        # Create figure
        dpi = DPI
        figsize = (
            settings.figsize[0] / dpi,
            settings.figsize[1] / dpi,
        )
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._fig.suptitle(settings.title)

        # Make sure that the number of plots and given settings is equal
        num_plots = settings.nrows * settings.ncols

        # Create axes and allocate lists to store image artists and line artists
        self._axes = []
        self._images = []
        self._lines = []
        self._scatters = []
        for i in range(num_plots):
            current_axis = self._fig.add_subplot(settings.nrows, settings.ncols, i + 1)
            # Create a primary axis
            self._axes.append(current_axis)
            # Create a list of images per axis
            self._images.append({})
            # Create a list of lines per axis
            self._lines.append({})
            # Create a list of scatter plots (PathCollections) per axis
            self._scatters.append({})

    def reset(self):
        pass

    def flush(self):
        plt.tight_layout()
        self._fig.canvas.draw()

    def close(self):
        plt.close(self._fig)

    def __del__(self):
        plt.close(self._fig)

    def draw_image(
        self,
        data: npt.NDArray[np.float32],
        axis_index: int,
        image_index: int,
        image_config: ImageConfig,
    ):
        # Create a new image and colorbar if it doesn't already exist
        if image_index not in self._images[axis_index]:
            self._images[axis_index][image_index] = self._axes[axis_index].imshow(
                data,
                cmap=image_config.cmap,
                vmin=image_config.vmin,
                vmax=image_config.vmax,
            )
            # NOTE: For some reason calling colorbar just once causes the loop to slow. It might have to do with colorbar
            # calling update when the image gets changed despite it being static. Could create a function that creates a colorbar
            # for a static image that is invisible.
            # Create colorbar
            self._fig.colorbar(
                self._images[axis_index][image_index],
                ax=self._axes[axis_index],
                fraction=0.046,
                pad=0.04,
            )
        # Otherwise set the data of the image to the new data
        else:
            self._images[axis_index][image_index].set_data(data)

    def draw_plot(
        self,
        xdata: npt.NDArray[np.float32],
        ydata: npt.NDArray[np.float32],
        axis_index: int,
        line_index: int,
        line_config: LineConfig,
    ):
        # Create a new line if it doesn't already exist
        if line_index not in self._lines[axis_index]:
            self._lines[axis_index][line_index] = self._axes[axis_index].plot(
                xdata, ydata, color=line_config.color, linestyle=line_config.linestyle
            )[0]
        # Otherwise set the data of the line to the new data
        else:
            self._lines[axis_index][line_index].set_data(xdata, ydata)

    def draw_scatter(
        self,
        xdata: npt.NDArray[np.float32],
        ydata: npt.NDArray[np.float32],
        axis_index: int,
        scatter_index: int,
        scatter_config: ScatterConfig,
    ):
        # Create a new line if it doesn't already exist
        if scatter_index not in self._scatters[axis_index]:
            self._scatters[axis_index][scatter_index] = self._axes[axis_index].scatter(
                xdata,
                ydata,
                facecolors=scatter_config.facecolors,
                edgecolors=scatter_config.edgecolors,
                marker=scatter_config.marker,
                linewidths=scatter_config.linewidths,
            )
        # Otherwise set the data of the scatter plot to the new data
        else:
            # Package x and y data into single 2D array
            packaged_data = np.column_stack((xdata, ydata))
            self._scatters[axis_index][scatter_index].set_offsets(packaged_data)

    def set_title(self, title: str, axis_index: int):
        # Set title
        self._axes[axis_index].set_title(title)

    def set_legend(self, legend: list[str], axis_index: int):
        # Set legend
        self._axes[axis_index].legend(legend)

    def set_axes_labels(self, xlabel: str, ylabel: str, axis_index: int):
        # Set x and y labels
        self._axes[axis_index].set_xlabel(xlabel)
        self._axes[axis_index].set_ylabel(ylabel)

    def set_axes_limits(
        self,
        x_min: float | None,
        x_max: float | None,
        y_min: float | None,
        y_max: float | None,
        axis_index: int,
    ):
        # Set axes limits
        self._axes[axis_index].set_xlim(x_min, x_max)
        self._axes[axis_index].set_ylim(y_min, y_max)
