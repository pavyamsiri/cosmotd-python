# Standard modules
import glob
import os

# External modules
import matplotlib as mpl
from matplotlib import pyplot as plt

# Internal modules
from cosmotd.plot.plotter import PLOT_CACHE
from cosmotd.plot.mpl_plotter.mpl_plotter import MplPlotter
from cosmotd.plot.settings import PlotterConfig


# Go from this file to project root
SUB_TO_ROOT = "/../../../"


class MplPngPlotter(MplPlotter):
    """This plotter uses `matplotlib` as its plotting backend.
    Plots are saved as a png upon flush in a folder called `plot_cache`.
    """

    def __init__(self, settings: PlotterConfig):
        # Set backend to a rasterizer to optimise for pngs.
        mpl.use("Agg")

        # Delete frames in plot_cache
        src_folder = os.path.dirname(os.path.realpath(__file__))
        file_name = f"{src_folder}{SUB_TO_ROOT}{PLOT_CACHE}/frame_*.png"
        for plot_file in glob.glob(file_name):
            os.remove(plot_file)

        # Counter used to name files
        self._count = 0

        # Initialise the actual plotter
        super().__init__(settings)

    def flush(self):
        plt.tight_layout()
        self._fig.canvas.draw()
        # Construct file name
        src_folder = os.path.dirname(os.path.realpath(__file__))
        file_name = f"{src_folder}{SUB_TO_ROOT}{PLOT_CACHE}/frame_{self._count}.png"

        # Save figure as png
        self._fig.savefig(
            file_name,
            facecolor="white",
            transparent=False,
        )
        # Increment frame count
        self._count += 1
