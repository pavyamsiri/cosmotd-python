# Standard modules
import os
from typing import Callable

# External modules
import ffmpeg
from matplotlib import pyplot as plt

# Internal modules
from cosmotd.plot.plotter import PLOT_FOLDER, VIDEO_FOLDER
from cosmotd.plot.mpl_plotter.mpl_png_plotter import MplPngPlotter, SUB_TO_ROOT
from cosmotd.plot.settings import PlotterConfig


class MplMp4Plotter(MplPngPlotter):
    """This plotter uses `matplotlib` as its plotting backend.
    Plots are saved as a png upon flush in a folder called `plot_cache`, and then upon close the png sequence will be stitched
    together into a mp4 in a folder called `video_cache`.
    """

    def __init__(
        self, settings: PlotterConfig, progress_callback: Callable[[int], None]
    ):
        super().__init__(settings, progress_callback)

    def close(self):
        plt.close(self._fig)
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
