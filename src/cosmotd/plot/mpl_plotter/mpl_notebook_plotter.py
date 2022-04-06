# External modules
import matplotlib as mpl

# Internal modules
from cosmotd.plot.mpl_plotter.mpl_plotter import MplPlotter
from cosmotd.plot.settings import PlotterConfig


class MplNotebookPlotter(MplPlotter):
    """This plotter uses `matplotlib` as its plotting backend. This is to be used for Jupyter notebooks"""

    def __init__(self, settings: PlotterConfig):
        mpl.use("nbAgg")
        super().__init__(settings)
