from __future__ import annotations

import pandas as pd


class DataExplorer:
    """
    Plotter class for data visualization.
    """
    class Evolution:
        _plotter: DataExplorer

        def __init__(self, plotter):
            self._plotter = plotter

        def min(self, var: str):
            """
            """

        def mean(self, var: str):
            """
            """

        def max(self, var: str):
            """
            """

    sim_history: pd.DataFrame
    history: Evolution

    def __init__(self, sim_history: pd.DataFrame):
        self.sim_history = sim_history
        self.result_evolution = self.Evolution(self)

    def variable_pair(self, var1: str, var2: str):
        """
        """

    def pareto_fronts(self,
                      x: str,
                      y: str,
                      frontcol: str = 'Front',
                      highlight_front: int = 0,
                      highlight_color: str = 'k'):
        """
        """
