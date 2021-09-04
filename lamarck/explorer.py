from __future__ import annotations
from typing import Callable

import pandas as pd
import matplotlib

from lamarck.optimizer import Optimizer


class DataExplorer:
    """
    Base Data Explorer class.
    """
    opt: Optimizer

    def __init__(self, opt: Optimizer):
        self.opt = opt

    @property
    def main_data(self) -> pd.DataFrame:
        return self.opt.datasets.population

    def variable_pair(self, x: str, y: str, **kw) -> matplotlib.axes.Axes:
        """
        Scatter plot of two variables.

        Parameters
        ----------
        :x:     `str` for the X axis column.
        :y:     `str` for the Y axis column.
        Key-Word Arguments - Pandas DataFrame Scatter Plot K-W Arguments
        """
        return self.main_data.plot.scatter(x=x, y=y, **kw)


class HistoryExplorer(DataExplorer):
    """
    Plotter class for data visualization of the History dataset.
    """
    @property
    def main_data(self) -> pd.DataFrame:
        return self.opt.datasets.history

    def history(self,
                column: str,
                metric: Callable[[object], object] = max,
                secondary_metric: Callable[[object], object] | None = None,
                gcol: str = 'generation',
                **kw) -> matplotlib.axes.Axes:
        """
        Scatter plot of the evolution of through multiple generations.
        Parameters
        ----------
        :column:            `str` for the Y axis column.
        :metric:            `function` (max, min, numpy.mean, custom, etc.)
        :secondary_metric:  `function` (max, min, numpy.mean, custom, etc.)
        :gcol:              Name of the 'generation' column (default: 'generation').
        Key-Word Arguments - Pandas DataFrame Scatter Plot K-W Arguments
        """
        df = self.main_data.groupby(gcol).agg(metric).reset_index()
        ax = df.plot.scatter(x=gcol, y=column, **kw)
        if secondary_metric is not None:
            df = self\
                .main_data\
                .groupby(gcol)\
                .agg(secondary_metric)\
                .reset_index()
            ax = df.plot.scatter(x=gcol, y=column, ax=ax, color='r', **kw)
        return ax


class ParetoExplorer(DataExplorer):
    """
    Plotter class for data visualization of the Simulation dataset for the Pareto Criteria.
    """
    @property
    def main_data(self) -> pd.DataFrame:
        return self.opt.datasets.simulation

    def fronts(self,
               x: str,
               y: str,
               hlfront: int | list | None = None,
               hlcolor: int | list = 'k',
               show_worst: bool = False,
               colormap: str = 'rainbow',
               frontcol: str = 'Front',
               **kw) -> matplotlib.axes.Axes:
        """
        Plot fronts in a 2D scatter plot the separates each front by color and
        highlight one specific front (if desired).

        Parameters
        ----------
        :x:             `str` for the X axis column.
        :y:             `str` for the Y axis column.
        :hlfront:       `int` or `list` for setting the highlighted front(s).
                        (default: None; if None is set, no fronts will be highlighted).
        :hlcolor:       `str` or `list` for setting the color of the highlighted front
                        (default: 'k').
        :show_worst:    `bool` set True to show the elements that didn't make the cut
                        (default: False).
        :colormap:      `str` for the desired colormap (default: 'rainbow').
        :frontcol:      Name of the 'Front' column (default: 'Front').
        Key-Word Arguments - Pandas DataFrame Scatter Plot K-W Arguments
        """
        dfplot = make_dfplot(self.main_data, show_worst)
        ax = dfplot.plot.scatter(x=x, y=y,
                                 c=frontcol,
                                 colormap=colormap,
                                 sharex=False,
                                 **kw)
        hlfront = set_hlfront(hlfront)
        if hlfront is not None:
            hlcolor = set_hlcolor(hlfront, hlcolor)
            ax = highlight_fronts(dfplot, x, y, hlfront, hlcolor, ax)
        return ax


def make_dfplot(df, show_worst, frontcol):
    if not show_worst:
        worst = df[frontcol].max()
        f = df[frontcol] != worst
        return df[f]
    else:
        return df


def set_hlfront(hlfront):
    if isinstance(hlfront, int):
        hlfront = [hlfront]
    return hlfront


def set_hlcolor(hlfront, hlcolor):
    if isinstance(hlcolor, str):
        return [hlcolor] * len(hlfront)
    elif len(hlcolor) == len(hlfront):
        return hlcolor
    else:
        s = f"""
        Number of `hlfront`s and `hlcolor`s differ.
        (hlfronts: {hlfront} | hlcolors: {hlcolor})
        """
        raise Exception(s)


def highlight_fronts(df, x, y, hlfronts, hlcolors, ax):
    fronts = df['front']
    for front, color in zip(hlfronts, hlcolors):
        ax = df[fronts == front]\
            .plot\
            .scatter(x=x, y=y, ax=ax, color=color)
    return ax
