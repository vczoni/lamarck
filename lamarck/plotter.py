import numpy as np


class PopulationPlotter:
    def __init__(self, pop):
        self._pop = pop

    def variable_pair(self, x, y, **kw):
        """
        Scatter plot of two variables.

        Parameters
        ----------
        :x:     `str` for the X axis column.
        :y:     `str` for the Y axis column.

        Key-Word Arguments - Pandas DataFrame Scatter Plot K-W Arguments
        """
        df = self._pop.datasets.output
        return df.plot.scatter(x=x, y=y, **kw)

    def history(self, column, metric='mean', secondary_metric=None, **kw):
        """
        Scatter plot of the evolution of through multiple generations.

        Parameters
        ----------
        :column:            `str` for the Y axis column.
        :metric:            `str` or `function`
        :secondary_metric:  `str` or `function`

        Key-Word Arguments - Pandas DataFrame Scatter Plot K-W Arguments
        """
        metric = set_metric(metric)
        gcol = 'generation'
        df = self._pop.datasets.history.groupby(gcol).agg(metric).reset_index()
        ax = df.plot.scatter(x=gcol, y=column, **kw)
        if secondary_metric is not None:
            secondary_metric = set_metric(secondary_metric)
            df = self._pop.datasets.history.groupby(gcol).agg(secondary_metric)\
                .reset_index()
            ax = df.plot.scatter(x=gcol, y=column, ax=ax, color='r', **kw)
        return ax


def set_metric(metric):
    if isinstance(metric, str):
        if metric.lower() == 'min':
            metric = min
        elif metric.lower() == 'max':
            metric = max
        elif metric.lower() in ['mean', 'average', 'avg']:
            metric = np.mean
    return metric


class PopulationPlotterPareto(PopulationPlotter):
    def fronts(self, x=None, y=None, hlfront=None, hlcolor='k',
               show_worst=False, colormap='rainbow', **kw):
        """
        Plot fronts in a 2D scatter plot the separates each front by color and
        highlight one specific front (if desired).

        Parameters
        ----------
        :x:             `str` for the X axis column (Default: None; if None is set,
                        the first fitness column will be used).
        :y:             `str` for the Y axis column (Default: None; if None is set,
                        the second fitness column will be used).
        :hlfront:       `int` or `list` for setting the highlighted front(s)
                        (default: None; if None is set, no fronts will be highlighted)
        :hlcolor:       `str` or `list` for setting the color of the highlighted front
                        (default: 'k')
        :show_worst:    `bool` set True to show the elements that didn't make the cut
                        (default: False)
        :colormap:      `str` for the desired colormap (default: 'rainbow')

        Key-Word Arguments - Pandas DataFrame Scatter Plot K-W Arguments
        """
        df = self._pop.datasets.fitness
        x, y = set_xy(x, y, self._pop.datasets._fitnesscols)
        dfplot = make_dfplot(df, show_worst)
        ax = dfplot.plot.scatter(x=x, y=y,
                                 c='front',
                                 colormap=colormap,
                                 sharex=False,
                                 **kw)
        hlfront = set_hlfront(hlfront)
        if hlfront is not None:
            hlcolor = set_hlcolor(hlfront, hlcolor)
            ax = highlight_fronts(dfplot, x, y, hlfront, hlcolor, ax)
        return ax


def set_xy(x, y, fitcols):
    if x is None:
        x = fitcols[0]
    if y is None:
        y = fitcols[1]
    return x, y


def make_dfplot(df, show_worst):
    if show_worst == False:
        worst = df['front'].max()
        f = df['front'] != worst
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
