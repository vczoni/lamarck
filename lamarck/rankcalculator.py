from __future__ import annotations
import numpy as np
import pandas as pd

from lamarck.utils import objective_ascending_map


def rank_formatter(name):
    def deco(rank_func):
        def wrapper(obj, *a, **kw):
            return rank_func(obj, *a, **kw).astype(int).rename(name)
        return wrapper
    return deco


class RankCalculator:
    """
    Fitness calculations based on the simulation results.
    """
    results: pd.DataFrame
    out: list | str

    def __init__(self, results_df: pd.DataFrame = pd.DataFrame(), out: list | str = ''):
        self.update(results_df=results_df, out=out)

    def update(self, results_df: pd.DataFrame = None, out: list | str = None) -> None:
        if results_df is not None:
            self.results = results_df.copy()
        if out is not None:
            self.out = out

    @rank_formatter('Rank')
    def single(self, objective: str) -> pd.Series:
        """
        Ranks one `output` to optimize according to a defined `objective`.
        """
        return self.results[self.out]\
            .rank(method='min', ascending=objective_ascending_map[objective])

    @rank_formatter('Rank')
    def ranked(self, objectives: list[str]) -> pd.Series:
        """
        Get the Gene Ranks based on a set of `outputs` and `objectives` in order of priority.
        """
        ranks = [
            self.results[priority].rank(method='min',
                                        ascending=objective_ascending_map[objective])
            for priority, objective in zip(self.out, objectives)]
        rank = ranks[-1]
        for r in ranks[::-1]:
            order = int(np.log10(r.max())) + 1
            factor = 10**order
            rscore = r * factor + rank
            rank = rscore.rank(method='min')
        return rank

    @rank_formatter('Rank')
    def pareto(self, objectives: list[str]) -> pd.Series:
        """
        Get the Pareto Ranks based on the `pareto fronts` and the `crowds` Series.
        """
        fronts = self.pareto_fronts(objectives)
        crowds = self.pareto_crowds(fronts)
        r1 = fronts.rank(method='dense', ascending=True)
        r2 = crowds.rank(method='dense', ascending=False)
        order1 = int(np.log10(r2.max())) + 1
        factor1 = 10**order1
        return (r1 * factor1 + r2).rank(method='min')

    def pareto_fronts(self, objectives: list[str]) -> pd.Series:
        """
        Get the Pareto Fronts.
        """
        norm_df = normalize_df_by_objective(self.results, self.out, objectives)
        dominators = get_dominators(norm_df)
        return get_fronts(dominators).rename('Front')

    def pareto_crowds(self, fronts: pd.Series) -> pd.Series:
        """
        Get the Pareto Crowds.
        """
        frontvals = sorted(fronts.unique())
        crowds = pd.Series(np.zeros(len(self.results[self.out])), index=self.results.index)
        for front in frontvals:
            f = fronts == front
            crowds[f] = get_crowd(self.results[f])
        return crowds.rename('Crowd')


def normalize_series_by_objective(series, objective):
    maxval = series.max()
    minval = series.min()
    data_range = maxval - minval
    abs_series = series - minval
    if objective == 'max':
        norm_series = abs_series/data_range
    elif objective == 'min':
        norm_series = 1 - abs_series/data_range
    return norm_series


def normalize_df_by_objective(df, outputs, objectives):
    data_dict = {
        output: normalize_series_by_objective(df[output], objective)
        for output, objective in zip(outputs, objectives)
    }
    return pd.DataFrame(data_dict, index=df.index)


def get_dominators(normalized_df):
    """
    Get the `dominators` based on the `nomalized_by_objective` df.
    """
    def dominator_mapper(row):
        diff = normalized_df - row
        f_equals = (diff == 0).all(axis=1)
        f_dominant = (diff >= 0).all(axis=1)
        return normalized_df.index[~f_equals & f_dominant]
    return normalized_df.apply(dominator_mapper, axis=1)


def get_fronts(dominators):
    """
    Get the array of `front` values base on the `dominators` array.
    """
    def isin_deco(arr):
        def isin(row):
            return row.isin(arr).all()
        return isin

    dom_arr = np.array([])
    front = 1
    fronts = pd.Series(np.zeros(len(dominators)), index=dominators.index)
    for _ in range(9):
        isin = isin_deco(dom_arr)
        f = dominators.apply(isin) & (fronts == 0)
        fronts[f] = front
        dom_arr = np.concatenate((dom_arr, f[f].index.to_numpy()))
        front += 1
    fronts[fronts == 0] = front
    return fronts.astype(int)


def get_crowd(df):
    s = pd.Series(np.zeros(len(df)), index=df.index)
    for _, cs in df.iteritems():
        infval = pd.Series([np.inf])
        si = pd\
            .concat([-infval, cs, infval])\
            .sort_values()
        sfvals = si[2:].values - si[:-2].values
        sf = pd.Series(sfvals, index=si.index[1:-1])
        s += sf
    return s
