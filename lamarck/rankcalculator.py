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

    def __init__(self, results_df: pd.DataFrame = pd.DataFrame()):
        self._update(results_df, '')

    def _update(self, results_df: pd.DataFrame, out: list | str) -> None:
        self.results = results_df.copy()
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
    def pareto(self, fronts: pd.Series, crowds: pd.Series) -> pd.Series:
        """
        Get the Pareto Ranks based on the `pareto fronts` and the `crowds` Series.
        """
        r1 = fronts.rank(method='dense', ascending=True)
        r2 = crowds.rank(method='dense', ascending=False)
        order1 = int(np.log10(r2.max())) + 1
        factor1 = 10**order1
        return (r1 * factor1 + r2).rank(method='min')

    def pareto_fronts(self, objectives: list[str], p: float) -> pd.Series:
        """
        Get the Pareto Fronts.
        """
        size = len(self.results)
        find_dominators = find_dominators_deco(self.results[self.out], objectives)
        dominators = self.results.apply(find_dominators, axis=1)
        fronts = pd.Series(np.zeros(size), index=self.results.index, dtype=int)
        lenvals = dominators.map(len)
        front = 0
        threshold = size * (1-p)
        while sum(fronts == 0) >= threshold:
            front += 1
            f = lenvals == 0
            fronts[f] = front
            lenvals[f] = None
            dominator_ids = f[f].index
            get_n_dominators = n_dominators_deco(dominator_ids)
            n_dominators = dominators.map(get_n_dominators)
            lenvals[~f] -= n_dominators[~f]
        fronts[fronts == 0] = front + 1
        return fronts.rename('Front')

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


def find_dominators_deco(df, objectives):
    def mapper(x):
        f = np.ones(len(df), dtype=bool)
        feq = np.ones(len(df), dtype=bool)
        for col, objective in zip(df.columns, objectives):
            if objective == 'min':
                check = df[col] <= x[col]
            elif objective == 'max':
                check = df[col] >= x[col]
            feq = feq & (df[col] == x[col])
            f = f & check
        return df.index[f & ~feq]
    return mapper


def n_dominators_deco(ids):
    def mapper(x):
        return len(np.intersect1d(ids, x))
    return mapper


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
