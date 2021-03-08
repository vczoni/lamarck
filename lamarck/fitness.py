import numpy as np
import pandas as pd

from lamarck.plotter import PopulationPlotterPareto


class FitnessBuilder:
    def __init__(self, population):
        self._pop = population
        self.multi_objective = MultiObjectiveFitness(population)

    def single_objective(self, output, objective):
        """
        Define an Output Variable as the objective and an Objective ('maximize'
        or 'minimize'.)

        Parameters
        ----------
        :output:    Variable name (`str`)
        :objective: Objective (`str`).
                    Objective options:
                    - 'minimize' (or 'min')
                    - 'maximize' (or 'max')
        """
        objective = standardize_objective(objective)
        fitness_df = get_single_criteria(self._pop, output)
        self._pop.add_fitness(fitness_df, [objective])


class MultiObjectiveFitness:
    def __init__(self, population):
        self._pop = population

    def ranked(self, priorities, objectives):
        """
        Define a list of Output Variables as the objective and a list of
        Objectives in the respective order.

        Parameters
        ----------
        :priorities:    `list` of Output Variables. The order of the output implies
                        the order of priority, meaning that the first output is the
                        one that is targeted to determine the Creature's Fitness,
                        but in case of a Draw, the next one is considered, and so on.
        :objectives:    `list` of Objectives. A `str` of one objective is allowed,
                        meaning all objectives are the same as the one that was
                        passed.
                        Objective options:
                        - 'minimize' (or 'min')
                        - 'maximize' (or 'max')
        """
        objectives = standardize_objectives(objectives)
        fitness_df = build_criteria_df(self._pop, priorities, objectives)
        self._pop.add_fitness(fitness_df, objectives)

    def pareto(self, outputs, objectives):
        """
        Define a list of Output Variables as the objective and a list of
        Objectives in the respective order. All Output Variables are treated
        equally, and there are no "Best" Creatures here, but fronts of
        creatures that offer different types of unique trade-offs.

        Parameters
        ----------
        :outputs:       `list` of Output Variables.
        :objectives:    `list` of Objectives. A `str` of one objective is allowed,
                        meaning all objectives are the same as the one that was
                        passed.
                        Objective options:
                        - 'minimize' (or 'min')
                        - 'maximize' (or 'max')
        """
        objectives = standardize_objectives(objectives)
        criteria_df = build_criteria_df(self._pop, outputs, objectives)
        criteria_df_norm = normalize(criteria_df)
        fronts = get_pareto_fronts(criteria_df_norm, objectives)
        crowd = get_pareto_crowds(criteria_df_norm, fronts)
        fitness_df = criteria_df_norm.assign(front=fronts, crowd=crowd)
        self._pop.add_fitness(fitness_df,
                              objectives=['min', 'max'],
                              fitness_cols=['front', 'crowd'])
        self._pop._set_plotter(PopulationPlotterPareto)


def build_criteria_df(pop, criteria, objectives):
    n_criteria = len(criteria)
    objectives = assert_objective_list(objectives, n_criteria)
    criteria_dfs = [get_single_criteria(pop, output, suffix=i)
                    for i, output in enumerate(criteria)]
    return pd.concat(criteria_dfs, axis=1)


def assert_objective_list(objectives, n):
    if isinstance(objectives, str):
        objectives = [objectives] * n
    elif not isinstance(objectives, (list, tuple)):
        raise TypeError(":objectives: must be `list`, `tuple` or `str`")
    return objectives


def get_single_criteria(pop, output, suffix=''):
    criteria = pop.datasets.output[output]
    data = {f'criteria{suffix}': criteria}
    return pd.DataFrame(data, index=pop.datasets.index)


def standardize_objectives(objectives):
    if isinstance(objectives, str):
        return standardize_objective(objectives)
    else:
        return [standardize_objective(objective) for objective in objectives]


def standardize_objective(objective):
    if objective.lower() in ['min', 'minimize']:
        return 'min'
    elif objective.lower() in ['max', 'maximize']:
        return 'max'
    else:
        raise Exception(":objective: must be either 'min' or 'max'.")


def normalize(df):
    return (df - df.mean()) / df.std()


def get_pareto_fronts(df, objectives):
    size = len(df)
    find_dominators = find_dominators_deco(df, objectives)
    dominators = df.apply(find_dominators, axis=1)
    fronts = pd.Series(np.zeros(size), index=df.index, dtype=int)
    lenvals = dominators.map(len)
    front = 0
    while sum(fronts == 0) > size/2:
        front += 1
        f = lenvals == 0
        fronts[f] = front
        lenvals[f] = None
        dominator_ids = f[f].index
        get_n_dominators = n_dominators_deco(dominator_ids)
        n_dominators = dominators.map(get_n_dominators)
        lenvals[~f] -= n_dominators[~f]
    fronts[fronts == 0] = front + 1
    return fronts


def get_pareto_crowds(df, fronts):
    frontvals = sorted(fronts.unique())
    crowds = pd.Series(np.zeros(len(df)), index=df.index)
    for front in frontvals:
        f = fronts == front
        crowds[f] = get_crowd(df[f])
    return crowds


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


# decorators

def find_dominators_deco(df, objectives):
    def mapper(x):
        f = np.ones(len(df), dtype=bool)
        for col, objective in zip(df.columns, objectives):
            if objective == 'min':
                check = df[col] < x[col]
            elif objective == 'max':
                check = df[col] > x[col]
            f = f & check
        return df.index[f]
    return mapper


def n_dominators_deco(ids):
    def mapper(x):
        return len(np.intersect1d(ids, x))
    return mapper
