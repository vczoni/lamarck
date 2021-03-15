import numpy as np
import pandas as pd
from lamarck.plotter import PopulationPlotterPareto
from lamarck.utils import is_objective_ascending, compare_to_value


class FitnessBuilder:
    def __init__(self, population):
        self._pop = population
        self.multi_objective = MultiObjectiveFitness(self)
        self.reset_constraints()
        self.add_constraint = ConstraintAssistant(self)

    def reset_constraints(self):
        self._constraints = {}

    def set_constraints(self, constraints):
        """
        Set Output Constraints to invalidade outputs that fall off the requirements.

        Parameters
        ----------
        :constraints:   `dict` with the output names as `keys` and the constraints
                        as `values`. The constraints must be tuples with the
                        contraint_type-contraint_value pair. 

        Example
        -------
            - constraints = {
                'x': ('gt', 0),    # x must be Greater Than 0
                'y': ('le', 'x')   # y must be Less or Equal to x
            }

        Available Constraints
        ---------------------
            - 'eq'      EQual to
            - 'neq'     Not EQual to
            - 'lt'      Less Than
            - 'le'      Less than or Equal to
            - 'gt'      Greater Than
            - 'ge'      Greater than or Equal to
        """
        self._constraints = constraints

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
        fconstraints = self.get_constraints_filter()
        fitness_df = get_single_criteria(self._pop, output, fconstraints)
        single_rank = single_rank_deco(output, objective)
        self._pop.set_fitness(fitness_df, [objective], single_rank)

    def get_constraints_filter(self):
        if any(self._constraints):
            output_df = self._pop.datasets.output.copy()
            f = pd.Series(np.ones(len(output_df), dtype=bool),
                          index=self._pop.datasets.index)
            for output, (oper, val) in self._constraints.items():
                if isinstance(val, str):
                    val = output_df[val]
                col = output_df[output]
                f = f & compare_to_value(col, oper, val)
        else:
            f = None
        return f


class MultiObjectiveFitness:
    def __init__(self, fitnessbuilder):
        self._fitnessbuilder = fitnessbuilder
        self._pop = fitnessbuilder._pop

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
        fconstraints = self._fitnessbuilder.get_constraints_filter()
        fitness_df = build_criteria_df(self._pop, priorities, objectives,
                                       fconstraints)
        ranked_rank = ranked_rank_deco(priorities, objectives)
        self._pop.set_fitness(fitness_df, objectives, ranked_rank)

    def pareto(self, outputs, objectives, p=0.5):
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
        :p:             `float` Proportion of the Population that will be classified.
                        (default: 0.5)
        """
        objectives = standardize_objectives(objectives)
        fconstraints = self._fitnessbuilder.get_constraints_filter()
        criteria_df = build_criteria_df(self._pop, outputs, objectives,
                                        fconstraints)
        criteria_df_norm = normalize(criteria_df)
        fronts = get_pareto_fronts(criteria_df_norm, objectives, p)
        crowd = get_pareto_crowds(criteria_df_norm, fronts)
        fitness_df = criteria_df_norm.assign(front=fronts, crowd=crowd)
        self._pop.set_fitness(fitness_df,
                              objectives=['min', 'max'],
                              rank_method=pareto_rank,
                              fitness_cols=['front', 'crowd'])
        self._pop._set_plotter(PopulationPlotterPareto)


def build_criteria_df(pop, criteria, objectives, fconstraints=None):
    n_criteria = len(criteria)
    objectives = assert_objective_list(objectives, n_criteria)
    criteria_dfs = [get_single_criteria(pop, output, fconstraints, suffix=i)
                    for i, output in enumerate(criteria)]
    return pd.concat(criteria_dfs, axis=1)


def assert_objective_list(objectives, n):
    if isinstance(objectives, str):
        objectives = [objectives] * n
    elif not isinstance(objectives, (list, tuple)):
        raise TypeError(":objectives: must be `list`, `tuple` or `str`")
    return objectives


def get_single_criteria(pop, output, fconstraints=None, suffix=''):
    if fconstraints is None:
        criteria = pop.datasets.output[output]
    else:
        criteria = pop.datasets.output[fconstraints][output]
    data = {f'criteria{suffix}': criteria}
    return pd.DataFrame(data, index=criteria.index)


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


def get_pareto_fronts(df, objectives, p):
    size = len(df)
    find_dominators = find_dominators_deco(df, objectives)
    dominators = df.apply(find_dominators, axis=1)
    fronts = pd.Series(np.zeros(size), index=df.index, dtype=int)
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

# ranking methods


def single_rank_deco(output, objective):
    def rank_wrapper(df):
        asc = is_objective_ascending(objective)
        return df[output].rank(method='dense', ascending=asc)
    return rank_wrapper


def ranked_rank_deco(priorities, objectives):
    def rank_wrapper(df):
        ranks = []
        for priority, objective in zip(priorities, objectives):
            asc = is_objective_ascending(objective)
            r = df[priority].rank(method='dense', ascending=asc)
            ranks.append(r)
        rank = ranks[-1]
        for r in ranks[::-1]:
            order = np.ceil(np.log10(r.max()))
            factor = 10**order
            rscore = r * factor + rank
            rank = rscore.rank(method='dense')
        return rank
    return rank_wrapper


def pareto_rank(df):
    r1 = df['front'].rank(method='dense')
    r2 = df['crowd'].rank(method='dense', ascending=False)
    order1 = np.ceil(np.log10(r2.max()))
    factor1 = 10**order1
    return (r1 * factor1 + r2).rank(method='dense')


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


# Constraint Assistant

class ConstraintAssistant:
    def __init__(self, fitnessbuilder):
        self._fitnessbuilder = fitnessbuilder

    def equal_to(self, output, value):
        constraint = {output: ('eq', value)}
        self._fitnessbuilder._constraints.update(constraint)

    def not_equal_to(self, output, value):
        constraint = {output: ('neq', value)}
        self._fitnessbuilder._constraints.update(constraint)

    def less_than(self, output, value):
        constraint = {output: ('lt', value)}
        self._fitnessbuilder._constraints.update(constraint)

    def less_or_equal(self, output, value):
        constraint = {output: ('le', value)}
        self._fitnessbuilder._constraints.update(constraint)

    def greater_than(self, output, value):
        constraint = {output: ('gt', value)}
        self._fitnessbuilder._constraints.update(constraint)

    def greater_or_equal(self, output, value):
        constraint = {output: ('ge', value)}
        self._fitnessbuilder._constraints.update(constraint)
