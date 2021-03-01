import numpy as np
import pandas as pd
from copy import deepcopy
from lamarck import Creature
from lamarck.plotter import PopulationPlotter, PopulationPlotterPareto


def return_new_pop_deco(df_gen_method):
    def wrapper(obj, *a, **kw):
        genome_blueprint = deepcopy(obj.genome_blueprint)
        df = df_gen_method(obj, *a, **kw)
        new_pop = Population(genome_blueprint)
        new_pop.populate.from_gene_dataframe(df)
        return new_pop
    return wrapper


class Population:
    """
    Populations are collections of genes in Pandas DataFrames. The basic flow of
    Population objects in GA Optimizations is:

        1- Define the Genome
        2- Create a Population based on the Genome
        3- Simulate the Population on an Environment
        4- Select the Fittest and eliminate the weaker Creatures
        5- Generate Offspring
        6- Mutate
        7- IF (Converged) THEN proceed ELSE Go To (3)
        8- End

    Example
    -------
    >>> pop_creator = PopulationCreator(genome_blueprint)
    # (step 2)
    >>> pop = pop_creator.create.det(n)
    # will create a population of (number of genes)**n Creatures
    # (step 3)
    >>> env = Environment()
    >>> env.set_process(the_function)
    >>> pop_fit = env.simulate(pop)
    # where "pop_fit" is the population with the results of the simulations
    >>> pareto_fit = pop_fit\
        .fitness\
        .multi_objective\
        .pareto(outputs=['a', 'b'], objective=['min', 'max'])                   # (step 4)
    # (step 5 & 6)
    >>> new_pop = pareto_fit.reproduce.cross_over(p=0.5, p_mutation=0.01)
    """

    def __init__(self, genome_blueprint):
        self.genome_blueprint = genome_blueprint
        self.populate = Populator(self)
        self.datasets = PopulationDatasets(self)
        self.generation = 0
        self.genes = list(genome_blueprint)
        # instances
        self.get_creature = CreatureGetter(self)
        self.apply_fitness = FitnessBuilder(self)
        self._set_plotter(PopulationPlotter)

    def __getitem__(self, key):
        return self.get_creature.from_index(key)

    def __len__(self):
        return len(self.datasets.main)

    def __repr__(self):
        specs = [f"{x} ({d['type']})"
                 for x, d in self.genome_blueprint.items()]
        genestr = ', '.join(specs)
        return f"""Population with {len(self)} Creatures with genes {genestr}.
        """

    @return_new_pop_deco
    def __add__(self, other):
        return pd\
            .concat((self.datasets.main, other.datasets.main))\
            .reset_index(drop=True)

    def _set_plotter(self, plotter_class):
        self.plot = plotter_class(self)

    @return_new_pop_deco
    def drop_duplicates(self):
        """
        Returns a Population based on this one, without duplicates.
        """
        return self.datasets.main.drop_duplicates().reset_index(drop=True)

    def copy(self):
        new_pop = Population(self.genome_blueprint)
        new_pop.datasets._absorb(self.datasets)
        return new_pop

    def add_outputs(self, output_df):
        """
        Parameters
        ----------
        :output_df:     `DataFrame`
        """
        self.datasets._create_output(output_df)

    def add_fitness(self, fitness_df, objectives):
        """
        Parameters
        ----------
        :fitness_df:    `DataFrame`
        :objectives:    `list`
        """
        self.datasets._create_fitness(fitness_df, objectives)

    def reset_fitness_objectives(self, fitness_cols, objectives):
        """
        Parameters
        ----------
        :fitness_cols:  `list`
        :objectives:    `list`
        """
        self.datasets._set_fitness_objectives(fitness_cols, objectives)


class Populator:
    def __init__(self, population):
        self._pop = population

    def add_creature(self, gene):
        """
        """

    def from_creature_list(self, creature_list):
        """
        """

    def from_gene_dataframe(self, dataframe):
        """
        Populates this Population object from a Pandas DataFrame with columns that
        coincide with the Genome Blueprint's Genes and Types.

        Example
        -------
        >>> genome_blueprint
        {'x': {'type': 'numeric',
         'domain': 'int',
         'ranges': {'min': 0, 'max': 10, 'progression': 'linear'}},
        'y': {'type': 'categorical', 'domain': ['A', 'B', 'C']},
        'z': {'type': 'vectorial',
         'domain': ['i', 'j', 'k'],
         'ranges': {'length': 3, 'replace': False}}}

        >>> df
           x  y          z
        0  0  A  (i, j, k)
        1  1  A  (j, i, k)
        2  0  B  (k, i, j)
        3  2  B  (i, j, k)
        4  7  C  (j, i, k)

        >>> pop = Population(genome_blueprint)
        >>> pop.populate.from_gene_dataframe(df)
        """
        self._pop.datasets._set(dataframe.copy())


class PopulationDatasets:
    def __init__(self, population):
        self._pop = population
        self._inputcols = None
        self._outputcols = None
        self._objectives = None
        self._fitnesscols = None
        self._index = None
        self.main = None
        self.input = None
        self.output = None
        self._fitness = None
        self.history = None

    @property
    def fitness(self):
        if self._fitness is None:
            return None
        else:
            ascending = [is_objective_ascending(objective)
                         for objective in self._objectives]
            return self._fitness\
                .sort_values(self._fitnesscols, ascending=ascending)

    def _set(self, df):
        self._inputcols = df.columns
        self.main = df
        self.input = self._get_input()
        self._index = self.input.index

    def _get_input(self):
        return create_id(self.main, self._pop.genes)

    def _create_output(self, output_df):
        self._inputcols = output_df.columns
        self.output = pd.concat((self.input, output_df), axis=1)

    def _create_fitness(self, fitness_df, objectives):
        fitness_cols = list(fitness_df.columns)
        self._set_fitness_objectives(fitness_cols, objectives)
        self._fitness = pd.concat((self.output, fitness_df), axis=1)

    def _set_fitness_objectives(self, cols, objectives):
        self._fitnesscols = cols
        self._objectives = objectives

    def _absorb(self, other):
        self._inputcols = other._inputcols
        self._outputcols = other._outputcols
        self._objectives = other._objectives
        self._fitnesscols = other._fitnesscols
        self._index = other._index
        self.main = other.main
        self.input = other.input
        self.output = other.output
        self._fitness = other._fitness
        self.history = other.history


def create_id(df, columns):
    df = df.copy()
    colgen = column_aggregator(df, columns)
    cols = tuple(colgen)
    def hashfun(x): return hash(x)
    idcol = pd.Series(tuple(zip(*cols))).apply(hashfun)
    return df.assign(id=idcol).set_index('id')


def column_aggregator(df, columns):
    for col in columns:
        yield df[col]


class CreatureGetter:
    def __init__(self, population):
        self._pop = population

    def from_id(self, idval):
        df = self._pop.datasets.input
        row = df.loc[idval]
        return make_creature_from_df_row(row, self._pop.genes)

    def from_index(self, index):
        df = self._pop.datasets.input
        row = df.iloc[index]
        return make_creature_from_df_row(row, self._pop.genes)


def make_creature_from_df_row(row, genes):
    genome = {gene: val for gene, val in row.items() if gene in genes}
    return Creature(genome)


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
        pop = self._pop.copy()
        objective = standardize_objective(objective)
        fitness_df = get_single_criteria(pop, output)
        pop.add_fitness(fitness_df, [objective])
        return pop


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
        pop = self._pop.copy()
        objectives = standardize_objectives(objectives)
        fitness_df = build_criteria_df(pop, priorities, objectives)
        pop.add_fitness(fitness_df, objectives)
        return pop

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
        pop = self._pop.copy()
        objectives = standardize_objectives(objectives)
        criteria_df = build_criteria_df(pop, outputs, objectives)
        criteria_df = normalize(criteria_df)
        fronts = get_pareto_fronts(criteria_df, objectives)
        crowd = get_pareto_crowds(criteria_df, fronts)
        fitness_df = criteria_df.assign(front=fronts).assign(crowd=crowd)
        criteria = ['front', 'crowd']
        fitness_objectives = ['min', 'max']
        pop.add_fitness(fitness_df, [])
        pop.reset_fitness_objectives(criteria, fitness_objectives)
        pop._set_plotter(PopulationPlotterPareto)
        return pop


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


def is_objective_ascending(objective):
    if objective.lower() in ['min', 'minimize']:
        ascending = True
    elif objective.lower() in ['max', 'maximize']:
        ascending = False
    else:
        raise Exception(":objective: must be either 'min' or 'max'.")
    return ascending


def get_single_criteria(pop, output, suffix=''):
    criteria = pop.datasets.output[output]
    data = {f'criteria{suffix}': criteria}
    return pd.DataFrame(data, index=pop.datasets._index)


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
