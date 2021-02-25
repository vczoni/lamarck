import numpy as np
import pandas as pd
from copy import deepcopy
from lamarck import Creature


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
        self.get_creature = CreatureGetter(self)
        self.fitness = Fitness(self)

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

    @return_new_pop_deco
    def drop_duplicates(self):
        """
        Returns a Population based on this one, without duplicates.
        """
        return self.datasets.main.drop_duplicates().reset_index(drop=True)

    def copy(self):
        new_pop = Population(self.genome_blueprint)
        new_pop.datasets.absorb(self.datasets)
        return new_pop

    def add_outputs(self, output_df):
        self.datasets._create_output(output_df)

    def add_fitness(self, fitness_df):
        if self.datasets.fitness is None:
            self.datasets._create_fitness(fitness_df)
        else:
            self.datasets._increment_fitness(fitness_df)


class Populator:
    def __init__(self, population):
        self._pop = population

    def add_creature(self, gene):
        pass

    def from_creature_list(self, creature_list):
        pass

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
        self.main = None
        self.input = None
        self.output = None
        self.fitness = None
        self.history = None

    def _set(self, df):
        self._inputcols = df.columns
        self.main = df
        self.input = self._get_input()

    def _get_input(self):
        return create_id(self.main, self._pop.genes)

    def _create_output(self, output_df):
        self.output = pd.concat((self.input, output_df), axis=1)

    def _create_fitness(self, fitness_df):
        self.fitness = self.input.merge(fitness_df).set_index(self.input.index)

    def _increment_fitness(self, incr_df):
        self.fitness = self.fitness.merge(incr_df).set_index(self.input.index)

    def absorb(self, other):
        self.main = other.main
        self.input = other.input
        self.output = other.output
        self.fitness = other.fitness
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


class Fitness:
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
        ascending = check_objective(objective)
        assign_criteria_inplace(pop, output)
        sort_criteria_inplace(pop, 'criteria', ascending)
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
        n_priorities = len(priorities)
        objectives = check_objectives(objectives, n_priorities)
        pop = self._pop.copy()
        for i, output in enumerate(priorities):
            assign_criteria_inplace(pop, output, suffix=i)
        criteria = [f'criteria{i}' for i in range(n_priorities)]
        ascending = [check_objective(objective) for objective in objectives]
        sort_criteria_inplace(pop, criteria, ascending)
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
        n_outputs = len(outputs)
        objectives = check_objectives(objectives, n_outputs)
        pop = self._pop.copy()
        for i, output in enumerate(outputs):
            assign_criteria_inplace(pop, output, suffix=i)
        assign_pareto_fronts_inplace(pop, objectives)
        criteria = ['front', 'crowd']
        ascending = check_objective(['min', 'max'])
        sort_criteria_inplace(pop, criteria, ascending)
        return pop


def check_objectives(objectives, n):
    if isinstance(objectives, str):
        objectives = [objectives] * n
    elif not isinstance(objectives, (list, tuple)):
        raise TypeError(":objectives: must be `list`, `tuple` or `str`")
    return objectives


def check_objective(objective):
    if objective.lower() in ['min', 'minimize']:
        ascending = True
    elif objective.lower() in ['max', 'maximize']:
        ascending = False
    else:
        raise Exception(":objective: must be either 'min' or 'max'.")
    return ascending


def assign_criteria_inplace(pop, output, suffix=''):
    criteria = pop.datasets.output[output]
    kw = {f'criteria{suffix}': criteria}
    fitness_df = pop.datasets.output.assign(**kw)
    pop.add_fitness(fitness_df)


def assign_pareto_fronts_inplace(pop, objectives):
    fitness_df = pop.datasets.fitness.copy()
    index = fitness_df.index
    size = len(fitness_df.input)
    fronts = pd.Series(np.zeros(size), index=index, dtype=int)
    dominants = pd.Series(np.zeros(size), index=index, dtype=object)
    # (...)


def sort_criteria_inplace(pop, criteria, ascending):
    pop.datasets.fitness\
        .sort_values(criteria, ascending=ascending, inplace=True)
