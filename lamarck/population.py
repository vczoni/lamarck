import numpy as np
import pandas as pd
from copy import deepcopy
from lamarck import Creature
from lamarck.fitness import FitnessBuilder
from lamarck.plotter import PopulationPlotter


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
        self.normal_population_level = 0
        # instances
        self.get_creature = CreatureGetter(self)
        self.apply_fitness = FitnessBuilder(self)
        self._set_plotter(PopulationPlotter)

    def __getitem__(self, key):
        return self.get_creature.from_index(key)

    def __len__(self):
        return len(self.datasets.input)

    def __repr__(self):
        specs = [f"{x} ({d['type']})"
                 for x, d in self.genome_blueprint.items()]
        genestr = ', '.join(specs)
        return f"""Population with {len(self)} Creatures with genes {genestr}.
        """

    def __add__(self, other):
        genome_blueprint = deepcopy(self.genome_blueprint)
        df = pd\
            .concat((self.datasets.input, other.datasets.input))\
            .reset_index(drop=True)
        new_pop = Population(genome_blueprint)
        new_pop.populate.from_gene_dataframe(df)
        return new_pop

    def _set_plotter(self, plotter_class):
        self.plot = plotter_class(self)
        self._plotter_class = plotter_class

    def drop_duplicates(self):
        """
        Returns a Population based on this one, without duplicates.
        """
        self.datasets._drop_duplicates()

    def copy(self):
        genome_blueprint = deepcopy(self.genome_blueprint)
        new_pop = Population(genome_blueprint)
        new_pop.datasets._absorb(self.datasets)
        new_pop.define()
        new_pop._set_plotter(self._plotter_class)
        return new_pop

    def define(self):
        """
        Define that this Population's size is the "normal" state.
        """
        self.normal_population_level = len(self)

    def set_output(self, output_df):
        """
        Parameters
        ----------
        :output_df:     `DataFrame`
        """
        self.datasets._set_output(output_df)

    def include_output(self, output_df):
        """
        Parameters
        ----------
        :output_df:     `DataFrame`
        """
        self.datasets._include_output(output_df)

    def add_fitness(self, fitness_df, objectives, fitness_cols=None):
        """
        Parameters
        ----------
        :fitness_df:    `DataFrame`
        :objectives:    `list`
        :fitness_cols:  `list` (Default: None)
        """
        self.datasets._set_fitness(fitness_df, objectives, fitness_cols)

    def reset_fitness_objectives(self, fitness_cols, objectives):
        """
        Parameters
        ----------
        :fitness_cols:  `list`
        :objectives:    `list`
        """
        self.datasets._set_fitness_objectives(fitness_cols, objectives)

    def select(self, p=0.5):
        """
        """
        pop = self.copy()
        pop.datasets._select(p)
        pop.set_generation(self.generation + 1)
        return pop

    def set_generation(self, generation):
        """
        """
        self.generation = generation


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
        self._pop.define()


class PopulationDatasets:
    def __init__(self, population):
        self._pop = population
        self._inputcols = None
        self._fitnesscols = None
        self._objectives = None
        self.input = None
        self.output = None
        self.fitness = None
        self.history = None

    @property
    def index(self):
        return self.input.index

    @property
    def _outputcols(self):
        return [c for c in self.output if c not in self.input]

    def _set(self, df):
        self._inputcols = self._pop.genes
        self.input = create_id(df[self._inputcols])

    def _drop_duplicates(self):
        self.input = self.input.drop_duplicates()
        self.assert_index_to_dataframes()

    def assert_index_to_dataframes(self):
        if self.output is not None:
            self.output = self.output.loc[self.index]
        if self.fitness is not None:
            self.fitness = self.fitness.loc[self.index]
            self._sort_fitness()

    def _set_output(self, output_df):
        final_output_df = pd.concat((self.input, output_df), axis=1)
        self.output = final_output_df

    def _include_output(self, output_df):
        output_df = pd.concat((self.input, output_df), axis=1)
        if self.output is None:
            self._set_output(output_df)
        else:
            self.output = pd.concat((self.output, output_df), axis=1)

    def _set_fitness(self, fitness_df, objectives, fitness_cols=None):
        if fitness_cols is None:
            fitness_cols = list(fitness_df.columns)
        self._set_fitness_objectives(fitness_cols, objectives)
        self.fitness = pd.concat((self.output, fitness_df), axis=1)
        self._sort_fitness()

    def _sort_fitness(self):
        ascending = [is_objective_ascending(objective)
                     for objective in self._objectives]
        self.fitness.sort_values(self._fitnesscols,
                                 ascending=ascending,
                                 inplace=True)

    def _set_fitness_objectives(self, cols, objectives):
        self._fitnesscols = cols
        self._objectives = objectives

    def _add_to_history(self, fitness_df):
        fitness_df_gen = fitness_df.assign(generation=self._pop.generation)
        if self.history is None:
            self.history = fitness_df_gen
        else:
            self.history = pd.concat((self.history, fitness_df_gen))

    def _absorb(self, other):
        self._inputcols = other._inputcols
        self._fitnesscols = other._fitnesscols
        self._objectives = other._objectives
        self.input = other.input
        self.output = other.output
        self.fitness = other.fitness
        self.history = other.history

    def _select(self, p):
        if self.fitness is None:
            raise Exception(
                "Population wasn't tested yet (Fitness Dataset does't exist)."
            )
        else:
            self._add_to_history(self.fitness)
            n_fittest = int(len(self.fitness) * p)
            index = self.fitness[0:n_fittest].index
            self.input = self.input.loc[index]
            self.output = self.output.loc[index]
            self.fitness = self.fitness.loc[index]


def create_id(df):
    df = df.copy()
    colgen = column_aggregator(df)
    cols = tuple(colgen)
    idcol = pd.Series(tuple(zip(*cols))).apply(hash)
    return df.assign(id=idcol).set_index('id')


def column_aggregator(df):
    for col in df.columns:
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


def is_objective_ascending(objective):
    if objective.lower() in ['min', 'minimize']:
        ascending = True
    elif objective.lower() in ['max', 'maximize']:
        ascending = False
    else:
        raise Exception(":objective: must be either 'min' or 'max'.")
    return ascending
