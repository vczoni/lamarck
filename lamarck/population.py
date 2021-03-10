import numpy as np
import pandas as pd
from copy import deepcopy
from lamarck import Creature
from lamarck.fitness import FitnessBuilder
from lamarck.repopulate import Repopulator
from lamarck.plotter import PopulationPlotter
from lamarck.utils import genome_already_exists, is_objective_ascending


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
    >>> env.simulate(pop)
    # "pop" will now store the results of the simulations
    >>> pop\
        .apply_fitness\
        .multi_objective\
        .pareto(outputs=['a', 'b'], objective=['min', 'max'])
    # (step 4)
    >>> selected_pop = pop.select(p=0.5)
    # (step 5)
    >>> selected_pop.reproduce.tournamet()
    # (step 5)
    >>> selected_pop.reproduce.mutate()
    """

    def __init__(self, genome_blueprint):
        self.genome_blueprint = genome_blueprint
        self.generation = 0
        self.genes = list(genome_blueprint)
        self.natural_population_level = 0
        self._fitness_rank = None
        # instances
        self.populate = Populator(self)
        self.datasets = PopulationDatasets(self)
        self.reproduce = Repopulator(self)
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
        new_pop.populate.from_genome_dataframe(df)
        new_pop.datasets._drop_duplicates()
        return new_pop

    def _set_plotter(self, plotter_class):
        self.plot = plotter_class(self)
        self._plotter_class = plotter_class

    def copy(self):
        genome_blueprint = deepcopy(self.genome_blueprint)
        new_pop = Population(genome_blueprint)
        new_pop.datasets._absorb(self.datasets)
        new_pop.natural_population_level = self.natural_population_level
        new_pop.generation = self.generation
        new_pop._fitness_rank = self._fitness_rank
        new_pop._set_plotter(self._plotter_class)
        return new_pop

    def define(self):
        """
        Define that this Population's size is the "normal" state.
        """
        self.natural_population_level = len(self)

    def save_to_history(self):
        """
        Appends the fitness dataset to the history dataset.
        """
        if self.datasets.fitness is not None:
            if self.generation not in self.datasets._generations:
                self.datasets._add_to_history(self.datasets.fitness)
            else:
                raise Exception("This generation already exists in history.")
        else:
            raise Exception("Fitness dataset not present.")

    def remove_from_history(self, generation):
        """
        Removes a generation from the history dataset.
        """
        f = self.datasets.history.generation != generation
        new_history = self.datasets.history[f]
        self.datasets._update_history(new_history)

    def set_outputs(self, outputs):
        """
        Parameters
        ----------
        :outputs:       `list`
        """
        self.datasets._set_outputs(outputs)

    def add_output(self, output_df):
        """
        Parameters
        ----------
        :output_df:     `DataFrame`
        """
        self.datasets._add_output(output_df)

    def set_fitness(self, fitness_df, objectives, rank_method,
                    fitness_cols=None):
        """
        Parameters
        ----------
        :fitness_df:    `DataFrame`
        :objectives:    `list`
        :rank_method:   `function`
        :fitness_cols:  `list` (Default: None)
        """
        self.datasets._set_fitness(fitness_df, objectives, fitness_cols)
        self._fitness_rank = rank_method

    @property
    def fitness_rank(self):
        if self._fitness_rank is not None:
            return self._fitness_rank(self.datasets.fitness)

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

    def creature(self, creature):
        """
        Adds a Creature to the Population.

        Parameters
        ----------
        :creature:  `Creature` that will be added to the Population.
        """
        self.from_genome(creature.genome)

    def from_genome(self, genome):
        """
        Adds a Creature to the Population, based on its Gene.

        Parameters
        ----------
        :genome:  `dict` representing the `Creature`'s genome.
        """
        raw_genome_df = pd.DataFrame({k: [v] for k, v in genome.items()})
        genome_df = create_id(raw_genome_df)
        self._pop.datasets._add_creature(genome_df)

    def from_creature_list(self, creature_list):
        """
        Adds all Creatures in a `list`.

        Parameters
        ----------
        :creature_list: `list` of Creatures that will be added to the Population.
        """
        for creature in creature_list:
            self.creature(creature)

    def from_genome_list(self, genome_list):
        """
        Adds all Creatures Genomes in a `list`.

        Parameters
        ----------
        :genome_list:   `list` of Genomes that will become the Creatures that will
                        be added to the Population.
        """
        for genome in genome_list:
            self.from_genome(genome)

    def from_genome_dataframe(self, dataframe):
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
        >>> pop.populate.from_genome_dataframe(df)
        """
        self._pop.datasets._set(dataframe.copy())


class PopulationDatasets:
    def __init__(self, population):
        self._pop = population
        self._outputcols = None
        self._fitnesscols = None
        self._objectives = []
        self.input = None
        self.output = None
        self.fitness = None
        self.history = None

    @property
    def index(self): return self.input.index

    @property
    def _inputcols(self): return self._pop.genes

    @property
    def _generations(self):
        if self.history is None:
            return []
        else:
            return self.history.generation.unique().tolist()

    def _set(self, df):
        self.input = create_id(df[self._inputcols])
        self._drop_duplicates()
        self.output = self.input.copy()
        self.fitness = self.input.copy()

    def _set_outputs(self, outputs):
        if self._outputcols is None:
            self._outputcols = outputs
            size = len(self._pop)
            outcols = {}
            for output in outputs:
                col = np.empty(size, dtype=float)
                col.fill(np.nan)
                outcols.update({output: col})
            self.output = self.output.assign(**outcols)

    def _add_output(self, output_df):
        self.output.update(output_df, overwrite=False)

    def _add_creature(self, genome_df):
        self.input = pd.concat((self.input, genome_df))
        self._drop_duplicates()
        self._update_datasets()

    def _update_datasets(self):
        self._drop_duplicates()
        self.output = pd\
            .merge(self.input, self.output, 'left')\
            .set_index(self.index)
        self.fitness = pd\
            .merge(self.input, self.fitness, 'left')\
            .set_index(self.index)

    def _set_fitness(self, fitness_df, objectives, fitness_cols=None):
        if fitness_cols is None:
            fitness_cols = list(fitness_df.columns)
        self._set_fitness_objectives(fitness_cols, objectives)
        self.fitness = pd.concat((self.output, fitness_df), axis=1)
        self._sort_fitness()

    def _drop_duplicates(self):
        self.input = self.input.drop_duplicates()

    @property
    def _ascending(self):
        return [is_objective_ascending(objective)
                for objective in self._objectives]

    def _sort_fitness(self):
        if self._fitnesscols is not None:
            self.fitness.sort_values(self._fitnesscols,
                                     ascending=self._ascending,
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

    def _update_history(self, history_df):
        self.history = history_df

    def _sort_history(self):
        sortcols = self._fitnesscols + ['generation']
        ascending = self._ascending + [True]
        self.history.sort_values(sortcols, ascending=ascending, inplace=True)

    def _absorb(self, other):
        self._outputcols = other._outputcols
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
