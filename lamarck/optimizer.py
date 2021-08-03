from __future__ import annotations
from typing import Callable
import hashlib
import pandas as pd

from lamarck.rankcalculator import RankCalculator
from lamarck.progressbars import ProgressBuilder, ProgressBuilderCollection


class Optimizer:
    """
    The Optimizer is responsible for storing the all the necessary Simulation
    objects, such as:
        - The relevant simulation datasets, such as:
        - The user defined process
    Also it has all the simulation control vars and multiple simulation methods.

    Optimizer Structure
    -------------------
    Optimizer
    ├── process: user defined function
    ├── Datasets as datasets
    │   ├── population: input data
    │   ├── results: output data
    │   ├── fitness: criteria data
    │   ├── history: historic fitness from multiple generations
    │   └── simulation: concatenation of :population:, :results: and :fitness:
    ├── SimulationConfig as config
    │   ├── max_generations
    │   ├── max_stall
    │   ├── p_selection
    │   ├── p_elitism
    │   ├── p_tournament
    │   ├── n_dispute
    │   ├── p_mutation
    │   └── max_mutated_genes
    ├── run()
    ├── Criteria as apply_fitness
    │   ├── single_criteria()
    │   └── MultiCriteria as multi_criteria
    │       ├── ranked()
    │       └── pareto()
    └── Simulations as simulate
        ├── single_criteria()
        └── MultiCriteriaSimulations as multi_criteria
            ├── ranked()
            └── pareto()
    """
    class Datasets:
        """
        Pertinent datasets for the simulation control.

        Datasets
        --------
        - population:   Gene data for all creatures in the population ("input" data);
        - results:      Results of the creatures according to each of the user-defined
                        function outputs ("output" data);
        - fitness:      Population Rankings defined by the defined criteria
                        ("performance" data);
        - simulation:   Concatenation of the main datasets (population-results-fitness);
        - history:      Simulation data from previous generations.
        """
        _raw_population: pd.DataFrame
        _outputs: tuple
        population: pd.DataFrame
        results: pd.DataFrame
        fitness: pd.DataFrame
        history: pd.DataFrame

        def __init__(self, population: pd.DataFrame, outputs: tuple):
            self._raw_population = population
            self._outputs = outputs
            self._build_initial_data()

        @property
        def simulation(self):
            datasets = (self.population, self.results, self.fitness)
            return pd.concat(datasets, axis=1)

        @staticmethod
        def _hash(obj: tuple) -> str:
            """
            Custom hashing function.
            """
            s = str(obj).encode()
            return hashlib.sha256(s).hexdigest()

        def _build_initial_data(self) -> None:
            """
            Create the initial Simulation Datasets.

            Datasets
            --------
            :population:    create dataset by copying the :_raw_population:
                            attribute and changing the index to the genes's hashs.
            :results:       create dataset with the same hashed index and with empty
                            columns named after the :_outputs: attribute values.
            :fitness:       create empty dataset with the hashed index.
            :history:       create empty dataset with the hashed index.
            """
            df = self._raw_population.copy()
            # Hashed Index
            cols = (df[col] for col in df.columns)
            hash_index = pd.Series(tuple(zip(*cols))).apply(self._hash)
            hash_index.name = 'id'
            # Population
            self.population = df.set_index(hash_index)
            # Results
            self.results = pd.DataFrame(columns=self._outputs, index=hash_index)
            # Fitness & History
            self.fitness = pd.DataFrame(index=hash_index)
            self.history = pd.DataFrame(index=hash_index)

    class SimulationConfig:
        """
        Simulation configurations.

        Configs
        -------
        - max_generations
        - max_stall
        - p_selection
        - p_elitism
        - p_tournament
        - n_dispute
        - p_mutation
        - max_mutated_genes
        """
        max_generations: int = 50
        max_stall: int = 5
        p_selection: float = 0.5
        p_elitism: float = 0.1
        p_tournament: float = 0.4
        n_dispute: int = 2
        p_mutation: float = 0.05
        max_mutated_genes: int = 1

    class Criteria:
        """
        Apply a Fitness Criteria on selected output(s), defining the objective
        (maximize/minimize).

        Available Criteria Methods
        --------------------------
        Single Objective            Select an output to maximize or minimize;
        Multi Objective - Ranked    Select multiple outputs and maximize/minimize them in
                                    order of priority;
        Multi Objective - Pareto    Select multiple outputs and find their Pareto Fronts
                                    by attributing a set of maximize/minimize objectives.
        """
        class MultiCriteria:
            """
            Dedicaded class for the Multi-Objective Criteria methods.
            """

            _opt: Optimizer

            def __init__(self, opt: Optimizer):
                self._opt = opt

            def ranked(self, outputs: list[str], objectives: list[str]) -> None:
                """
                Select multiple :outputs: to optimize according to a set of defined :objectives:
                in order of priority.

                Parameters
                ----------
                :outputs:       Output Variables.
                :objectives:    Objectives.

                Available Objectives
                --------------------
                'min'   Minimize the output (best genes are the ones that return lower values);
                'max'   Maximize the output (best genes are the ones that return greater values)

                Example
                -------
                opt.apply_fitness.multi_criteria.ranked(outputs=['x', 'y']
                                                        objectives=['max', 'min'])
                ...will rank according to the biggest 'x' values, then to the lowest 'y' values.

                Updates
                -------
                self.datasets.results DataFrame
                """
                self._opt.rank_calculator.update(self._opt.datasets.results, outputs)
                rank = self._opt.rank_calculator.ranked(objectives)
                concat_data = (
                    [self._opt.datasets.results[col].rename(f'Criteria{i+1}')
                     for i, col in enumerate(outputs)]
                    + [rank]
                )
                self._opt.datasets.fitness = pd.concat(concat_data, axis=1)

            def pareto(self, outputs: list[str], objectives: list[str]) -> None:
                """
                Select multiple :outputs: to optimize according to a set of defined :objectives:
                that will define the Pareto Fronts. The Ranking method will consider both the
                `front` and the `crowd` metrics of all genes.

                Parameters
                ----------
                :outputs:       Output Variables.
                :objectives:    Objectives.

                Available Objectives
                --------------------
                'min'   Minimize the output (best genes are the ones that return lower values);
                'max'   Maximize the output (best genes are the ones that return greater values)

                Example
                -------
                opt.apply_fitness.multi_criteria.pareto(outputs=['x', 'y']
                                                        objectives=['max', 'min'])
                ...will rank according to the pareto fronts (ascending) and then the `crowd`
                values (descending).

                Updates
                -------
                self.datasets.results DataFrame
                """
                self._opt.rank_calculator.update(self._opt.datasets.results, outputs)
                fronts = self._opt.rank_calculator.pareto_fronts(objectives)
                crowd = self._opt.rank_calculator.pareto_crowds(fronts)
                rank = self._opt.rank_calculator.pareto(objectives)
                concat_data = (fronts, crowd, rank)
                self._opt.datasets.fitness = pd.concat(concat_data, axis=1)

        _opt: Optimizer
        multi_criteria: MultiCriteria

        def __init__(self, opt: Optimizer):
            self._opt = opt
            self.multi_criteria = self.MultiCriteria(opt)

        def single_criteria(self, output: str, objective: str) -> None:
            """
            Select one :output: to optimize according to a defined :objective:.

            Parameters
            ----------
            :output:       Output Variable.
            :objective:    Objective.

            Available Objectives
            --------------------
            'min'   Minimize the output (best genes are the ones that return lower values);
            'max'   Maximize the output (best genes are the ones that return greater values)

            Updates
            -------
            self.datasets.results DataFrame
            """
            self._opt.rank_calculator.update(self._opt.datasets.results, output)
            rank = self._opt.rank_calculator.single(objective)
            concat_data = (
                self._opt.datasets.results[output].rename('Criteria'),
                rank
            )
            self._opt.datasets.fitness = pd.concat(concat_data, axis=1)

    _bars: ProgressBuilderCollection
    process: Callable[[], dict]
    datasets: Datasets
    config: SimulationConfig
    apply_fitness: Criteria
    rank_calculator: RankCalculator

    def __init__(self, population: pd.DataFrame, process: Callable[[], dict]):
        self._bars = ProgressBuilderCollection()
        outputs = process.__code__.co_consts[-1]
        self.process = process
        self.datasets = self.Datasets(population, outputs)
        self.config = self.SimulationConfig()
        self.apply_fitness = self.Criteria(self)
        self.rank_calculator = RankCalculator()

    def run(self, quiet: bool = False) -> pd.DataFrame:
        """
        Run the process for all genes in the Population.

        If :param quiet: is set to False, a bar will show the simulation progress.

        Updates
        -------
        self.datasets.results DataFrame
        """
        def run_generator_quiet():
            for index, row in self.datasets.population.iterrows():
                yield pd.DataFrame(self.process(**row), index=pd.Series([index], name='id'))

        def run_generator():
            creatures = len(self.datasets.population)
            pbar = ProgressBuilder(creatures, 20)
            self._bars.add_builder('Creature', pbar)
            self._bars.Creature.start_timer()
            for i, (index, row) in enumerate(self.datasets.population.iterrows()):
                yield pd.DataFrame(self.process(**row), index=pd.Series([index], name='id'))
                self._bars.update(Creature=i)
                self._bars.print()

        if quiet:
            sim_generator = run_generator_quiet()
        else:
            sim_generator = run_generator()
        self.datasets.results = pd.concat(sim_generator)
