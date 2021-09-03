from __future__ import annotations
from typing import Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from lamarck.rankcalculator import RankCalculator
from lamarck.population import Population
from lamarck.reproduce import Populator
from lamarck.utils import objective_ascending_map


class Optimizer:
    """
    The Optimizer is responsible for storing the all the necessary Simulation
    objects, such as:
        - The relevant simulation datasets
        - The user defined process
    Also it has all the simulation control vars and multiple simulation methods.

    Optimizer Structure
    -------------------
    Optimizer
    ├── population
    ├── process: user defined function
    ├── Datasets as datasets
    │   ├── population: input data
    │   ├── results: output data
    │   ├── fitness: criteria data
    │   ├── history: historic fitness from multiple generations
    │   └── simulation: concatenation of :population:, :results: and :fitness:
    ├── copy()
    ├── run()
    ├── run_multithread()
    ├── Criteria as apply_fitness
    │   ├── single_criteria()
    │   └── MultiCriteria as multi_criteria
    │       ├── ranked()
    │       └── pareto()
    ├── SimulationConfig as config
    │   ├── max_generations
    │   ├── max_stall
    │   ├── p_selection
    │   ├── n_dispute
    │   ├── n_parents
    │   ├── children_per_relation
    │   ├── p_mutation
    │   ├── max_mutated_genes
    │   ├── children_per_mutation
    │   ├── multithread
    │   └── max_workers
    └── Simulator as simulate
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
        _opt: Optimizer
        _outputs: tuple
        population: pd.DataFrame
        results: pd.DataFrame
        fitness: pd.DataFrame
        history: pd.DataFrame

        def __init__(self, opt: Optimizer):
            self._opt = opt

            def get_outputs(func):
                outputs = func.__code__.co_consts
                if isinstance(outputs[-1], tuple):
                    return outputs[-1]
                else:
                    return (outputs[-1],)

            self._outputs = get_outputs(opt.process)
            self._build_initial_data()

        @property
        def simulation(self):
            datasets = (self.population, self.results, self.fitness)
            return pd.concat(datasets, axis=1)

        def _build_initial_data(self) -> None:
            """
            Create the initial Simulation Datasets.

            Datasets
            --------
            :population:    create dataset by copying the data from the Optimizer's :population:
                            attribute and changing the index to the genes's hashs.
            :results:       create dataset with the same hashed index and with empty
                            columns named after the :_outputs: attribute values.
            :fitness:       create empty dataset with the hashed index.
            :history:       create empty dataset with the hashed index.
            """
            self.update_data()
            self.reset_history_data(pd.DataFrame())

        def update_data(self):
            df = self._opt.population.hash_index()
            self.population = df
            self.results = pd.DataFrame(columns=self._outputs, index=df.index)
            self.fitness = pd.DataFrame(index=df.index)

        def assign_fitness(self, fitness_data: pd.DataFrame) -> None:
            self.fitness = fitness_data.copy()

        def reset_history_data(self, history: pd.DataFrame, generation: int = 0) -> None:
            self.history = history.assign(generation=generation)

        def add_history_data(self, history: pd.DataFrame, generation: int = 0) -> None:
            self.history = pd.concat((self.history, history.assign(generation=generation)))

        def get_best_criature(self, outputs: list | str, objectives: list | str) -> pd.Series:
            """
            Get the best creature from the simulation results by selecting the columns to sort
            and the objectives ('min' or 'max') to determine how to sort them.
            """
            if isinstance(outputs, str):
                outputs = [outputs]
            if isinstance(objectives, str):
                objectives = [objectives]
            ascending = [objective_ascending_map[objective] for objective in objectives]
            return self.simulation.sort_values(outputs, ascending=ascending).iloc[0]

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
        _opt: Optimizer

        def __init__(self, opt: Optimizer):
            self._opt = opt

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
            fitness_data = pd.concat(concat_data, axis=1)
            self._opt.datasets.assign_fitness(fitness_data)

        def multi_criteria_ranked(self, outputs: list[str], objectives: list[str]) -> None:
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
            fitness_data = pd.concat(concat_data, axis=1)
            self._opt.datasets.assign_fitness(fitness_data)

        def multi_criteria_pareto(self, outputs: list[str], objectives: list[str]) -> None:
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
            fitness_data = pd.concat(concat_data, axis=1)
            self._opt.datasets.assign_fitness(fitness_data)

    @dataclass
    class SimulationConfig:
        """
        Simulation configurations.

        Configs
        -------
        - max_generations
        - max_stall
        - p_selection
        - n_dispute
        - n_parents
        - children_per_relation
        - p_mutation
        - max_mutated_genes
        - children_per_mutation
        - multithread
        - max_workers
        """
        max_generations: int = 20
        max_stall: int = 5
        p_selection: float = 0.5
        n_dispute: int = 2
        n_parents: int = 2
        children_per_relation: int = 2
        p_mutation: float = 0.1
        max_mutated_genes: int = 1
        children_per_mutation: int = 1
        multithread: bool = True
        max_workers: int | None = None

    class Simulator:
        """
        Run genetic simulations by selecting a Fitness Criteria on selected output(s),
        defining the objectives (maximize/minimize).

        The simulation will run multiple generations and perform a selection criteria,
        then reproductions and mutations, all according to the `config` settings.

        Available Criteria Methods
        --------------------------
        Single Objective            Select an output to maximize or minimize;
        Multi Objective - Ranked    Select multiple outputs and maximize/minimize them in
                                    order of priority;
        Multi Objective - Pareto    Select multiple outputs and find their Pareto Fronts
                                    by attributing a set of maximize/minimize objectives.
        """
        class MultiCriteriaSimulations:
            """
            Multi-objective simulations
            ---------------------------
            - Ranked
            - Pareto
            """
            _opt: Optimizer

            def __init__(self, opt):
                self._opt = opt

            @staticmethod
            def _stall_criteria(
                    outputs: list[str],
                    objectives: list[str]) -> Callable[[int, pd.DataFrame, pd.DataFrame], int]:
                def check_stall(n_stall: int,
                                new_pop: pd.DataFrame,
                                old_pop: pd.DataFrame) -> int:
                    ascendings = [objective_ascending_map[objective] for objective in objectives]
                    new_best = new_pop.sort_values(outputs, ascending=ascendings).iloc[0]
                    old_best = old_pop.sort_values(outputs, ascending=ascendings).iloc[0]
                    stall_condition = True
                    for output, objective in zip(outputs, objectives):
                        ascending = objective_ascending_map[objective]
                        if ascending:
                            stall_condition &= new_best[output] >= old_best[output]
                        else:
                            stall_condition &= new_best[output] <= old_best[output]
                    return n_stall+1 if stall_condition else 0
                return check_stall

            def ranked(self,
                       outputs: list[str],
                       objectives: list[str],
                       quiet: bool = False,
                       seed: int | None = None) -> Optimizer:
                """
                Run optimization with multiple objectives, selecting the best of them with the
                Ranked Criteria.

                Parameters
                ----------
                :outputs:       The process outputs that will be optimized
                :objectives:    The output objectives ('min' for `minimizing` or 'max' for
                                `maximizing`)
                :quiet:         If `False`, a bar will show the simulation progress (default:
                                `False`).
                :seed:          Random Number Generator control (default: None).
                """
                np.random.seed(seed)

                def apply_fitness(opt: Optimizer) -> None:
                    return opt.apply_fitness.multi_criteria_ranked(outputs=outputs,
                                                                   objectives=objectives)

                check_stall = self._stall_criteria(outputs, objectives)

                opt = self._opt.copy()
                return optimize(opt, apply_fitness, check_stall, quiet)

            def pareto(self,
                       outputs: list[str],
                       objectives: list[str],
                       quiet: bool = False,
                       seed: int | None = None) -> Optimizer:
                """
                Run optimization with multiple objectives, selecting the best of them with the
                Pareto Criteria.

                Parameters
                ----------
                :outputs:       The process outputs that will be optimized
                :objectives:    The output objectives ('min' for `minimizing` or 'max' for
                                `maximizing`)
                :quiet:         If `False`, a bar will show the simulation progress (default:
                                `False`).
                :seed:          Random Number Generator control (default: None).
                """
                np.random.seed(seed)

                def apply_fitness(opt: Optimizer) -> None:
                    return opt.apply_fitness.multi_criteria_pareto(outputs=outputs,
                                                                   objectives=objectives)

                check_stall = self._stall_criteria(outputs, objectives)

                opt = self._opt.copy()
                return optimize(opt, apply_fitness, check_stall, quiet)

        _opt: Optimizer
        multi_criteria: MultiCriteriaSimulations

        def __init__(self, opt: Optimizer):
            self._opt = opt
            self.multi_criteria = self.MultiCriteriaSimulations(opt)

        def single_criteria(self,
                            output: str,
                            objective: str,
                            quiet: bool = False,
                            seed: int | None = None) -> Optimizer:
            """
            Run optimization with a single objective.

            Parameters
            ----------
            :output:    The process output that will be optimized
            :objective: The output objective ('min' for `minimizing` or 'max' for `maximizing`)
            :quiet:     If `False`, a bar will show the simulation progress (default: `False`).
            :seed:      Random Number Generator control (default: None).
            """
            np.random.seed(seed)

            def apply_fitness(opt: Optimizer) -> None:
                return opt.apply_fitness.single_criteria(output=output, objective=objective)

            def check_stall(n_stall: int, new_pop: pd.DataFrame, old_pop: pd.DataFrame) -> int:
                ascending = objective_ascending_map[objective]
                new_best = new_pop[output].sort_values(ascending=ascending).iloc[0]
                old_best = old_pop[output].sort_values(ascending=ascending).iloc[0]
                if ascending:
                    stall_condition = new_best >= old_best
                else:
                    stall_condition = new_best <= old_best
                return n_stall+1 if stall_condition else 0

            opt = self._opt.copy()
            return optimize(opt, apply_fitness, check_stall, quiet)

    population: Population
    process: Callable[[], dict]
    datasets: Datasets
    apply_fitness: Criteria
    rank_calculator: RankCalculator
    config: SimulationConfig
    simulate: Simulator

    def __init__(self, population: Population, process: Callable[[], dict]):
        self.process = process
        self.set_population(population)
        self.apply_fitness = self.Criteria(self)
        self.rank_calculator = RankCalculator()
        self.config = self.SimulationConfig()
        self.simulate = self.Simulator(self)

    def set_population(self, population: Population) -> None:
        self.population = population
        self.datasets = self.Datasets(self)

    def update_population(self, population: Population) -> None:
        self.population = population
        self.datasets.update_data()

    def copy(self) -> Optimizer:
        """
        Create a copy of this instance.
        """
        new_opt = Optimizer(self.population.copy(), self.process)
        new_opt.datasets.results = self.datasets.results.copy()
        new_opt.datasets.fitness = self.datasets.fitness.copy()
        new_opt.datasets.history = self.datasets.history.copy()
        return new_opt

    def run(self, quiet: bool = False) -> None:
        """
        Run the process for all genes in the Population.

        Parameters
        ----------
        :quiet: If `False`, a bar will show the simulation progress (default: `False`).

        Updates
        -------
        self.datasets.results DataFrame
        """
        def run_generator_quiet():
            for index, row in self.datasets.population.iterrows():
                yield pd.DataFrame(self.process(**row), index=pd.Series([index], name='id'))

        def run_generator():
            bar = tqdm(total=self.population.size, position=0, desc='Simulating Population')
            for index, row in self.datasets.population.iterrows():
                yield pd.DataFrame(self.process(**row), index=pd.Series([index], name='id'))
                bar.update()
            bar.close()

        if quiet:
            sim_generator = run_generator_quiet()
        else:
            sim_generator = run_generator()
        self.datasets.results = pd.concat(sim_generator)

    def run_multithread(self,
                        max_workers: int | None = None,
                        quiet: bool = False) -> None:
        """
        Run the process for all genes in the Population.

        Parameters
        ----------
        :max_workers:   Number of concurrent workers. If `None`, the algorithm will get the
                        maximum possible (default: `None`).
        :quiet:         If `False`, a bar will show the simulation progress (default: `False`).

        Updates
        -------
        self.datasets.results DataFrame
        """
        def sim(x):
            index, row = x
            return pd.DataFrame(self.process(**row), index=pd.Series([index], name='id'))

        iterable = self.datasets.population.iterrows()
        if quiet:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                creature_result_list = executor.map(sim, iterable)
        else:
            population_size = self.population.size
            creature_result_list = thread_map(
                sim,
                iterable,
                total=population_size,
                position=0,
                desc='Simulating Population',
                max_workers=max_workers)
        self.datasets.results = pd.concat(creature_result_list)


def select_fittest(ranked_pop_data: pd.DataFrame,
                   p: float = 0.5,
                   rank_col: str = 'Rank') -> pd.DataFrame:
    """
    Select the fraction :p: of creatures from a ranked population data based on a Rank column.
    """
    n_selection = int(round(len(ranked_pop_data) * p))
    return ranked_pop_data.sort_values(rank_col)[0:n_selection]


def run_sim(opt: Optimizer, quiet: bool = False) -> Optimizer:
    """
    Creates a copy of the Optimizer and use it to run a simulation for later criteria
    selection.
    """
    if opt.config.multithread:
        opt.run_multithread(max_workers=opt.config.max_workers, quiet=quiet)
    else:
        opt.run(quiet=quiet)
    return opt


def reproduce(n: int,
              parent_pop: Population,
              parent_fitness_data: pd.DataFrame,
              opt: Optimizer) -> Population:
    blueprint = opt.population.blueprint
    # reproduce
    reproduce = Populator(blueprint)
    # sexual
    n_offspring_sexual = np.ceil((1 - opt.config.p_mutation) * n)
    offspring_sexual = reproduce.sexual(
        ranked_pop_data=parent_fitness_data,
        n_offspring=n_offspring_sexual,
        n_dispute=opt.config.n_dispute,
        n_parents=opt.config.n_parents,
        children_per_relation=opt.config.children_per_relation)
    new_pop = offspring_sexual
    # sexual
    n_offspring_asexual = n - n_offspring_sexual
    if n_offspring_asexual > 0:
        offspring_mutation = reproduce.asexual(
            ranked_pop_data=parent_fitness_data,
            n_offspring=n_offspring_asexual,
            n_mutated_genes=opt.config.max_mutated_genes,
            children_per_creature=opt.config.children_per_mutation)
        new_pop += offspring_mutation
    return new_pop


def repopulate(opt: Optimizer) -> Population:
    """
    Select fittest creatures and make them generate the new generation.
    """
    blueprint = opt.population.blueprint
    target_size = opt.population.size
    # select
    fittest_data = select_fittest(ranked_pop_data=opt.datasets.simulation,
                                  p=opt.config.p_selection)
    fittest_pop = Population(fittest_data[blueprint.genes.names], blueprint)
    parent_pop = fittest_pop.copy()
    while fittest_pop.size < target_size:
        pop_short = opt.population.size - fittest_pop.size
        offspring_pop = reproduce(n=pop_short,
                                  parent_pop=parent_pop,
                                  parent_fitness_data=fittest_data,
                                  opt=opt)
        fittest_pop = (fittest_pop + offspring_pop).unique()
    return fittest_pop


def optimize(opt: Optimizer,
             apply_fitness: Callable[[], None],
             check_stall: Callable[[int, pd.DataFrame, pd.DataFrame], int],
             quiet: bool) -> Optimizer:
    """
    Run simulation multiple times until the best Creatures are found.
    """
    n_stall = 0
    config = opt.config
    old_sim_data = opt.datasets.simulation.copy()
    if not quiet:
        pbar1 = tqdm(total=config.max_generations, position=1)
    for i in range(config.max_generations):
        generation = i + 1
        descr = \
            f'Generation {generation} of {config.max_generations} '\
            f'(stall: {n_stall} of {config.max_stall})'
        if not quiet:
            pbar1.set_description(descr)
        # simulate
        simulated_opt = run_sim(opt, quiet)
        # apply fitness
        apply_fitness(opt)
        # add history to optimizer
        opt.datasets.add_history_data(opt.datasets.simulation, generation)
        # check stall
        new_sim_data = opt.datasets.simulation.copy()
        n_stall = check_stall(n_stall, new_sim_data, old_sim_data)
        if not quiet:
            pbar1.update()
        if n_stall > config.max_stall:
            break
        else:
            # repopulate
            new_pop = repopulate(simulated_opt)
            # update opt's population
            opt.update_population(new_pop)
            old_sim_data = new_sim_data
    if not quiet:
        pbar1.close()
    return opt
