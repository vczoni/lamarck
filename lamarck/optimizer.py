from __future__ import annotations
from typing import Callable
import hashlib
import pandas as pd


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
    │   └── history: historic fitness from multiple generations
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
    ├── Criteria as apply
    │   ├── single_criteria()
    │   └── MultiCriteriaSimulations as multi_criteria
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

    process: Callable[[], dict]
    datasets: Datasets

    def __init__(self, population: pd.DataFrame, process: Callable[[], dict]):
        outputs = process.__code__.co_consts[-1]
        self.process = process
        self.datasets = self.Datasets(population, outputs)

    def run(self) -> pd.DataFrame:
        """
        Run the process for all genes in the Population.

        Returns
        -------
        A Pandas DataFrame with the genetic ("input") data and its results ("output")
        data.

        Updates
        -------
        self.datasets.results DataFrame
        """
        sim_generator = (
            pd.DataFrame(self.process(**row), index=pd.Series([index], name='id'))
            for index, row in self.datasets.population.iterrows()
        )
        out_data = pd.concat(sim_generator)
        self.datasets.results.update(out_data)
        self.datasets.results = self.datasets.results.astype(out_data.dtypes)
        return pd.concat((self.datasets.population, out_data), axis=1)
