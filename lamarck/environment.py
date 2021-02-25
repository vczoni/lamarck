import pandas as pd
import numpy as np
import concurrent
from lamarck import Creature, Population


class Environment:
    def __init__(self):
        self.config = EnvironmentConfig()

    def simulate(self, obj):
        """
        Simulates a `Creature` or a `Population` that has the same genome as the Process's
        Input Parameters.

        Parameters
        ----------
        :obj:   `Creature` or `Population` object

        Examples
        --------
        >>> env.simulate(creature)
        {'out_1': 10, 'out_2': 16.8}

        >>> env.simulate(population)
        Population with <n> Creatures with genes <gene1>, <gene2>, (...).
        """
        if isinstance(obj, Creature):
            return self._simulate_creature(obj)
        elif isinstance(obj, Population):
            return self._simulate_population(obj)
        else:
            raise Exception(":obj: must be a Creature or a Population.")

    def _simulate_creature(self, creature):
        return self.config.process(**creature.genome)

    def _simulate_population(self, pop):
        if self.config.multi:
            return self._simulate_multithread(pop)
        else:
            return self._simulate_serial(pop)

    def _simulate_serial(self, pop):
        fitness_dict = {creature.id: self._simulate_creature(creature)
                        for creature in pop}
        return build_output(fitness_dict, pop)

    def _simulate_multithread(self, pop):
        fitness_dict = {}

        def func(creature):
            fitness = self._simulate_creature(creature)
            fitness_dict.update({creature.id: fitness})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(func, pop)
        return build_output(fitness_dict, pop)


def build_output(fitness_dict, pop):
    pop = pop.copy()
    df = build_output_df_from_fitness_dict(fitness_dict)
    pop.add_outputs(df)
    return pop


def build_output_df_from_fitness_dict(fitness_dict):
    index = list(fitness_dict.keys())
    fitness_sample = fitness_dict[next(iter(fitness_dict))]
    outputs = {key: [] for key in fitness_sample}
    for data in fitness_dict.values():
        for output in outputs.keys():
            outputs[output].append(data[output])
    return pd.DataFrame(outputs, index=index)


class EnvironmentConfig:
    def __init__(self):
        self.multi = False
        self.output_varibles = None
        self.process = None

    def __repr__(self):
        attrlist = [f'{var}:   {val}' for var, val in self.__dict__.items()]
        return '\n'.join(attrlist)

    def set_multi(self, multi):
        """
        Set Multi Thread simulation (True or False).
        """
        self.multi = multi

    def set_output_varibles(self, *output_variables):
        """
        Set the names of the output variables (represented by the keys of the
        `dict` that the process function returns).

        Parameters
        ----------
        :output_variables:  `str` or `list` of `str` with the name(s) of the
                            output variable(s) 
        """
        if isinstance(output_variables, str):
            output_variables = (output_variables,)
        elif not isinstance(output_variables, (list, tuple)):
            raise TypeError(':output_variables: must be `list` or `str`.')
        self.output_varibles = output_variables

    def set_process(self, process):
        """
        Sets the Behaviour of this Environment's nature.

        Parameters
        ----------
        :process:   `function` that MUST have only the Genome Blueprint's Genes
                    as parameters and must return the output as a dictionary

        Example
        -------
        genome_blueprint = {
            'x': {
                'type': 'numeric',
                'domain': 'int',
                'ranges': {
                    'min': 0,
                    'max': 10,
                    'progression': 'linear'}},
            'y': {
                'type': 'numeric',
                'domain': 'float',
                'ranges': {
                    'min': 0,
                    'max': 1,
                    'progression': 'linear'}}

        def func(x, y):
            return {
                'out_1': x * y,
                'out_2': x**y - x
            }

        env = Environment()
        env.set_process(func)

        This way, the Environment will be prepare to simulate any Population that
        has this Genome Blueprint.
        """
        self.process = process
