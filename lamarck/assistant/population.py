import itertools
import numpy as np
import pandas as pd
from lamarck import Population
from lamarck.utils import (random_linear_dist,
                           random_log_dist,
                           deterministic_linear,
                           deterministic_log,
                           vectorial_distribution,
                           vectorial_distribution_set)


class PopulationCreator:
    def __init__(self, genome_blueprint=None):
        self.create = Creator(self)
        self.set_genome_blueprint(genome_blueprint)

    def set_genome_blueprint(self, genome_blueprint):
        self.genome_blueprint = genome_blueprint


class Creator:
    def __init__(self, pop_creator):
        self._pop_creator = pop_creator

    @property
    def _bp(self):
        return self._pop_creator.genome_blueprint

    def det(self, n):
        """
        Create Gene Values Deterministically, distributed according to its Genome
        Blueprint.

        Parameters
        ----------
        :n:     `int` that set the amount of values for every gene. 

        Examples
        --------
        >>> genome_blueprint = {
            'x': {
                'type': 'numeric',
                'domain': 'int',
                'ranges': {
                    'min': 0,
                    'max': 10,
                    'progression': 'linear'}},
            'y': {
                'type': 'categorical',
                'domain': ['i', 'j', 'k']},
            'z': {
                'type': 'vectorial',
                'domain': ['A', 'B', 'C', D, E],
                'ranges': {
                    'length': 3,
                    'replace': False}}
        }
        >>> creator = PopulationCreator(genome_blueprint)
        >>> creator.create.det(5)
        Will create a population with:
        - x values equal to:    [0, 2, 5, 7, 10]
        - y values equal to:    ['i', 'j', 'k']
        - z values equal to:    [('A', 'B', 'C'), ('B', 'A', 'E'), ('C', 'B', 'E'),
                                 ('D', 'C', 'E'), ('E', 'D', 'C')]
        which totalizes 5*3*5 = 75 different genes, or 75 different Creatures in the
        Population.
        """
        if isinstance(n, int):
            return deterministically_make_population(n, self._bp)
        else:
            raise TypeError('Parameter `n` must be `int`.')

    def det_custom(self, ndict):
        """
        Create Gene Values Deterministically, distributed according to its Genome
        Blueprint.

        Parameters
        ----------
        :ndict: `dict` the specify the amount per gene.

        Examples
        --------
        >>> genome_blueprint = {
            'x': {
                'type': 'numeric',
                'domain': 'int',
                'ranges': {
                    'min': 0,
                    'max': 10,
                    'progression': 'linear'}},
            'y': {
                'type': 'categorical',
                'domain': ['i', 'j', 'k']},
            'z': {
                'type': 'vectorial',
                'domain': ['A', 'B', 'C', D, E],
                'ranges': {
                    'length': 3,
                    'replace': False}}
        }
        >>> creator = PopulationCreator(genome_blueprint)
        >>> creator.create.det_custom({'x': 6, 'y': 2})
        Will create a population with:
        - x values equal to:    [0, 2, 4, 6, 8, 10]
        - y values equal to:    ['i', 'k']
        - no values for z,
        which totalizes 6*2 = 12 different genes, or 10 different Creatures in the
        Population. 
        """
        if isinstance(ndict, int):
            return deterministically_make_population_from_dict(ndict, self._bp)
        else:
            raise TypeError('Parameter `ndict` must be `int`.')

    def rand(self, n):
        """
        Create Gene Values Randomly, distributed according to its Genome Blueprint.
        Parameters
        ----------
        :n:     `int` that set the amount of genes that will be made into Creatures in
                the Population.

        Examples
        --------
        >>> >>> genome_blueprint = {
            'x': {
                'type': 'numeric',
                'domain': 'int',
                'ranges': {
                    'min': 0,
                    'max': 10,
                    'progression': 'linear'}},
            'y': {
                'type': 'categorical',
                'domain': ['i', 'j', 'k']},
            'z': {
                'type': 'vectorial',
                'domain': ['A', 'B', 'C', D, E],
                'ranges': {
                    'length': 3,
                    'replace': False}}
        }
        >>> creator = PopulationCreator(genome_blueprint)
        >>> creator.create.rand(50)
        Will create a population with 50 members, with the following values for the
        genes:
        - x values equal to:    any integer between 0 and 10
        - y values equal to:    any object from the list ['i', 'j', 'k']
        - z values equal to:    any sequence of 3 itens from all possible permutations
                                of ['A', 'B', 'C', D, E]
        """
        if isinstance(n, int):
            return randomly_make_population(n, self._bp)
        else:
            raise TypeError('Parameter `n` must be `int`.')


# for deterministic numeric distribution

def deterministically_make_population(n, blueprint):
    n_dict = {name: n for name in blueprint}
    return deterministically_make_population_from_dict(n_dict, blueprint)


def deterministically_make_population_from_dict(ndict, blueprint):
    gene_names = []
    gene_vals = []
    for gene_name, specs in blueprint.items():
        n = ndict[gene_name]
        gene_names.append(gene_name)
        vals = get_vals_from_blueprint(n, specs)
        gene_vals.append(vals)
    genes = itertools.product(*gene_vals)
    gene_df = pd.DataFrame(genes, columns=gene_names)
    pop = Population(blueprint)
    pop.populate.from_genome_dataframe(gene_df)
    return pop


def get_vals_from_blueprint(n, specs):
    gene_type = specs['type']
    domain = specs['domain']
    if gene_type == 'numeric':
        ranges = specs['ranges']
        return generate_numeric_distribution(n, ranges, domain)
    elif gene_type == 'categorical':
        return generate_categorical_distribution(n, domain)
    elif gene_type == 'vectorial':
        ranges = specs['ranges']
        return generate_vectorial_distribution(n, ranges, domain)
    else:
        raise Exception(
            'Gene type must be "numeric", "categorical" or "vectorial"'
            f' but instead got {gene_type}.')


def generate_numeric_distribution(n, ranges, domain):
    start = ranges['min']
    stop = ranges['max']
    progression = ranges['progression']
    dtype = get_dtype(domain)
    if progression == 'linear':
        return deterministic_linear(start, stop, n, dtype)
    elif progression == 'log':
        return deterministic_log(start, stop, n, dtype)
    else:
        raise Exception(
            'Gene progression must be "linear" or "log"'
            f' but instead got {progression}.')


def generate_categorical_distribution(n, domain):
    end = len(domain) - 1
    indexes = np.linspace(0, end, n, dtype=int)
    return np.array(domain)[indexes]


def generate_vectorial_distribution(n, ranges, domain):
    length = ranges['length']
    replace = ranges['replace']
    if replace:
        return vectorial_distribution(n, length, domain)
    else:
        return vectorial_distribution_set(n, length, domain)


def get_dtype(domain):
    if domain == 'int':
        return int
    elif domain == 'float':
        return float


# for randomly numeric distribution


def randomly_make_population(n, blueprint):
    genes = {gene_name: get_random_vals_from_blueprint(n, specs)
             for gene_name, specs in blueprint.items()}
    gene_df = pd.DataFrame(genes)
    pop = Population(blueprint)
    pop.populate.from_genome_dataframe(gene_df)
    return pop


def get_random_vals_from_blueprint(n, specs):
    gene_type = specs['type']
    domain = specs['domain']
    if gene_type == 'numeric':
        ranges = specs['ranges']
        return generate_numeric_random_distribution(n, ranges, domain)
    elif gene_type == 'categorical':
        return generate_categorical_random_distribution(n, domain)
    elif gene_type == 'vectorial':
        ranges = specs['ranges']
        return generate_vectorial_random_distribution(n, ranges, domain)
    else:
        raise Exception(
            'Gene type must be "numeric", "categorical" or "vectorial"'
            f' but instead got {gene_type}.')


def generate_numeric_random_distribution(n, ranges, domain):
    start = ranges['min']
    stop = ranges['max']
    progression = ranges['progression']
    dtype = get_dtype(domain)
    if progression == 'linear':
        return random_linear_dist(start, stop, n, dtype)
    elif progression == 'log':
        return random_log_dist(start, stop, n, dtype)
    else:
        raise Exception(
            'Gene progression must be "linear" or "log"'
            f' but instead got {progression}.')


def generate_categorical_random_distribution(n, domain):
    return np.random.choice(domain, n)


def generate_vectorial_random_distribution(n, ranges, domain):
    replace = ranges['replace']
    length = ranges['length']
    return [tuple(np.random.choice(domain, length, replace=replace))
            for _ in range(n)]
