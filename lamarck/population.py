from __future__ import annotations

from copy import deepcopy
from typing import Callable
import itertools
import pandas as pd
import numpy as np

from lamarck.genes import GeneCollection
from lamarck.utils import hash_cols, VectorialOverloadException


class Population:
    """
    Population class.
    """
    data: pd.DataFrame
    blueprint: Blueprint

    def __init__(self, data: pd.DataFrame, blueprint: Blueprint):
        self.blueprint = blueprint
        self.reset_data(data)

    def __len__(self):
        return self.size

    def __add__(self, other):
        return self.merge(other)

    @property
    def size(self) -> int:
        return len(self.data)

    @staticmethod
    def empty(blueprint: Blueprint):
        """
        Create empty Population.
        """
        data = pd.DataFrame(columns=blueprint.genes.names)
        return Population(data, blueprint)

    def reset_data(self, data: pd.DataFrame) -> None:
        self.data = data[self.blueprint.genes.names]

    def hash_index(self) -> pd.DataFrame:
        """
        Returns the Population Dataset (which is a Pandas DataFrame) with hashed index.
        """
        hashed_index = hash_cols(self.data)
        return self.data.set_index(hashed_index)

    def copy(self) -> Population:
        """
        Returns a new Population with the same data as this one.
        """
        new_data = self.data.copy()
        return Population(new_data, self.blueprint.copy())

    def merge(self, pop: Population) -> Population:
        """
        Returns a new Population by combines data from other population.

        Example
        -------
        new_pop = pop1.merge(pop2)

        Alternative
        -----------
        new_pop = pop1 + pop2
        """
        new_data = pd.concat((self.data, pop.data)).reset_index(drop=True)
        return Population(new_data, self.blueprint.copy())

    def unique(self) -> Population:
        """
        Returns a new population without duplicated Creatures.
        """
        new_data = self.data.drop_duplicates()
        return Population(new_data, self.blueprint.copy())


class BlueprintBuilder:
    """
    Genome Blueprint Builder Class.

    Genome Specifications
    ---------------------
        1. Integer
            1.1. Domain: `tuple`

        2. Float
            2.1. Domain: `tuple`

        3. Categorical
            3.1. Domain: `tuple`

        4. Boolean

        5. Array
            5.1. Domain: `tuple`
            5.2. Length: int

        6. Set
            6.1. Domain: `tuple`
            6.2. Length: int
    """
    _blueprint: dict

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """
        Resets the Blueprint, making it an empty `dict`.
        """
        self._blueprint = {}

    def get_blueprint(self) -> dict:
        """
        Returns a copy of the actual state of the Blueprint.
        """
        blueprint_dict = deepcopy(self._blueprint)
        return Blueprint(blueprint_dict)

    def add_integer_gene(self, name: str, domain: tuple) -> None:
        """
        Add a Integer gene specs for the blueprint.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    Pair of values for `min` and `max` values.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_integer_gene(name='x', domain=(0, 10))
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - x
            |- type: integer
            |- specs
                |- domain: (0, 10)
        """
        genespecs = {
            'type': 'integer',
            'specs': {'domain': domain}
        }
        self._blueprint.update({name: genespecs})

    def add_float_gene(self, name: str, domain: tuple) -> None:
        """
        Add a Float gene specs for the blueprint.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    Pair of values for `min` and `max` values.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_float_gene(name='x', domain=(0, 10))
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - x
            |- type: float
            |- specs
                |- domain: (0, 10)
        """
        genespecs = {
            'type': 'float',
            'specs': {'domain': domain}
        }
        self._blueprint.update({name: genespecs})

    def add_categorical_gene(self, name: str, domain: tuple) -> None:
        """
        Add a Categorical gene specs for the blueprint.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    `tuple` with all categories.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_categorical_gene(name='letters', domain=('a', 'b', 'c'))
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - letters
            |- type: categorical
            |- specs
                |- domain: ('a', 'b', 'c')
        """
        genespecs = {
            'type': 'categorical',
            'specs': {'domain': domain}
        }
        self._blueprint.update({name: genespecs})

    def add_boolean_gene(self, name: str) -> None:
        """
        Add a Boolean gene specs for the blueprint.

        Parameters
        ----------
        :name:  Gene name.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_boolean_gene(name='match_flag')
        >>> builder.add_boolean_gene(name='is_random')
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - match_flag
            |- type: boolean
            |- specs
        - is_random
            |- type: boolean
            |- specs
        """
        genespecs = {
            'type': 'boolean',
            'specs': {}
        }
        self._blueprint.update({name: genespecs})

    def add_array_gene(self, name: str, domain: list | tuple, length: int) -> None:
        """
        Add an Array gene specs for the blueprint.

        An `Array` Gene is a `Vectorial` where its permitted to repeat a value in the same
        vector.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    `tuple` with all possible categories contained in the vector.
        :length:    Length of the vectors.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_array_gene(name='sequence',
        ...                        domain=('A', 'T', 'C', 'G'),
        ...                        length=1000)
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - sequence
            |- type: array
            |- specs
                |- domain: ('A', 'T', 'C', 'G')
                |- length: 1000
        """
        genespecs = {
            'type': 'array',
            'specs': {'domain': domain,
                      'length': length}
        }
        self._blueprint.update({name: genespecs})

    def add_set_gene(self, name: str, domain: list | tuple, length: int) -> None:
        """
        Add an Set gene specs for the blueprint.

        A `Set` Gene is a `Vectorial` where its NOT permitted to repeat a value in the same
        vector.

        Warning: the vector CANNOT have duplicate values, which means its :length: parameter
        CAN'T be greater than the length of its :domain:.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    `tuple` with all possible categories contained in the vector.
        :length:    Length of the vectors (cannot be grater than the length of its :domain:).

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_set_gene(name='cities',
        ...                      domain=('C1', 'C2', 'C3', 'C4', 'C5', 'C6'),
        ...                      length=5)
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - cities
            |- type: set
            |- specs
                |- domain: ('C1', 'C2', 'C3', 'C4', 'C5', 'C6')
                |- length: 5
        """
        check_vector_validity(domain, length)
        genespecs = {
            'type': 'set',
            'specs': {'domain': domain,
                      'length': length}
        }
        self._blueprint.update({name: genespecs})


def check_vector_validity(domain, length):
    domain_lenght = len(domain)
    if domain_lenght < length:
        raise VectorialOverloadException(length, domain_lenght)


class Blueprint:
    """
    Blueprint object. Creates the gene generators for Creature Creation.

    Blueprint Dict Example
    ----------------------
    blueprint_dict = {
        'w': {
            'type': 'float',
            'specs': {'domain': (8, 15)}},
        'x': {
            'type': 'float',
            'specs': {'domain': (0, 10)}},
        'y': {
            'type': 'categorical',
            'specs': {'domain': ('A', 'B', 'C')}},
        'z': {
            'type': 'vectorial',
            'specs': {'domain': (0, 1, 2, 3, 4),
                      'replacement': False,
                      'length': 5}},
        'w': {
            'type': 'set',
            'specs': {'domain': (0, 1, 2, 3, 4),
                      'length': 5}},
        'flag': {
            'type': 'boolean'
            'specs': {}}
    }
    blueprint = Blueprint(blueprint_dict)
    """
    class PopDataGenerator:
        """
        Handles deterministic or random Population dataset creation.
        """
        _bp: Blueprint

        def __init__(self, _bp: Blueprint):
            self._bp = _bp

        def deterministic(self, n: int | dict) -> pd.DataFrame:
            """
            Deterministically generates genetic combinations by iterating through
            all of the genes's linearly divided values according to the :n: number
            of values per gene.

            Parameters
            ----------
            :n:     Number of evenly spaced values (subdivisions). The values are
                    selected according to each Gene's Blueprint. If a `dict` is passed,
                    the it is required to declare all genes as Keys and their number of
                    subdivisions as Values.

                    Subdivisions Examples:
                        1. float: min=10, max=90, n=5
                            - values = (10, 30, 50, 70, 90)
                        2. categorical: domain=('A', 'B', 'C', 'D', 'E'), n=3
                            - values = ('A', 'C', 'E')
                        3. array: domain=(1, 2, 3, 4), length=3, n=4
                            - values = ((1, 1, 1),
                                        (2, 1, 1),
                                        (3, 1, 1),
                                        (4, 1, 1))
                        4. set: domain=(1, 2, 3, 4), length=3, n=4
                            - values = ((1, 2, 3),
                                        (2, 1, 3),
                                        (3, 1, 2),
                                        (4, 1, 2))
                    - The Population size will normally be the number n to the power
                    of the number of genes, except if some duplicate genomes are
                    generated (e.g. 4 genes and n=5 will end up generating 5**4=625
                    different combinations of solutions).

            Examples
            --------
            >>> blueprint_dict = {
            ...     'x': {
            ...         'type': 'float',
            ...         'specs': {'domain': (0, 10)}},
            ...     'y': {
            ...         'type': 'categorical',
            ...         'specs': {'domain': ('A', 'B', 'C')}},
            ...     'z': {
            ...         'type': 'array',
            ...         'specs': {'domain': (0, 1, 2, 3, 4),
            ...                   'length': 5}},
            ...     'w': {
            ...         'type': 'set',
            ...         'specs': {'domain': (0, 1, 2, 3, 4),
            ...                   'length': 5}},
            ...     'flag': {
            ...         'type': 'boolean',
            ...         'specs': {}}
            ... }
            >>> blueprint = Blueprint(blueprint_dict)

            - Example 1: n as an Integer (n=3)

            >>> data = blueprint.populate.deterministic(n=3)
            >>> len(data)
            162

            - Example 2: n as a Dictionary
            >>> n_dict = {'x': 5, 'y': 3, 'z': 6, 'w': 2, 'flag': 2}
            >>> data = blueprint.populate.deterministic(n=n_dict)
            >>> len(data)
            360
            """
            if isinstance(n, int):
                n_dict = {gene: n for gene in self._bp._dict}
            else:
                n_dict = n

            def get_unique_linspace(gene):
                genes = self._bp.genes[gene].get_linspace(n_dict[gene])
                return np.unique(genes)

            ranges = [get_unique_linspace(gene) for gene in self._bp._dict]
            data = itertools.product(*ranges)
            columns = list(self._bp._dict)
            data = pd.DataFrame(data=data, columns=columns)
            pop_data = self._bp.apply_constraints(data).drop_duplicates()
            return Population(pop_data, self._bp)

        def random(self, n: int, seed: int | None = None) -> pd.DataFrame:
            """
            Randomly generates the population data.

            Parameters
            ----------
            :n:     Number of randomly generated solutions. The values will be randomly
                    generated and assigned to each Genome individually.
                    - The number 'n' will normally end up being the Population size,
                    except if some duplicate genomes are generated or the constraints end
                    up trimming a portion of the data.
            :seed:  Random Number Generator control. If set to 'None', the seed will not
                    be enforced (default: None).

            Examples
            --------
            >>> blueprint_dict = {
            ...     'x': {
            ...         'type': 'float',
            ...         'specs': {'domain': (0, 10)}},
            ...     'y': {
            ...         'type': 'categorical',
            ...         'specs': {'domain': ('A', 'B', 'C')}},
            ...     'z': {
            ...         'type': 'array',
            ...         'specs': {'domain': (0, 1, 2, 3, 4),
            ...                   'length': 5}},
            ...     'w': {
            ...         'type': 'set',
            ...         'specs': {'domain': (0, 1, 2, 3, 4),
            ...                   'length': 5}},
            ...     'flag': {
            ...         'type': 'boolean',
            ...         'specs': {}}
            ... }
            >>> blueprint = Blueprint(blueprint_dict)
            >>> data = blueprint.populate.random(n=1000, seed=123)
            >>> len(data)
            1000
            """
            data_dict = {gene: self._bp.genes[gene].get_random(n, seed).tolist()
                         for gene in self._bp._dict}
            data = pd.DataFrame(data_dict)
            pop_data = self._bp.apply_constraints(data).drop_duplicates()
            return Population(pop_data, self._bp)

    _dict: dict
    _constraints: list
    genes = GeneCollection
    populate: PopDataGenerator

    def __init__(self, blueprint_dict: dict):
        self._dict = blueprint_dict
        self.reset_constraints()
        self.genes = GeneCollection(blueprint_dict)
        self.populate = self.PopDataGenerator(self)

    def __repr__(self):
        strlist = []
        for genename, config in self._dict.items():
            genetype = config['type']
            specs = config['specs']
            specstr = ''.join([f'\n        |- {name}: {val}' for name, val in specs.items()])
            genestr = f'- {genename}\n    |- type: {genetype}\n    |- specs{specstr}'
            strlist.append(genestr)
        return '\n'.join(strlist)

    def show(self) -> None:
        """
        Show blueprint specs.
        """
        print(self)

    def copy(self) -> Blueprint:
        """
        Returns copy of this blueprint.
        """
        bp_dict = deepcopy(self._dict)
        return Blueprint(bp_dict)

    def reset_constraints(self) -> None:
        """
        Resets the Constraints, making it an empty `dict`.
        """
        self._constraints = []

    def add_constraint(self, func: Callable[[], bool]) -> None:
        """
        Add a constraint function fir trimming possible input values based on
        user-defined functions that receive a set of the genes as parameters and
        return a boolean value.

        Parameters
        ----------
        :func:  Python function with same parameters as some (or all) of the genes
                names. Function must return a `bool` value.

        Example
        -------
        - Genes x and y are integers and y cannot be bigger than 2*x
        - Code:
        >>> from lamarck import Blueprint
        >>> blueprint_dict = {
        ...     'x': {
        ...         'type': 'integer',
        ...         'specs': {'domain': (0, 10)}},
        ...     'y': {
        ...         'type': 'float',
        ...         'specs': {'domain': (0, 10)}}
        ... }
        >>> blueprint = Blueprint(blueprint_dict)
        >>> def constraint(x, y): return 2*x >= y
        >>> blueprint.add_constraint(constraint)

        * note how the function :constraint(): has parameters that matches some
          (or all) of the specified genes.
        """
        self._constraints.append(func)

    def apply_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all registered constraint functions for trimming the data.

        Parameters
        ----------
        :data:  DataFrame that has at least all the columns as the parameters
                specified in all declared constraints.
        """
        f = pd.Series(np.ones(len(data), dtype=bool), index=data.index)
        for constraint in self._constraints:
            varnames = constraint.__code__.co_varnames
            nvars = constraint.__code__.co_argcount
            params = list(varnames[0:nvars])
            f = f & data[params].apply(lambda x: constraint(*x), axis=1)
        return data[f]
