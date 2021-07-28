from __future__ import annotations

from abc import ABC, abstractclassmethod
from copy import deepcopy
from typing import Callable
import itertools
import math
import pandas as pd
import numpy as np

from lamarck.utils import VectorialOverloadException


class Gene(ABC):
    """
    Abstract class for handling the different types of genes.
    """
    domain: object

    @abstractclassmethod
    def get_linspace(self, n: int) -> np.ndarray:
        pass

    @abstractclassmethod
    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        pass


class NumericGene(Gene):
    """
    Numeric Gene class.
    """
    domain: type
    range: list | tuple

    def __init__(self, specs: dict):
        self.domain = specs['domain']
        self.range = specs['range']

    def get_linspace(self, n: int) -> np.ndarray:
        start = min(self.range)
        end = max(self.range)
        linspaced = np.linspace(start, end, n, dtype=self.domain)
        return np.unique(linspaced)

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        start = min(self.range)
        end = max(self.range)
        if self.domain is int:
            return np.random.randint(start, end+1, n)
        elif self.domain is float:
            return np.random.uniform(start, end, n)
        else:
            raise TypeError(f"Invalid numeric type {self.domain}")


class CategoricalGene(Gene):
    """
    Categorical Gene class.
    """
    domain: list | tuple

    def __init__(self, specs: dict):
        self.domain = specs['domain']

    def get_linspace(self, n: int) -> np.ndarray:
        domain_array = np.array(self.domain)
        end = len(domain_array) - 1
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.choice(self.domain, n)


class VectorialGene(Gene):
    """
    Vectorial Gene class.
    """
    domain: list | tuple
    replacement: bool
    length: int

    def __init__(self, specs: dict):
        self.domain = specs['domain']
        self.replacement = specs['replacement']
        self.length = specs['length']

    def get_linspace(self, n: int) -> np.ndarray:
        if self.replacement:
            vectors = list(itertools.product(self.domain, repeat=self.length))
            end = len(self.domain)**self.length - 1
        else:
            vectors = list(itertools.permutations(self.domain, r=self.length))
            end = math.perm(len(self.domain), self.length) - 1
        domain_array = self._convert_to_array(vectors)
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        vectors = [tuple(np.random.choice(self.domain, self.length, replace=self.replacement))
                   for _ in range(n)]
        return self._convert_to_array(vectors)

    @staticmethod
    def _convert_to_array(ls):
        arr = np.empty(len(ls), dtype=object)
        arr[:] = ls
        return arr


class BooleanGene(Gene):
    """
    Boolean Gene class.
    """
    domain: type

    def __init__(self, specs: None):
        self.domain = bool

    def get_linspace(self, n: None = None) -> np.ndarray:
        return np.array([False, True], dtype=self.domain)

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.randint(0, 2, n, dtype=bool)


genetype_dict = {
    'numeric': NumericGene,
    'categorical': CategoricalGene,
    'vectorial': VectorialGene,
    'boolean': BooleanGene,
}


class GeneCollection:
    """
    Gene Collection.
    """
    _dict: dict

    def __init__(self, blueprint_dict):
        self._dict = {}
        for genename, geneconfig in blueprint_dict.items():
            genetype = geneconfig['type']
            geneclass = genetype_dict[genetype]
            specs = geneconfig['specs']
            gene = geneclass(specs)
            self._dict.update({genename: gene})
        self.__dict__.update(self._dict)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self._dict)[key]
        return self._dict[key]


class BlueprintBuilder:
    """
    Genome Blueprint Builder Class.

    Genome Specifications
    ---------------------
        1. Numeric
            1.1. Domain: type {int, float}
            1.2. Range: `list` or `tuple`

        2. Categorical
            2.1. Domain: `list` or `tuple`

        3. Vectorial
            3.1. Domain: `list` or `tuple`
            3.2. Replacement: bool
            3.3. Length: int

        4. Boolean
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

    def add_numeric_gene(self, name: str, domain: type, range: list | tuple) -> None:
        """
        Add a Numeric gene specs for the blueprint.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    Class of the variable (`int` or `float`).
        :range:     Pair of values for `min` and `max` values.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_numeric_gene(name='x', domain=int, range=[0, 10])
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - x
            |- type: numeric
            |- specs
                |- domain: <class 'int'>
                |- range: [0, 10]
        """
        if domain not in [int, float]:
            raise TypeError("domain must be either int or float")
        genespecs = {
            'type': 'numeric',
            'specs': {'domain': domain,
                      'range': range}
        }
        self._blueprint.update({name: genespecs})

    def add_categorical_gene(self, name: str, domain: list | tuple) -> None:
        """
        Add a Categorical gene specs for the blueprint.

        Parameters
        ----------
        :name:      Gene name.
        :domain:    List of all categories.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_categorical_gene(name='letters', domain=['a', 'b', 'c'])
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - letters
            |- type: categorical
            |- specs
                |- domain: ['a', 'b', 'c']
        """
        genespecs = {
            'type': 'categorical',
            'specs': {'domain': domain}
        }
        self._blueprint.update({name: genespecs})

    def add_vectorial_gene(self, name: str, domain: list | tuple,
                           replacement: bool, length: int) -> None:
        """
        Add a Vectorial gene specs for the blueprint.

        Parameters
        ----------
        :name:          Gene name.
        :domain:        List of all possible categories contained in the vector.
        :replacement:   Flag to define if its permitted to repeat a value in the
                        same vector (Warning: if this is set to False, then the
                        vector CANNOT have duplicate values, which means its
                        :length: parameter CAN'T be greater than the length of
                        its :domain:).
        :length:        Length of the vectors.

        Example
        -------
        >>> builder = BlueprintBuilder()
        >>> builder.add_vectorial_gene(name='sequence',
        ...                            domain=['A', 'T', 'C', 'G'],
        ...                            replacement=True,
        ...                            length=1000)
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - sequence
            |- type: vectorial
            |- specs
                |- domain: ['A', 'T', 'C', 'G']
                |- replacement: True
                |- length: 1000
        """
        if not replacement:
            check_vector_validity(domain, length)
        genespecs = {
            'type': 'vectorial',
            'specs': {'domain': domain,
                      'replacement': replacement,
                      'length': length}
        }
        self._blueprint.update({name: genespecs})

    def add_boolean_gene(self, name: str) -> None:
        """
        Add a Boolean gene specs for the blueprint.

        Parameters
        ----------
        :name:          Gene name.

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
            'type': 'numeric',
            'specs': {'domain': float,
                      'range': [8, 15]}},
        'x': {
            'type': 'numeric',
            'specs': {'domain': float,
                      'range': [0, 10]}},
        'y': {
            'type': 'categorical',
            'specs': {'domain': ['A', 'B', 'C']}},
        'z': {
            'type': 'vectorial',
            'specs': {'domain': [0, 1, 2, 3, 4],
                      'replacement': False,
                      'length': 5}},
        'flag': {
            'type': 'boolean'
            'specs': {}}
    }
    blueprint = Blueprint(blueprint_dict)
    """
    _dict: dict
    _constraints: list
    genes = GeneCollection

    class PopDataGenerator:
        """
        Handles deterministic or random Population dataset creation.
        """
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
                        1. numeric: min=10, max=90, n=5
                            - values = (10, 30, 50, 70, 90)
                        2. categorical: domain=('A', 'B', 'C', 'D', 'E'), n=3
                            - values = ('A', 'C', 'E')
                        3. vectorial: domain=(1, 2, 3, 4), length=3, replacement=True, n=4
                            - values = ((1, 1, 1),
                                        (2, 1, 1),
                                        (3, 1, 1),
                                        (4, 1, 1))
                        5. vectorial: domain=(1, 2, 3, 4), length=3, replacement=False, n=4
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
            ...         'type': 'numeric',
            ...         'specs': {'domain': float,
            ...                   'range': [0, 10]}},
            ...     'y': {
            ...         'type': 'categorical',
            ...         'specs': {'domain': ['A', 'B', 'C']}},
            ...     'z': {
            ...         'type': 'vectorial',
            ...         'specs': {'domain': [0, 1, 2, 3, 4],
            ...                   'replacement': False,
            ...                   'length': 5}},
            ...     'flag': {
            ...         'type': 'boolean',
            ...         'specs': {}}
            ... }
            >>> blueprint = Blueprint(blueprint_dict)

            - Example 1: n as an Integer (n=3)

            >>> data = blueprint.get_pop_data.deterministic(n=3)
            >>> len(data)
            54

            - Example 2: n as a Dictionary
            # disclaimer: Boolean genes doesn't need any value
            >>> n_dict = {'x': 5, 'y': 3, 'z': 6, 'flag': None}
            >>> data = blueprint.get_pop_data.deterministic(n=n_dict)
            >>> len(data)
            180
            """
            if isinstance(n, int):
                n_dict = {gene: n for gene in self._bp._dict}
            else:
                n_dict = n
            ranges = [self._bp.genes[gene].get_linspace(n_dict[gene]) for gene in self._bp._dict]
            data = itertools.product(*ranges)
            columns = list(self._bp._dict)
            data = pd.DataFrame(data=data, columns=columns)
            return self._bp.apply_constraints(data).drop_duplicates()

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
            ...         'type': 'numeric',
            ...         'specs': {'domain': float,
            ...                   'range': [0, 10]}},
            ...     'y': {
            ...         'type': 'categorical',
            ...         'specs': {'domain': ['A', 'B', 'C']}},
            ...     'z': {
            ...         'type': 'vectorial',
            ...         'specs': {'domain': [0, 1, 2, 3, 4],
            ...                   'replacement': False,
            ...                   'length': 5}},
            ...     'flag': {
            ...         'type': 'boolean',
            ...         'specs': {}}
            ... }
            >>> blueprint = Blueprint(blueprint_dict)
            >>> data = blueprint.get_pop_data.random(n=1000)
            >>> len(data)
            1000
            """
            data_dict = {gene: self._bp.genes[gene].get_random(n, seed).tolist()
                         for gene in self._bp._dict}
            data = pd.DataFrame(data_dict)
            return self._bp.apply_constraints(data).drop_duplicates()

    def __init__(self, blueprint_dict: dict):
        self._dict = blueprint_dict
        self.reset_constraints()
        self.genes = GeneCollection(blueprint_dict)
        self.get_pop_data = self.PopDataGenerator(self)

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
        - Genes x and y are numeric and y cannot be bigger than 2*x
        - Code:
        >>> from lamarck import Blueprint
        >>> blueprint_dict = {
        ...     'x': {
        ...         'type': 'numeric',
        ...         'specs': {'domain': int,
        ...                   'range': [0, 10]}},
        ...     'y': {
        ...         'type': 'numeric',
        ...         'specs': {'domain': int,
        ...                   'range': [0, 10]}}
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
