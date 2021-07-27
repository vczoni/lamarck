from __future__ import annotations

from abc import ABC, abstractclassmethod
from copy import deepcopy
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


class CategoricalGene(Gene):
    """
    Categorical Gene class.
    """
    domain: list | tuple

    def __init__(self, specs: dict):
        self.domain = specs['domain']

    def get_linspace(self, n: int):
        domain_array = np.array(self.domain)
        end = len(domain_array) - 1
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]


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

    def get_linspace(self, n: int):
        if self.replacement:
            vectors = list(itertools.product(self.domain, repeat=self.length))
            end = len(self.domain)**self.length - 1
        else:
            vectors = list(itertools.permutations(self.domain, r=self.length))
            end = math.perm(len(self.domain), self.length) - 1
        domain_array = np.array(vectors)
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]


class BooleanGene(Gene):
    """
    Boolean Gene class.
    """
    domain: type

    def __init__(self):
        self.domain = bool

    def get_linspace(self, n: None = None) -> np.ndarray:
        return np.array([False, True], dtype=self.domain)


class GeneCollection:
    """
    Gene Collection.
    """
    _dict: dict

    def __init__(self, blueprint_dict):
        self._dict = {}
        for genename, specs in blueprint_dict.items():
            genetype = specs['type']
            if genetype == 'numeric':
                gene = NumericGene(specs)
            elif genetype == 'categorical':
                gene = CategoricalGene(specs)
            elif genetype == 'vectorial':
                gene = VectorialGene(specs)
            elif genetype == 'boolean':
                gene = BooleanGene()
            else:
                raise Exception(f"Invalid gene type '{genetype}'.")
            self._dict.update({genename: gene})
        self.__dict__.update(self._dict)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self._dict)[key]
        return self._dict[key]


class GenomeBlueprintBuilder:
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
        >>> builder = GenomeBlueprintBuilder()
        >>> builder.add_numeric_gene(name='x', domain=int, range=[0, 10])
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - x
            |- type: numeric
            |- domain: <class 'int'>
            |- range: [0, 10]
        """
        if domain not in [int, float]:
            raise TypeError("domain must be either int or float")
        genespecs = {
            'type': 'numeric',
            'domain': domain,
            'range': range
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
        >>> builder = GenomeBlueprintBuilder()
        >>> builder.add_categorical_gene(name='letters', domain=['a', 'b', 'c'])
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - letters
            |- type: categorical
            |- domain: ['a', 'b', 'c']
        """
        genespecs = {
            'type': 'categorical',
            'domain': domain
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
        >>> builder = GenomeBlueprintBuilder()
        >>> builder.add_vectorial_gene(name='sequence',
        ...                            domain=['A', 'T', 'C', 'G'],
        ...                            replacement=True,
        ...                            length=1000)
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - sequence
            |- type: vectorial
            |- domain: ['A', 'T', 'C', 'G']
            |- replacement: True
            |- length: 1000
        """
        if not replacement:
            check_vector_validity(domain, length)
        genespecs = {
            'type': 'vectorial',
            'domain': domain,
            'replacement': replacement,
            'length': length
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
        >>> builder = GenomeBlueprintBuilder()
        >>> builder.add_boolean_gene(name='match_flag')
        >>> builder.add_boolean_gene(name='is_random')
        >>> blueprint = builder.get_blueprint()
        >>> blueprint.show()
        - match_flag
            |- type: boolean
        - is_random
            |- type: boolean
        """
        genespecs = {
            'type': 'boolean'
        }
        self._blueprint.update({name: genespecs})


def check_vector_validity(domain, length):
    domain_lenght = len(domain)
    if domain_lenght < length:
        raise VectorialOverloadException(length, domain_lenght)


class Blueprint:
    """
    Blueprint object. Creates the gene generators for Creature Creation.
    """
    _dict: dict
    _constraints: dict
    genes = GeneCollection

    class PopDataGenerator:
        """
        Handles deterministic or random Population dataset creation.
        """
        def __init__(self, _bp: Blueprint):
            self._bp = _bp

        def deterministic(self, n: int) -> pd.DataFrame:
            """
            Deterministically generates genetic combinations by iterating through
            all of the genes's linearly divided values according to the :n: number
            of values per gene.

            Parameters
            ----------
            :n:     Number of evenly spaced values (subdivisions). The values are
                    selected according to each Gene's Blueprint.
                    Examples:
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
            """
            ranges = [gene.get_linspace(n) for gene in self._bp.genes]
            data = itertools.product(*ranges)
            columns = list(self._bp._dict)
            return pd.DataFrame(data=data, columns=columns)

    def __init__(self, blueprint_dict: dict):
        self._dict = blueprint_dict
        self.reset_constraints()
        self.genes = GeneCollection(blueprint_dict)
        self.get_pop_data = self.PopDataGenerator(self)

    def show(self) -> None:
        """
        Show blueprint specs.
        """
        showstr = '\n'.join([
            f'- {gene}' + ''.join([f'\n    |- {key}: {val}' for key, val in specs.items()])
            for gene, specs in self._dict.items()
        ])
        print(showstr)

    def reset_constraints(self) -> None:
        """
        Resets the Constraints, making it an empty `dict`.
        """
        self._constraints = {}

    def add_constraint(self, name: str, func: object) -> None:
        """
        Add a constraint function fir trimming possible input values based on
        user-defined functions that receive a set of the genes as parameters and
        return a boolean value.

        Parameters
        ----------
        :name:  Constraint name.
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
        ...         'domain': int,
        ...         'range': [0, 10]},
        ...     'y': {
        ...         'type': 'numeric',
        ...         'domain': int,
        ...         'range': [0, 10]}
        ... }
        >>> blueprint = Blueprint(blueprint_dict)
        >>> def constraint(x, y): 2*x >= y
        >>> blueprint.add_constraint(name='2x_ge_y', func=constraint)

        * note how the function :constraint(): has parameters that matches some
          (or all) of the specified genes.
        """
        self._constraints.update({name: func})
