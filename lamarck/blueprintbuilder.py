from __future__ import annotations

from copy import deepcopy
from lamarck.utils import VectorialOverloadException


class GenomeBlueprintBuilder:
    """
    Genome Blueprint Builder Class.
    
    Genome Specifications
    ---------------------
        1. Numeric
            1.1. Domain: `str {'int', 'float'}`
            1.2. Ranges
                - min: number
                - max: number
        
        2. Categorical
            2.1. Domain: `list` or `tuple`

        3. Vectorial
            3.1. Domain: `list` or `tuple`
            3.2. Ranges
                - length: int
                - replace: bool {True, False}
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
        return deepcopy(self._blueprint)
    
    def add_numeric_gene(self, name: str, domain: object, range: list | tuple
        ) -> None:
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
        >>> builder.get_blueprint()
        {'x': {'type': 'numeric', 'domain': <class 'int'>, 'range': [0, 10]}}
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
        >>> builder.get_blueprint()
        {'letters': {'type': 'categorical', 'domain': ['a', 'b', 'c']}}
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
        >>> builder.get_blueprint()
        {'sequence': {'type': 'vectorial', 'domain': ['A', 'T', 'C', 'G'], 'replacement': True, 'length': 1000}}
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
        >>> builder.get_blueprint()
        {'match_flag': {'type': 'boolean'}, 'is_random': {'type': 'boolean'}}
        """
        genespecs = {
            'type': 'boolean'
        }
        self._blueprint.update({name: genespecs})


def check_vector_validity(domain, length):
    domain_lenght = len(domain)
    if domain_lenght < length:
        raise VectorialOverloadException(length, domain_lenght)
