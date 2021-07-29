from __future__ import annotations

from abc import ABC, abstractclassmethod
import itertools
import math
import numpy as np


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
