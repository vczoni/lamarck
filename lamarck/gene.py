from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractstaticmethod
import itertools
import math
import numpy as np

from lamarck.sexual import sexual_reproduction


class GeneReproductor(ABC):
    """
    Set of reproduction methods:
        - sexual: mix genes from "parents"
        - asexual: mutate gene
    """
    _gene: Gene

    @abstractstaticmethod
    def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
        pass

    @abstractclassmethod
    def asexual(self) -> tuple:
        pass


class Gene(ABC):
    """
    Abstract class for handling the different types of genes.

    Structure
    ---------
    Gene
    ├── get_linspace()
    ├── get_random()
    └── reproduce
        ├── sexual()
        └── asexual()

    """
    name: str
    domain: object
    reproduce: GeneReproductor

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
    class Reproductor(GeneReproductor):
        _gene: Gene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return sexual_reproduction.scalar_mix(parent_genes, n_children, seed)

        def asexual(self):
            return 0,

    name: str
    domain: type
    range: list | tuple
    reproduce = Reproductor

    def __init__(self, name: str, domain: type, range: list | tuple):
        self.name = name
        self.domain = domain
        self.range = range
        self.reproduce = self.Reproductor(self)

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
    class Reproductor(GeneReproductor):
        _gene: Gene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return sexual_reproduction.scalar_mix(parent_genes, n_children, seed)

        def asexual(self) -> tuple:
            return 0,

    name: str
    domain: list | tuple
    reproduce = Reproductor

    def __init__(self, name: str, domain: dict):
        self.name = name
        self.domain = domain
        self.reproduce = self.Reproductor(self)

    def get_linspace(self, n: int) -> np.ndarray:
        domain_array = np.array(self.domain)
        end = len(domain_array) - 1
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.choice(self.domain, n)


class BooleanGene(CategoricalGene):
    """
    Boolean Gene class.
    """
    name: str
    domain: tuple = (True, False)
    reproduce = CategoricalGene.Reproductor

    def __init__(self, name: str):
        self.name = name
        self.reproduce = self.Reproductor(self)

    def get_linspace(self, n: None = None) -> np.ndarray:
        return np.array([False, True], dtype=bool)

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.randint(0, 2, n, dtype=bool)


class VectorialGene(Gene):
    """
    Vectorial Gene class.
    """
    name: str
    domain: list | tuple
    length: int
    reproduce = GeneReproductor

    def __init__(self, name: str, domain: list | tuple, length: int):
        self.name = name
        self.domain = domain
        self.length = length

    def get_linspace(self, n: int) -> np.ndarray:
        pass

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        pass

    @staticmethod
    def _convert_to_array(ls):
        arr = np.empty(len(ls), dtype=object)
        arr[:] = ls
        return arr


class ArrayGene(VectorialGene):
    """
    Array Gene class.
    """
    class Reproductor(GeneReproductor):
        _gene: Gene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return sexual_reproduction.vectorial_cross(parent_genes, n_children, seed)

        def asexual(self) -> tuple:
            return 0,

    name: str
    domain: list | tuple
    length: int
    reproduce = Reproductor

    def __init__(self, name: str, domain: list | tuple, length: int):
        super().__init__(name, domain, length)
        self.reproduce = self.Reproductor(self)

    def get_linspace(self, n: int) -> np.ndarray:
        vectors = list(itertools.product(self.domain, repeat=self.length))
        end = len(self.domain)**self.length - 1
        domain_array = self._convert_to_array(vectors)
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        vectors = [tuple(np.random.choice(self.domain, self.length, replace=True))
                   for _ in range(n)]
        return self._convert_to_array(vectors)


class SetGene(VectorialGene):
    """
    Set Gene class.
    """
    class Reproductor(GeneReproductor):
        _gene: Gene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return sexual_reproduction.vectorial_cross_unique(parent_genes, n_children, seed)

        def asexual(self) -> tuple:
            return 0,

    name: str
    domain: list | tuple
    length: int
    reproduce = Reproductor

    def __init__(self, name: str, domain: list | tuple, length: int):
        super().__init__(name, domain, length)
        self.reproduce = self.Reproductor(self)

    def get_linspace(self, n: int) -> np.ndarray:
        vectors = list(itertools.permutations(self.domain, r=self.length))
        end = math.perm(len(self.domain), self.length) - 1
        domain_array = self._convert_to_array(vectors)
        index = np.unique(np.linspace(0, end, n, dtype=int))
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        vectors = [tuple(np.random.choice(self.domain, self.length, replace=False))
                   for _ in range(n)]
        return self._convert_to_array(vectors)


genetype_dict = {
    'numeric': NumericGene,
    'categorical': CategoricalGene,
    'boolean': BooleanGene,
    'array': ArrayGene,
    'set': SetGene
}


class GeneCollection:
    """
    Gene Collection.
    """
    _dict: dict

    def __init__(self, blueprint_dict):
        self._dict = {}
        for genename, geneconfig in blueprint_dict.items():
            geneclass = genetype_dict[geneconfig['type']]
            specs = geneconfig['specs']
            gene = geneclass(genename, **specs)
            self._dict.update({genename: gene})
        self.__dict__.update(self._dict)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self._dict)[key]
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    @property
    def names(self):
        return list(self._dict)
