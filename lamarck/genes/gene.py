from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractstaticmethod
import itertools
import math
import numpy as np

from lamarck.genes.mixer import gene_mixer


class Gene:
    """
    Base Class for the Gene.
    """

    class GeneReproductor(ABC):
        """
        Base class for the 'reproduce' structure in the Gene Base Class.
        """
        _gene: Gene

        def asexual(self, n: int, seed: int | None = None) -> tuple:
            return tuple(self._gene.get_random(n, seed))

        @abstractstaticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            pass

    name: str
    domain: tuple
    reproduce: GeneReproductor

    @abstractclassmethod
    def get_linspace(self, n: int) -> np.ndarray:
        pass

    @abstractclassmethod
    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        pass


class ScalarGene(Gene):
    """
    Base class for the Scalar Genes:

    Scalar Genes
    ------------
    - IntegerGene
    - FloatGene
    - CategoricalGene
    - BooleanGene
    """
    class GeneReproductor(Gene.GeneReproductor):
        _gene: ScalarGene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return gene_mixer.scalar_mix(parent_genes, n_children, seed)

    name: str
    domain: tuple
    reproduce: GeneReproductor

    def __init__(self, name: str, domain: tuple):
        self.name = name
        self.domain = domain
        self.reproduce = self.GeneReproductor(self)


class IntegerGene(ScalarGene):
    """
    Integer Gene Type.

    Attributes
    ----------
    :name:      `str` Gene name.
    :domain:    `tuple` with min and max values.

    Example
    -------
    int_gene = IntegerGene(name='int_gene', domain=(1, 10))
    """
    def get_linspace(self, n: int) -> np.ndarray:
        start, end = sorted(self.domain)
        return np.linspace(start, end, n, dtype=int)

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        start, end = sorted(self.domain)
        return np.random.randint(start, end+1, n)


class FloatGene(ScalarGene):
    """
    Float Gene Type.

    Attributes
    ----------
    :name:      `str` Gene name.
    :domain:    `tuple` with min and max values.

    Example
    -------
    float_gene = FloatGene(name='float_gene', domain=(1., 15.5))
    """
    def get_linspace(self, n: int) -> np.ndarray:
        start, end = sorted(self.domain)
        return np.linspace(start, end, n, dtype=float)

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        start, end = sorted(self.domain)
        return np.random.uniform(start, end, n)


class CategoricalGene(ScalarGene):
    """
    Categorical Gene Type.

    Attributes
    ----------
    :name:      `str` Gene name.
    :domain:    `tuple` with all categories. The "categories" can consist of just any kind of
                Python object.

    Example
    -------
    cat_gene = CategoricalGene(name='cat_gene', domain=('A', 'B', 'C'))
    """
    def get_linspace(self, n: int) -> np.ndarray:
        domain_array = np.array(self.domain)
        end = len(domain_array) - 1
        index = np.linspace(0, end+1, n+1, dtype=int)[:-1]
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.choice(self.domain, n)


class BooleanGene(ScalarGene):
    """
    Boolean Gene Type.

    Attributes
    ----------
    :name:      `str` Gene name.
    :domain:    `tuple` (False, True).

    Example
    -------
    bool_gene = BooleanGene(name='bool_gene')
    """
    def __init__(self, name: int):
        super().__init__(name=name, domain=(False, True))

    def get_linspace(self, n: int) -> np.ndarray:
        domain_array = np.array(self.domain)
        index = np.linspace(0, 2, n+1, dtype=int)[:-1]
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.randint(0, 2, n, dtype=bool)


class VectorialGene(Gene):
    """
    Base class for the Vectorial Genes:

    Vectorial Genes
    ---------------
    - ArrayGene
    - SetGene
    """
    class GeneReproductor(Gene.GeneReproductor):
        pass

    name: str
    domain: tuple
    length: int
    reproduce: GeneReproductor

    def __init__(self, name: str, domain: tuple, length: int):
        self.name = name
        self.domain = domain
        self.length = length
        self.reproduce = self.GeneReproductor(self)

    @staticmethod
    def _convert_to_array(ls):
        arr = np.empty(len(ls), dtype=object)
        arr[:] = ls
        return arr


class ArrayGene(VectorialGene):
    """
    Array Gene Type.

    Attributes
    ----------
    name:       `str` Gene name.
    :domain:    `tuple` with all categories. The "categories" can consist of just any kind of
                Python object.
    :length:    `int` value of the array length.

    Example
    -------
    array_gene = ArrayGene(name='array_gene', domain=('i', 'j', 'k'), length=10)
    """
    class GeneReproductor(Gene.GeneReproductor):
        _gene: ArrayGene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return gene_mixer.vectorial_cross(parent_genes, n_children, seed)

    def get_linspace(self, n: int) -> np.ndarray:
        vectors = list(itertools.product(self.domain, repeat=self.length))
        end = len(self.domain)**self.length - 1
        domain_array = self._convert_to_array(vectors)
        index = np.linspace(0, end, n, dtype=int)
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        a = np.array(self.domain, dtype=object)
        vectors = [tuple(np.random.choice(a, self.length, replace=True))
                   for _ in range(n)]
        return self._convert_to_array(vectors)


class SetGene(VectorialGene):
    """
    Set Gene Type.

    Attributes
    ----------
    name:       `str` Gene name.
    :domain:    `tuple` with all categories. The "categories" can consist of just any kind of
                Python object.
    :length:    `int` value of the array length. The SetGene's :length: cannot be greater than
                the length of its :domain: since the vectors are made WITHOUT replacement. It
                IS possible to create domains with repeated values though (e.g. (1, 2, 2, 3),
                ('a', 'a', 'b'), etc).

    Example
    -------
    set_gene = ArrayGene(name='set_gene', domain=('i', 'j', 'k'), length=3)
    """
    class GeneReproductor(Gene.GeneReproductor):
        _gene: ArrayGene

        def __init__(self, gene):
            self._gene = gene

        @staticmethod
        def sexual(parent_genes: tuple, n_children: int = 2, seed: int | None = None) -> tuple:
            return gene_mixer.vectorial_cross_unique(parent_genes, n_children, seed)

    def get_linspace(self, n: int) -> np.ndarray:
        vectors = list(itertools.permutations(self.domain, r=self.length))
        end = math.perm(len(self.domain), self.length) - 1
        domain_array = self._convert_to_array(vectors)
        index = np.linspace(0, end, n, dtype=int)
        return domain_array[index]

    def get_random(self, n: int, seed: int | None = None) -> np.ndarray:
        np.random.seed(seed)
        a = np.array(self.domain, dtype=object)
        vectors = [tuple(np.random.choice(a, self.length, replace=False))
                   for _ in range(n)]
        return self._convert_to_array(vectors)


genetype_dict = {
    'integer': IntegerGene,
    'float': FloatGene,
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
