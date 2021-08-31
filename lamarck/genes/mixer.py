from __future__ import annotations

import numpy as np


class GeneMixer:
    """
    Sexual reproduction class that combine parents's genes by "mixing" or "crossing" them.

    The difference between "mix" and "cross" here is that "mixing" just gets any gene from
    random parents and picks one of them, while "crossing" means the combination of two
    vectorial genes by selecting a cut point and slicing 2 vectors in two and swapping the
    latter segment from the first vector with the latter segment from the second vector.

    Methods
    -------
    - scalar_mix()
    - vectorial_mix()
    - vectorial_cross()
    - vectorial_cross_unique()
    """
    @staticmethod
    def scalar_mix(parent_genes: tuple,
                   n_children: int = 2,
                   seed: int | None = None) -> tuple:
        """
        Mix values of non-vectorial genes (may return exact same `tuple` as the :parents:
        `tuple` because it's completely random)

        Parameters
        ----------
        :parent_genes:  Parent genes as a tuple of values
        :n_children:    Number of generated children genes (default: 2)
        :seed:          Random Number Generator control (default: None)
        """
        np.random.seed(seed)
        return tuple(np.random.choice(parent_genes) for _ in range(n_children))

    @staticmethod
    def vectorial_mix(parent_genes: tuple(tuple),
                      n_children: int = 2,
                      seed: int | None = None) -> tuple(tuple):
        """
        Mix values of vectorial genes (may return exact same `tuple` as the :parents: `tuple`
        because it's completely random)

        Parameters
        ----------
        :parent_genes:  Parent genes as a tuple of values
        :n_children:    Number of generated children genes (default: 2)
        :seed:          Random Number Generator control (default: None)
        """
        def get_mix(vectors):
            m = np.array(vectors)
            idx0 = np.random.choice(a=m.shape[0], size=m.shape[1])
            idx1 = tuple(range(m.shape[1]))
            return tuple(m[idx0, idx1])

        np.random.seed(seed)
        return tuple(get_mix(parent_genes) for _ in range(n_children))

    @staticmethod
    def vectorial_cross(parent_genes: tuple(tuple),
                        n_children: int = 2,
                        seed: int | None = None) -> tuple(tuple):
        """
        Mix values of vectorial genes (may return exact same `tuple` as the :parents: `tuple`
        because it's completely random) by crossing values in no-replacement vectors.

        Parameters
        ----------
        :parent_genes:  Parent genes as a tuple of values
        :n_children:    Number of generated children genes (default: 2)
        :seed:          Random Number Generator control (default: None)
        """
        def get_random_cross_over(v1, v2):
            cutpoint = np.random.randint(1, len(v1))
            return v1[0:cutpoint] + v2[cutpoint:]

        def get_mix(vectors):
            idx = np.random.choice(a=len(vectors), size=2, replace=False)
            v1, v2 = [tuple(v) for v in np.array(vectors)[idx]]
            return get_random_cross_over(v1, v2)

        np.random.seed(seed)
        return tuple(get_mix(parent_genes) for _ in range(n_children))

    @staticmethod
    def vectorial_cross_unique(parent_genes: tuple(tuple),
                               n_children: int = 2,
                               seed: int | None = None) -> tuple(tuple):
        """
        Mix values of vectorial genes (may return exact same `tuple` as the :parents: `tuple`
        because it's completely random) by crossing values in no-replacement vectors.

        Parameters
        ----------
        :parent_genes:  Parent genes as a tuple of values
        :n_children:    Number of generated children genes (default: 2)
        :seed:          Random Number Generator control (default: None)
        """
        def get_random_cross_over(v1, v2):
            cutpoint = np.random.randint(1, len(v1))
            g1 = v1[0:cutpoint]
            rem = [v for v in v2 if v not in g1]
            remlen = len(v1) - cutpoint
            g2 = tuple(rem[0:remlen])
            return g1 + g2

        def get_mix(vectors):
            idx = np.random.choice(a=len(vectors), size=2, replace=False)
            v1, v2 = [tuple(v) for v in np.array(vectors)[idx]]
            return get_random_cross_over(v1, v2)

        np.random.seed(seed)
        return tuple(get_mix(parent_genes) for _ in range(n_children))


gene_mixer = GeneMixer()
