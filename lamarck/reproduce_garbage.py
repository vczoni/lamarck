from __future__ import annotations
from typing import Callable
import itertools
import numpy as np
import pandas as pd

from lamarck import Blueprint
from lamarck.gene import VectorialGene
from lamarck.utils import hash_cols


def select_fittest(
        ranked_data: pd.DataFrame,
        p: float = 0.5,
        rank_col: str = 'Rank') -> pd.DataFrame:
    """
    Select the fraction :p: of creatures from a ranked population data based on
    a Rank column.
    """
    n_selection = int(round(len(ranked_data) * p))
    return ranked_data.sort_values(rank_col)[0:n_selection]


class Reproductor:
    """
    Apply Reproduction techniques to generate new Creatures based on a Blueprint object
    and the fittest Creatures's genomes. Also applies random mutations.

    Population Methods
    ------------------
    - tournament(ranked_population, total_offspring, n_dispute=2, n_offspring=2)
    - elitism(ranked_population, total_offspring)

    Creature Methods
    ----------------
    - cross_over(parents, n_offspring=2)
    - mutation(p, max_genes=None)
    """

    class CrossOver:
        """
        Cross Over assistant. Helps tracking the gene types to properly execute cross-over
        methods.
        """
        blueprint: Blueprint
        get_cromosome: Callable[[list[str]], list]

        def __init__(self, blueprint: Blueprint):
            self._blueprint = blueprint

        def set_chromosome_getter(self, parents: pd.DataFrame) -> None:
            """
            Define the `get chromosome` function.
            """
            all_genes = parents.columns
            genes = [gene for gene in all_genes if gene in self._blueprint.genes.names]

            # Setting a limit for performance reasons
            too_many_chromosomes = len(genes) > 16
            if too_many_chromosomes:
                def get_chromosome(seed: int | None = None) -> pd.Series:
                    np.random.seed(seed)
                    chrom = np.random.choice(a=[0, 1], size=len(genes)).tolist()
                    return self.build_non_vectorial_gene_vector(df=parents, seq=chrom)
            else:
                non_vec_genes = [
                    gene for gene in genes
                    if not isinstance(self._blueprint.genes[gene], VectorialGene)]
                cromspace = self._get_chromosome_space(non_vec_genes)
                cromspace_size = len(cromspace)

                def get_chromosome(seed: int | None = None) -> pd.Series:
                    row = np.random.randint(cromspace_size)
                    chrom = cromspace.iloc[row].tolist()
                    return self.build_non_vectorial_gene_vector(df=parents[non_vec_genes],
                                                                seq=chrom)

            self.get_cromosome = get_chromosome

        def _get_chromosome_space(self, genes: list[str]) -> pd.DataFrame:
            """
            Define the space of all possible gene combinations.

            Parameters
            ----------
            :genes: Names of the genes that will be crossed. If `None`, all blueprint genes
                    will be considered (default: None).
            """
            # Gathering non-vectorial genes
            parents = [0, 1]
            n_genes = len(genes)
            cromspace = pd.DataFrame(list(itertools.product(*[parents]*n_genes)))
            f = cromspace.nunique(axis=1) == 2
            return cromspace[f]

        @staticmethod
        def build_non_vectorial_gene_vector(df: pd.DataFrame, seq: list) -> pd.Series:
            dt_lst = [df.iloc[r, i] for i, r in enumerate(seq)]
            return pd.Series(dt_lst, index=df.T.index)

        @staticmethod
        def combine_vectors_with_replacement(v1: tuple, v2: tuple) -> tuple:
            idx_piece = np.random.choice(a=[True, False], size=len(v1))
            return tuple(np.array(v1)[idx_piece]) + tuple(np.array(v2)[~idx_piece])

        @staticmethod
        def combine_vectors_without_replacement(v1: tuple, v2: tuple) -> tuple:
            cutpoint = np.random.randint(1, len(v1))
            g1 = v1[0:cutpoint]
            rem = [v for v in v2 if v not in g1]
            remlen = len(v1) - cutpoint
            return g1 + tuple(rem[0:remlen])

    blueprint: Blueprint
    _cross_over_assist = CrossOver

    def __init__(self, blueprint: Blueprint):
        self.blueprint = blueprint
        self._cross_over_assist = self.CrossOver(blueprint)

    def cross_over(
            self,
            parents: pd.DataFrame,
            seed: int | None = None) -> pd.DataFrame:
        """
        Apply the cross-over routine based on the :parents: data.
        """
        self._cross_over_assist.set_chromosome_getter(parents)

    def tournament(
            self,
            ranked_population: pd.DataFrame,
            total_offspring: int,
            n_dispute: int = 2,
            n_offspring: int = 2,
            rank_column: str = 'Rank',
            hash_index: bool = True,
            seed: int | None = None) -> pd.DataFrame:
        """
        `Tournament` Reproduction method: Generates new offspring by randomly picking
        :n_dispute: candidates and selecting the fittest as one of the :n_parents: parents.

        Parameters
        ----------
        :ranked_population: A Population with a Rank column to determine the fittest creatures.
        :total_offspring:   Total number of children generated by this routine. This number
                            might be just an approximation since there can be limitations to
                            the total amount of possible creatures generated by this method
                            (the output population doesn't have duplicates).
        :n_dispute:         Number of Creatures disputing for one of the Parent slots
                            (default: 2).
        :n_offspring:       Number of children generated by each 'tournament' (default: 2).
        :rank_column:       Name of the ranking column (default: 'Rank').
        :hash_index:        Flag that determines if the `lamarck tuple_hash` method will be
                            applyed to the offspring.
        :seed:              Random Number Generator control (default: None)
        """
        np.random.seed(seed)

        def select_candidates(pop, n):
            indexes = np.random.choice(pop.index, n, replace=False)
            return pop.loc[indexes]

        def select_parent(pop, n_dispute):
            candidates = select_candidates(pop, n_dispute)
            return candidates.sort_values(rank_column).iloc[0]

        def select_parents(pop, n_dispute):
            parent_indexes = []
            for _ in range(2):
                valid_indexes = set(pop.index).difference(parent_indexes)
                f = pop.index.isin(valid_indexes)
                valid_pop = pop[f]
                parent = select_parent(valid_pop, n_dispute)
                parent_indexes.append(parent.name)
            f = pop.index.isin(parent_indexes)
            return pop[f]

        n_loops = int(total_offspring / n_offspring) + 1
        offspring_list = []
        for _ in range(n_loops):
            parents = select_parents(ranked_population, n_dispute)
            offspring = self.cross_over(parents, n_offspring)
            if hash_index:
                offspring.index = hash_cols(offspring)
            offspring_list.append(offspring)
        return pd.concat(offspring_list).drop_duplicates()
