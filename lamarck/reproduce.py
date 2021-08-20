from __future__ import annotations

import numpy as np
import pandas as pd
from lamarck import Blueprint
from lamarck.gene import GeneCollection
from lamarck.utils import ParentOverloadException


def select_parents_by_tournament(ranked_pop: pd.DataFrame,
                                 n_dispute: int,
                                 n_parents: int = 2,
                                 rank_column: str = 'Rank',
                                 seed: int | None = None) -> pd.DataFrame:
    """
    Select :n_parents: Parents by randomly selecting groups of :n_dispute: creatures and
    picking the highest ranked of them.

    Parameters
    ----------
    :ranked_pop:    A Population with a Rank column to determine the fittest creatures.
    :n_dispute:     Number of Creatures disputing for one of the Parent slots (default: 2).
    :n_parents:     Number of generated parents (default: 2).
    :rank_column:   Name of the ranking column (default: 'Rank').
    :seed:          Random Number Generator control (default: None).
    """
    # check parameters validity
    minumum_population_required = n_parents + n_dispute - 1
    total_population = len(ranked_pop)
    if minumum_population_required > total_population:
        raise ParentOverloadException(total_population, minumum_population_required)

    def select_parent(pop, n):
        popsize = len(pop)
        idx = np.random.choice(popsize, n, replace=False)
        candidates = pop.iloc[idx]
        return candidates.sort_values(rank_column).iloc[0]

    np.random.seed(seed)
    parent_index = []
    for _ in range(n_parents):
        valid_index = set(ranked_pop.index).difference(parent_index)
        parent = select_parent(ranked_pop.loc[valid_index], n_dispute)
        parent_index.append(parent.name)
    return ranked_pop.loc[parent_index]


class ChildGenerator:
    """
    Generate children sexually or asexually.
    """
    @staticmethod
    def sexual(parents: pd.DataFrame,
               gene_collection: GeneCollection,
               n_children: int = 2,
               seed: int | None = None) -> pd.DataFrame:
        """
        Sexually generates offspring based on a the :parents: DataFrame.

        Parameters
        ----------
        :parents:           A Population with a Rank column to determine the fittest creatures.
        :gene_collection:   The set of `Genes` that will be crossed and mix for generating the
                            child.
        :n_children:        Number of children generated by each parent relation (default: 2).
        """
        children_data = {}
        for gene in gene_collection:
            parents_genes = tuple(parents[gene.name].tolist())
            children_genes = gene.reproduce.sexual(parents_genes, n_children, seed)
            children_data.update({gene.name: children_genes})
        offspring = pd.DataFrame(children_data).drop_duplicates()
        offspring_size = len(offspring)
        if offspring_size < n_children:
            offspring = ChildGenerator.sexual(parents,
                                              gene_collection,
                                              n_children,
                                              seed)
        return offspring

    @staticmethod
    def asexual():
        """
        """


child_generator = ChildGenerator()


class Populator:
    """
    Class for generating new Creaturs through Sexual and Asexual (exclusive Mutations)
    reproductions. The mutations may also occur on some of the newly generated offspring
    by random chance.

    Sexual Reproduction
        - Mix genes from multiple parents (2 or more), generating children (2 or more) with the
          new mixed genomes.

    Asexual Reproduction
        - Generate offspring by just forcing mutations on some genes from a single parent
          creature.
    """
    blueprint: Blueprint

    def __init__(self, blueprint):
        self.blueprint = blueprint

    def sexual(self,
               ranked_pop: pd.DataFrame,
               n_offspring: int,
               n_dispute: int = 2,
               n_parents: int = 2,
               children_per_relation: int = 2,
               rank_column: str = 'Rank',
               seed: int | None = None) -> pd.DataFrame:
        """
        Sexually generates :n_offspring: children based on a ranked DataFrame.

        Parameters
        ----------
        :ranked_pop:            A Population with a Rank column to determine the fittest
                                creatures.
        :n_offspring:           Total number of children generated by this routine. This
                                number might be just an approximation since there can be
                                limitations to the total amount of possible creatures
                                generated by this method (the output population doesn't have
                                duplicates).
        :n_dispute:             Number of Creatures disputing for one of the Parent slots
                                (default: 2).
        :n_parents:             Number of parents selected by each 'tournament' (default: 2).
        :children_per_relation: Number of children generated by each parent relation
                                (default: 2).
        :rank_column:           Name of the ranking column (default: 'Rank').
        :seed:                  Random Number Generator control (default: None).
        """
        np.random.seed(seed)
        n_relations = np.ceil(n_offspring / children_per_relation).astype(int)
        offspring = []
        for _ in range(n_relations):
            parents = select_parents_by_tournament(
                ranked_pop, n_dispute, n_parents, rank_column)
            kids = child_generator.sexual(parents,
                                          self.blueprint.genes,
                                          children_per_relation)
            offspring.append(kids)
        return pd.concat(offspring).reset_index(drop=True)

    def asexual(self):
        """
        """
