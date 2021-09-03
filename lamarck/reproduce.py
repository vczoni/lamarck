from __future__ import annotations

import numpy as np
import pandas as pd

from lamarck.genes import GeneCollection
from lamarck.population import Blueprint, Population
from lamarck.utils import ParentOverloadException


def select_parent(pop_data, n, rank_column):
    popsize = len(pop_data)
    idx = np.random.choice(popsize, n, replace=False)
    candidates = pop_data.iloc[idx]
    return candidates.sort_values(rank_column).iloc[0]


def select_parents_by_tournament(ranked_pop_data: pd.DataFrame,
                                 n_dispute: int,
                                 n_parents: int = 2,
                                 rank_column: str = 'Rank',
                                 seed: int | None = None) -> pd.DataFrame:
    """
    Select :n_parents: Parents by randomly selecting groups of :n_dispute: creatures and
    picking the highest ranked of them.

    Parameters
    ----------
    :ranked_pop_data:   A Population with a Rank column to determine the fittest creatures.
    :n_dispute:         Number of Creatures disputing for one of the Parent slots (default: 2).
    :n_parents:         Number of generated parents (default: 2).
    :rank_column:       Name of the ranking column (default: 'Rank').
    :seed:              Random Number Generator control (default: None).
    """
    np.random.seed(seed)

    # check parameters validity
    minumum_population_required = n_parents + n_dispute - 1
    total_population = len(ranked_pop_data)
    if minumum_population_required > total_population:
        raise ParentOverloadException(total_population, minumum_population_required)

    parent_index = []
    for _ in range(n_parents):
        valid_index = set(ranked_pop_data.index).difference(parent_index)
        parent = select_parent(ranked_pop_data.loc[valid_index], n_dispute, rank_column)
        parent_index.append(parent.name)
    return ranked_pop_data.loc[parent_index]


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
                            children.
        :n_children:        Number of children generated by each parent relation (default: 2).
        :seed:              Random Number Generator control (default: None).
        """
        np.random.seed(seed)
        children_data = {}
        for gene in gene_collection:
            parents_genes = tuple(parents[gene.name].tolist())
            children_genes = gene.reproduce.sexual(parents_genes, n_children)
            children_data.update({gene.name: children_genes})
        offspring = pd.DataFrame(children_data).drop_duplicates()
        offspring_size = len(offspring)
        if offspring_size < n_children:
            offspring = ChildGenerator.sexual(parents,
                                              gene_collection,
                                              n_children)
        return offspring

    @staticmethod
    def asexual(parent_creature: pd.Series,
                gene_collection: GeneCollection,
                n_children: int = 1,
                n_genes: int = 1,
                seed: int | None = None) -> pd.DataFrame:
        """
        Asexually generates offspring based on a the :parent_creature: Series.

        Parameters
        ----------
        :parent_creature:   A Pandas Series with the Creature data.
        :gene_collection:   The set of `Genes` that will be mutated for generating the children.
        :n_children:        Number of children generated by each parent relation (default: 2).
        :n_genes:           Number of Genes that will be randomly selected for mutation.
        :seed:              Random Number Generator control (default: None).
        """
        np.random.seed(seed)
        children_data = {gene: [val]*n_children for gene, val in parent_creature.iteritems()
                         if gene in gene_collection.names}
        for i_children in range(n_children):
            genes_to_mutate = np.random.choice(gene_collection, size=n_genes, replace=False)
            for gene in genes_to_mutate:
                new_gene = gene.reproduce.asexual(1)[0]
                children_data[gene.name][i_children] = new_gene
        offspring = pd.DataFrame(children_data).drop_duplicates()
        offspring_size = len(offspring)
        if offspring_size < n_children:
            offspring = ChildGenerator.asexual(parent_creature,
                                               gene_collection,
                                               n_children,
                                               n_genes)
        return offspring


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
               ranked_pop_data: pd.DataFrame,
               n_offspring: int,
               n_dispute: int = 2,
               n_parents: int = 2,
               children_per_relation: int = 2,
               rank_column: str = 'Rank',
               seed: int | None = None) -> Population:
        """
        Sexually generates :n_offspring: children based on a ranked DataFrame.

        Parameters
        ----------
        :ranked_pop_data:       A Population with a Rank column to determine the fittest
                                creatures.
        :n_offspring:           Total number of children generated by this routine (may generate
                                identical children).
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
        offspring_list = []
        for _ in range(n_relations):
            parents = select_parents_by_tournament(
                ranked_pop_data, n_dispute, n_parents, rank_column)
            kids = child_generator.sexual(parents,
                                          self.blueprint.genes,
                                          children_per_relation)
            offspring_list.append(kids)
        offspring_data = pd.concat(offspring_list)[self.blueprint.genes.names]\
            .reset_index(drop=True)
        return Population(offspring_data, self.blueprint.copy())

    def asexual(self,
                ranked_pop_data: pd.DataFrame,
                n_offspring: int,
                n_mutated_genes: int = 1,
                children_per_creature: int = 1,
                rank_column: str = 'Rank',
                seed: int | None = None) -> Population:
        """
        Asexually generates :n_offspring: children based on a ranked DataFrame.

        Parameters
        ----------
        :ranked_pop_data:       A Population with a Rank column to determine the fittest
                                creatures.
        :n_offspring:           Total number of children generated by this routine (may generate
                                identical children).
        :n_mutated_genes:       Number of genes that will be randomly selected and mutaded (may
                                mutate to the same value).
        :children_per_creature: Number of children generated by each creature mutation
                                (default: 1).
        :rank_column:           Name of the ranking column (default: 'Rank').
        :seed:                  Random Number Generator control (default: None).
        """
        np.random.seed(seed)
        n_mutations = np.ceil(n_offspring / children_per_creature).astype(int)
        offspring_list = []
        for _ in range(n_mutations):
            parent_creature = select_parent(ranked_pop_data, 1, rank_column)
            kids = child_generator.asexual(parent_creature,
                                           self.blueprint.genes,
                                           children_per_creature)
            offspring_list.append(kids)
        offspring_data = pd.concat(offspring_list)[self.blueprint.genes.names]\
            .reset_index(drop=True)
        return Population(offspring_data, self.blueprint.copy())
