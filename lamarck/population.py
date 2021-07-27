import pandas as pd


class PopulationDatasets:
    """
    Set of Datasets describing the Population

    Datasets
    --------
    1. Creature Dataset:    Data of the set of genes for each Creature in the
                            Population.
    4. History Dataset:     Historic Fitness data from previous generations.
    """
    creatures: pd.DataFrame
    history: pd.DataFrame


class Population:
    """
    Population class for the Genetic Algorithm Simulations.

    Stores all the genetic data from all Creatures in the Population.
    """
    datasets: PopulationDatasets
