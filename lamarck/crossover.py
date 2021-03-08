import numpy as np


class CrossOver:
    def __init__(self, population):
        self.pop = population

    @property
    def genome_blueprint(self): return self.pop.genome_blueprint

    def tournament(self, n_dispute=2):
        """
        """
        n_pop = len(self.pop)
        parent1, parent2 = get_parents(n_pop, n_dispute)
        child1, child2 = cross_over(parent1, parent2)


def get_parents(n_pop, n_dispute):
    parent1 = min(np.random.choice(n_pop, n_dispute, replace=False))
    parent2 = parent1
    while parent2 == parent1:
        parent2 = min(np.random.choice(n_pop, n_dispute, replace=False))
    return parent1, parent2
