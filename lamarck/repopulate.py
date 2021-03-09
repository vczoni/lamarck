import numpy as np


class Repopulator:
    def __init__(self, population):
        self.tournament = Tournament(population)
        self.elitism = Elitism(population)


class Reproduce:
    def __init__(self, population):
        self._pop = population

    @property
    def _genome_blueprint(self): return self._pop.genome_blueprint

    def _populate(self, parent1, parent2):
        child_genomes = self._get_offspring(parent1, parent2)
        self._pop.populate.from_genome_list(list(child_genomes))

    def _get_offspring(self, parent1, parent2):
        sgenome1 = construct_straight_genome(parent1.genome)
        sgenome2 = construct_straight_genome(parent2.genome)
        gb = self._genome_blueprint
        schild1, schild2 = self._cross_over(sgenome1, sgenome2)
        child_genome1 = deconstruct_straight_genome(schild1, gb)
        child_genome2 = deconstruct_straight_genome(schild2, gb)
        return child_genome1, child_genome2

    def _cross_over(self, sgenome1, sgenome2):
        if all(sgenome1 == sgenome2):
            raise Exception(f'Genomes must never match (genome: {sgenome1}).')
        flag = True
        while flag:
            idx = self._random_index()
            schild1 = sgenome1.copy()
            schild2 = sgenome2.copy()
            schild1[idx] = sgenome2[idx]
            schild2[idx] = sgenome1[idx]
            flag = all(schild1 == schild2) or all(idx) or not any(idx)
        return schild1, schild2

    def _random_index(self):
        idx = np.zeros(0, dtype=bool)
        for spec in self._genome_blueprint.values():
            if spec['type'] == 'vectorial':
                n = spec['ranges']['length']
                if spec['ranges']['replace']:
                    idx_piece = np.random.choice(a=[True, False], size=n)
                else:
                    val = np.random.randint(0, 2, dtype=bool)
                    idx_piece = np.tile(val, n)
            else:
                idx_piece = np.random.choice(a=[True, False], size=1)
            idx = np.concatenate((idx, idx_piece))
        return idx

    def mutate(self, prob):
        """
        Parameters
        ----------
        :prob:  `float`
        """


class Tournament(Reproduce):
    def __call__(self, n_dispute=2, n_children=None):
        """
        Generate offspring by Tournament selection.

        The `Tournament` selection consists of rounding up :n_dispute: Creatures
        and selecting the fittest of the bunch as one of the Parents, and then
        doing the same process for the second parent.

        Parameters
        ----------
        :n_dispute:     Number of Creatures to dispute each of the two `parent`
                        positions. A bigger number means a less random and more
                        elitist parenting selection (default: 2).
        :n_children:    The total amount of children generated. If None is set,
                        the algorithm will keep generating children until the
                        "natural population level" is reached (default: None).
        """
        n_potential_parents = len(self._pop)
        n_final_pop = get_n_final_pop(self._pop, n_children)
        while len(self._pop) < n_final_pop:
            parent1, parent2 = self\
                ._get_parents(n_potential_parents, n_dispute)
            self._populate(parent1, parent2)

    def _get_parents(self, n_pop, n_dispute):
        idx1, idx2 = get_parents_min_index(n_pop, n_dispute)
        parent1 = self._pop.get_creature.from_index(idx1)
        parent2 = self._pop.get_creature.from_index(idx2)
        return parent1, parent2


def get_parents_min_index(n_pop, n_dispute):
    idx1 = min(np.random.choice(n_pop, n_dispute, replace=False))
    idx2 = idx1
    while idx2 == idx1:
        idx2 = min(np.random.choice(n_pop, n_dispute, replace=False))
    return idx1, idx2


class Elitism(Reproduce):
    def __call__(self, n_children=None):
        """
        Generate offspring by Elitism selection.

        The `Elitism` selection consists of selecting the top Creatures as the
        most recurring parents, in "best-status-combination" scenarios.

        Parameters
        ----------
        :n_children:    The total amount of children generated. If None is set,
                        the algorithm will keep generating children until the
                        "natural population level" is reached (default: None).

        Idea
        ----
        couple 01:  creature01 + creature02     ("sum" = 03, "top" = 01)
        couple 02:  creature01 + creature03     ("sum" = 04, "top" = 01)
        couple 03:  creature01 + creature04     ("sum" = 05, "top" = 01)
        couple 04:  creature02 + creature03     ("sum" = 05, "top" = 02)
        couple 05:  creature01 + creature05     ("sum" = 06, "top" = 01)
        couple 06:  creature02 + creature04     ("sum" = 06, "top" = 02)
        couple 07:  creature01 + creature06     ("sum" = 07, "top" = 01)
        couple 08:  creature02 + creature05     ("sum" = 07, "top" = 02)
        couple 09:  creature03 + creature04     ("sum" = 07, "top" = 03)
        couple 10:  creature01 + creature07     ("sum" = 08, "top" = 01)
        couple 11:  creature02 + creature06     ("sum" = 08, "top" = 02)
        couple 12:  creature03 + creature05     ("sum" = 08, "top" = 03)

        Recurrence (taking this example):
            - creature01:     6 times
            - creature02:     5 times
            - creature03:     4 times
            - creature04:     3 times
            - creature05:     3 times
            - creature06:     2 times
            - creature07:     1 times
        """
        n_final_pop = get_n_final_pop(self._pop, n_children)
        parent_generator = generate_elite_parent_pair()
        while len(self._pop) < n_final_pop:
            idx1, idx2 = next(parent_generator)
            parent1, parent2 = self._get_parents(idx1, idx2)
            self._populate(parent1, parent2)

    def _get_parents(self, idx1, idx2):
        parent1 = self._pop.get_creature.from_index(idx1)
        parent2 = self._pop.get_creature.from_index(idx2)
        return parent1, parent2


# I know this code (the next two functions) is filthy but this turned out to be
# a fairly hard problem and I had to keep going...
# I might embellish this later but fwiw this horrible thing is actually working =)
# (Also thank you StackOverflow for the second function =D)

def generate_elite_parent_pair():
    s = 1
    ls = [0, 1]
    while True:
        subsum = subset_sum(ls, s)
        while True:
            try:
                crs = next(subsum)
                if len(crs) == 2:
                    yield crs
            except:
                s += 1
                ls.append(ls[-1] + 1)
                break
    return


def subset_sum(numbers, target, partial=[], partial_sum=0):
    if partial_sum == target:
        yield tuple(partial)
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1:]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)


def get_n_final_pop(pop, n_children):
    n_potential_parents = len(pop)
    if n_children is None:
        n_final_pop = pop.natural_population_level
    else:
        n_final_pop = n_potential_parents + n_children
    return n_final_pop


def construct_straight_genome(genome):
    sgenome = []
    for geneval in genome.values():
        if isinstance(geneval, tuple):
            sgenome += list(geneval)
        else:
            sgenome.append(geneval)
    return np.array(sgenome, dtype=object)


def deconstruct_straight_genome(sgenome, genome_blueprint):
    genome = {}
    i = 0
    for gene, spec in genome_blueprint.items():
        if spec['type'] == 'vectorial':
            n = spec['ranges']['length']
            geneval = tuple(sgenome[i:i+n])
        else:
            n = 1
            geneval = sgenome[i]
        i += n
        genome.update({gene: geneval})
    return genome
