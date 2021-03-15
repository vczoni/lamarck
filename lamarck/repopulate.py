import numpy as np
from lamarck.utils import random_linear, random_log


class Repopulator:
    def __init__(self, population):
        self.tournament = Tournament(population)
        self.elitism = Elitism(population)
        self.mutate = Mutation(population)


class Reproduce:
    def __init__(self, population):
        self._pop = population

    @property
    def _genome_blueprint(self): return self._pop.genome_blueprint

    def _get_parents(self, idx1, idx2):
        id1 = self._pop.datasets.fitness.index[idx1]
        id2 = self._pop.datasets.fitness.index[idx2]
        parent1 = self._pop.get_creature.from_id(id1)
        parent2 = self._pop.get_creature.from_id(id2)
        return parent1, parent2

    def _populate(self, genomes):
        self._pop.populate.from_genome_list(list(genomes))

    def _cross_over(self, pgenome1, pgenome2):
        flag = True
        while flag:
            schild1 = {}
            schild2 = {}
            for gene, spec in self._genome_blueprint.items():
                pg1 = pgenome1[gene]
                pg2 = pgenome2[gene]
                if spec['type'] == 'vectorial':
                    if spec['ranges']['replace']:
                        g1, g2 = cross_vectorial_genes(pg1, pg2)
                    else:
                        g1, g2 = cross_unique_vectorial_genes(pg1, pg2)
                else:
                    g1, g2 = cross_genes(pg1, pg2)
                schild1.update({gene: g1})
                schild2.update({gene: g2})
            flag = schild1 == schild2
        return schild1, schild2


def cross_genes(pg1, pg2):
    ls = [pg1, pg2]
    i = np.random.randint(0, 2)
    g1 = ls[i]  # pylint: disable=invalid-sequence-index
    g2 = ls[1 - i]
    return g1, g2


def cross_vectorial_genes(pg1, pg2):
    idx_piece = np.random.choice(a=[True, False], size=len(pg1))
    g1 = tuple(np.array(pg1)[idx_piece])
    g2 = tuple(np.array(pg2)[~idx_piece])
    return g1, g2


def cross_unique_vectorial_genes(pg1, pg2):
    cutpoint = np.random.randint(1, len(pg1))
    g1 = pg1[0:cutpoint]
    g2 = pg2[0:cutpoint]
    rem1 = [v for v in pg2 if v not in g1]
    rem2 = [v for v in pg1 if v not in g2]
    remlen = len(pg1) - cutpoint
    g1 += tuple(rem1[0:remlen])
    g2 += tuple(rem2[0:remlen])
    return g1, g2


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
        n = n_final_pop - len(self._pop)
        genome_list = []
        for _ in range(int(n/2)+1):
            idx1, idx2 = get_parents_min_index(n_potential_parents,
                                               n_dispute)
            parent1, parent2 = self._get_parents(idx1, idx2)
            child_genomes = self._cross_over(parent1.genome,
                                             parent2.genome)
            genome_list += list(child_genomes)
        self._populate(genome_list)


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
                        Repopulator
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
        n = n_final_pop - len(self._pop)
        genome_list = []
        for _ in range(int(n/2)+1):
            idx1, idx2 = next(parent_generator)
            parent1, parent2 = self._get_parents(idx1, idx2)
            child_genomes = self._cross_over(parent1.genome,
                                             parent2.genome)
            genome_list += list(child_genomes)
        self._populate(genome_list)


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


class Mutation(Reproduce):
    def __call__(self, p, n_genes=1):
        """
        Generates new creatures by randomly picking Creatures based on the
        :mutation probability (p): and randomly changing :n_genes: gene(s).

        Parameters
        ----------
        :p:         `float` - Mutation probability.
        :n_genes:   `int` - Number of genes tha will be altered (default: 1).
        """
        n = len(self._pop)
        f_creatures = np.random.rand(n) < p
        if any(f_creatures):
            to_be_mutated = self._pop.datasets.input[f_creatures]
            genome_list = [mutate(x[1].to_dict(),
                                  n_genes,
                                  self._genome_blueprint)
                           for x in to_be_mutated.iterrows()]
            self._populate(genome_list)


def mutate(genome, n, blueprint):
    genes = list(genome.keys())
    idxs = np.random.choice(range(len(genes)), replace=False, size=n)
    for idx in idxs:
        gene = genes[idx]
        val = genome[gene]
        specs = blueprint[gene]
        if specs['type'] == 'numeric':
            new_val = mutate_numeric(val, specs)
        elif specs['type'] == 'categorical':
            new_val = mutate_categorical(val, specs)
        elif specs['type'] == 'vectorial':
            new_val = mutate_vectorial(val, specs)
        genome.update({gene: new_val})
    return genome


def mutate_numeric(val, specs):
    minval = specs['ranges']['min']
    maxval = specs['ranges']['max']
    flag = True
    while flag:
        if specs['ranges']['progression'] == 'linear':
            new_val = random_linear(minval, maxval)
        if specs['ranges']['progression'] == 'log':
            new_val = random_log(minval, maxval)
        if specs['domain'] == 'int':
            new_val = int(new_val)
        flag = new_val == val
    return new_val


def mutate_categorical(val, specs):
    domain = specs['domain']
    flag = True
    while flag:
        new_val = np.random.choice(domain)
        flag = new_val == val
    return new_val


def mutate_vectorial(val, specs):
    domain = specs['domain']
    length = specs['ranges']['length']
    replace = specs['ranges']['replace']
    flag = True
    while flag:
        new_val = np.random.choice(domain, length, replace=replace)
        flag = new_val == val
    return new_val


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
