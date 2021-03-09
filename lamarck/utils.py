

def get_id(genome):
    genomevals = tuple(genome.values())
    return hash(genomevals)


def genome_already_exists(genome, pop):
    return get_id(genome) in pop.datasets.index


def creature_already_exists(creature, pop):
    return creature.id in pop.datasets.index
