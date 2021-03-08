

def get_id(genome):
    genomevals = tuple(genome.values())
    return hash(genomevals)
