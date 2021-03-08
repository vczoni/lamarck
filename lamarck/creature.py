from lamarck.utils import get_id


class Creature:
    def __init__(self, genome=None):
        self.id = None
        self.genome = None
        self.set_genome(genome)

    def __repr__(self):
        return f'Creature <{self.id}> - genome: {self.genome}'

    def set_genome(self, genome):
        self.genome = genome
        self._genomevals = tuple(genome.values())
        self.id = get_id(self.genome)
