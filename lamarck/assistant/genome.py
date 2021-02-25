from copy import deepcopy


class GenomeCreator:
    """
    This class is used to assist the constructuion of the Genome Blueprint,
    which is used by the PopulationCreator to, well, create Populations.

    A Genome Blueprint is not very easy to build, so this class helps clarifying
    the confusing configurations necessary to properly instruct the
    PopulationCreator.

    Example of a Genome Blueprint

    genome_blueprint = {
        'x': {
            'type': 'numeric',
            'domain': 'int',
            'ranges': {
                'min': 0,
                'max': 10,
                'progression': 'linear',
            }
        },
        'y': {
            'type': 'categorical',
            'domain': ['A', 'B', 'C'],
        },
        'z': {
            'type': 'vectorial',
            'domain': ['i', 'j', 'k'],
            'ranges': {
                'length': 5,
                'replace': True,
            }
        },
    }
    """

    def __init__(self):
        self.reset_genome_blueprint()
        self.add_gene_specs = GeneCreator(self)

    def reset_genome_blueprint(self):
        self._genome_blueprint = {}

    def get_genome_blueprint(self):
        return deepcopy(self._genome_blueprint)


class GeneCreator:
    """
    Gene Creation Assistant
    """

    def __init__(self, genome_creator):
        self._genome_creator = genome_creator

    def numeric(self, name, min, max, progression, domain):
        """
        Generate gene specs for a `numeric` variable.

        Parameters
        ----------
        :name:          `str` that defines the Gene name
        :min:           Gene's minimum possible value
        :max:           Gene's maximum possible value
        :progression:   The way this numeric variable behaves
                        Options:
                            - 'linear'
                            - 'log'
        :domain:        The Gene's domain
                        Options:
                            - 'int'
                            - 'float'
        """
        specs = {
            'type': 'numeric',
            'domain': domain,
            'ranges': {
                'min': min,
                'max': max,
                'progression': progression,
            }
        }
        self._genome_creator._genome_blueprint.update({name: specs})

    def categorical(self, name, domain):
        """
        Generate gene specs for a `categorical` variable.

        Parameters
        ----------
        :name:          `str` that defines the Gene name
        :domain:        `list` containing the entire Gene's domain (All possible values)
                        Examples:
                            1. domain=['Alpha', 'Beta', 'Gamma']
                            2. domain=[1, 2, 4, 8]
        """
        specs = {
            'type': 'categorical',
            'domain': domain,
        }
        self._genome_creator._genome_blueprint.update({name: specs})

    def vectorial(self, name, length, replace, domain):
        """
        Generate gene specs for a `vectorial` variable.

        Parameters
        ----------
        :name:          `str` that defines the Gene name
        :length:        `int` representing the vector's length
        :replace:   `bool` indicating if the vector's values can appear multiple times (True)
                        or just once (False)
        :domain:        `list` containing the entire Gene's domain (All possible values)
                        Example:
                            1. domain=['Alpha', 'Beta', 'Gamma']
                            2. domain=[1, 2, 4, 8]
        """
        specs = {
            'type': 'vectorial',
            'domain': domain,
            'ranges': {
                'length': length,
                'replace': replace,
            }
        }
        self._genome_creator._genome_blueprint.update({name: specs})
