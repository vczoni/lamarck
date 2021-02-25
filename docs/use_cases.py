from lamarck import Creature, Environment
from lamarck.assistant import GenomeCreator, PopulationCreator


# Creature

genome = {
    'x': 5,
    'y': 'B',
    'z': 'k'
}
creature = Creature()
creature.set_genome(genome)

>>> creature.id  # the creature ID is a hash value generated from their specific genome
# 6271

>>> creature.genome
# {'x': 5, 'y': 'B', 'z': 'k'}

>>> creature._genomevals
# [5, 'B', 'k']


# Genome creation

# Method 1 - Straightforward but Complex
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

# Method 2 - Assisted by the GenomeCreator
genome_creator = GenomeCreator()
genome_creator.add_gene_specs.numeric(name='x', min=0, max=10, progression='linear', domain='int')
genome_creator.add_gene_specs.categorical(name='y', domain=['A', 'B', 'C'])
genome_creator.add_gene_specs.vectorial(name='z', length=5, replace=True, domain=['i', 'j', 'k'])
genome_blueprint = genome_creator.get_genome_blueprint()


# Population creation

pop_creator = PopulationCreator()
pop_creator.set_genome_blueprint(genome_blueprint)

# create initial population
# deterministic
# creates n values per gene, deterministically distributed according to their type
pop = pop_creator.create.det(n=4)
# random
# creates n values per gene, randomly distributed according to their type
pop = pop_creator.create.rand(n=4)
# mixture
pop1 = pop_creator.create.det(n=4)
pop2 = pop_creator.create.rand(n=4)
pop = pop1 + pop2

# Population data
>>> pop.datasets.input
# +----+---+--+--+
# |  id|  x| y| z|
# +----+---+--+--+
# |3315| 10| A| i|
# |7681|  8| A| j|
# |  12|  1| C| k|
# |-331|  4| B| k|
# +----+---+--+--+

>>> pop.generation
# 1


# Environment - its the class that stores the process that evaluates populations

env = Environment()


# The Process

# the dummy process must be a function that has the genes as parameters and returns a dict with the output(s)
# that is(are) set to be optimized
def dummy_process(x, y, z):
    distance = x * ord(y)
    if z == 'i':
        distance += x / 10
        time = distance - x
    elif z == 'j':
        distance -= (x + 12)
        time = ord(y.lower())
    elif z == 'k':
        distance = x ** 1.5 - distance
        time = distance / x
    return {
        'distance': abs(distance),
        'time': abs(time)
    }


env.set_process(dummy_process)
env.set_output_varibles('time', 'distance')


# Simulating

# Creature Fitness
# returns a dict ({'distance': 318.82, 'time': 63.76})
creature_fitness = env.simulate_creature(creature)
# Population Fitness
# returns a Population object with a Fitness dataset
pop_out = env.simulate(pop)


# Fitness

>>> pop_out.generation
# 1

# Fitness data
>>> pop_out.datasets.output
# +----+----+---+--+--+---------+-----+
# | gen|  id|  x| y| z| distance| time|
# +----+----+---+--+--+---------+-----+
# |   1|3315| 10| A| i|      651|  641|
# |   1|7681|  8| A| j|      500|   97|
# |   1|  12|  1| C| k|       66|   66|
# |   1|-331|  4| B| k|      256|   64|
# +----+----+---+--+--+---------+-----+


# Fitness Criteria - Natural Selection
# Single objective
single_fitness = pop_out.apply_fitness.single_objective(output='distance', objective='min')  # optimizes only for (min) distance
# Multi objective
# ranked
ranked_fitness = pop_out.apply_fitness.multi_objective.ranked(priority=['distance', 'time'], objective=['min', 'max'])  # optimizes for (min) distance, then for (max) time
# pareto
pareto_fitness = pop_out.apply_fitness.multi_objective.pareto(outputs=['distance', 'time'], objective=['min', 'max'])  # optimizes for (min) distance and for (max) time


# Selection

# Selection data
>>> single_fitness.datasets.fitness
# +----+----+---+--+--+---------+-----+---------+--------+
# | gen|  id|  x| y| z| distance| time| criteria| fitness|
# +----+----+---+--+--+---------+-----+---------+--------+
# |   1|3315| 10| A| i|      651|  641|      651|     0.0|
# |   1|7681|  8| A| j|      500|   97|      500|   0.258|
# |   1|  12|  1| C| k|       66|   66|       66|     1.0|
# |   1|-331|  4| B| k|      256|   64|      256|   0.675|
# +----+----+---+--+--+---------+-----+---------+--------+

>>> ranked_fitness.datasets.fitness
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+
# | gen|  id|  x| y| z| distance| time| criteria1| criteria2| fitness1| fitness2|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+
# |   1|3315| 10| A| i|      651|  641|       651|       641|      0.0|      1.0|
# |   1|7681|  8| A| j|      500|   97|       500|        97|    0.258|    0.057|
# |   1|  12|  1| C| k|       66|   66|        66|        66|      1.0|    0.003|
# |   1|-331|  4| B| k|      256|   64|       256|        64|    0.675|      0.0|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+

>>> pareto_fitness.datasets.fitness
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+
# | gen|  id|  x| y| z| distance| time| criteria1| criteria2| fitness1| fitness2| front| crowd_dist|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+
# |   1|3315| 10| A| i|      651|  641|       651|       641|      0.0|      1.0|     1|        Inf|
# |   1|7681|  8| A| j|      500|   97|       500|        97|    0.258|    0.057|     1|      0.998|
# |   1|  12|  1| C| k|       66|   66|        66|        66|      1.0|    0.003|     1|        Inf|
# |   1|-331|  4| B| k|      256|   64|       256|        64|    0.675|      0.0|     2|        Inf|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+

# Dataset plots
>>> single_fitness.datasets.plot.output_pair(x='time', y='distance')
# distance
#  |
#  |x    x
#  |  x        x
#  |         x
#  |       x   x
#  |            x
#  |__________________
#                    time

>>> pareto_fitness.datasets.plot.criteria(x='time', y='distance', original_values=False)
# distance
#  |                  o: F1
#  |o    x            x: F2
#  |  o        +      +: F3
#  |         x
#  |       o   x
#  |            o
#  |__________________
#                    time

>>> pareto_fitness.datasets.plot.history.single_objective(objective='time')
# time
#  |             .........
#  |      .......   -----
#  |     .    ------
#  |  ...  ---
#  |.. ----            .: best creature
#  |---                -: average
#  |____________________
#                    generation


# Sort
sorted_data = pareto_fitness.datasets.sort()


# Generate Offspring (Cross Over)
# No need to kill creatures... The weaker ones just wont reproduce
# Strongest get to be part of the next generation (they just wont be tested again)

# Mutate
# the top 50% will be selected as parents and there is a 1% chance of gene mutation
new_pop = pareto_fitness.reproduce.cross_over(p=0.5, p_mutation=0.01)


# After having the offspring, the new population will have a new set of Creatures, so the "Hystory" datasets will keep track of the past
# Generations and their results
>>> new_pop.datasets.history.input
# +----+----+---+--+--+
# | gen|  id|  x| y| z|
# +----+----+---+--+--+
# |   1|3315| 10| A| i|
# |   1|7681|  8| A| j|
# |   1|  12|  1| C| k|
# |   1|-331|  4| B| k|
# +----+----+---+--+--+

>>> new_pop.datasets.history.input_full
# +----+----+---+--+--+
# | gen|  id|  x| y| z|
# +----+----+---+--+--+
# |   1|3315| 10| A| i|
# |   1|7681|  8| A| j|
# |   1|  12|  1| C| k|
# |   1|-331|  4| B| k|
# |   2|3315| 10| A| i|
# |   2|8554| 10| C| k|
# |   2|  12|  1| C| k|
# |   2| 141|  1| A| i|
# +----+----+---+--+--+

>>> new_pop.datasets.history.output
# +----+----+---+--+--+---------+-----+
# | gen|  id|  x| y| z| distance| time|
# +----+----+---+--+--+---------+-----+
# |   1|3315| 10| A| i|      651|  641|
# |   1|7681|  8| A| j|      500|   97|
# |   1|  12|  1| C| k|       66|   66|
# |   1|-331|  4| B| k|      256|   64|
# +----+----+---+--+--+---------+-----+

>>> new_pop.datasets.history.output_full
# +----+----+---+--+--+---------+-----+
# | gen|  id|  x| y| z| distance| time|
# +----+----+---+--+--+---------+-----+
# |   1|3315| 10| A| i|      651|  641|
# |   1|7681|  8| A| j|      500|   97|
# |   1|  12|  1| C| k|       66|   66|
# |   1|-331|  4| B| k|      256|   64|
# |   2|3315| 10| A| i|      651|  641|
# |   2|8554| 10| C| k|         |     |
# |   2|  12|  1| C| k|       66|   66|
# |   2| 141|  1| A| i|         |     |
# +----+----+---+--+--+---------+-----+

>>> new_pop.datasets.history.fitness
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+
# | gen|  id|  x| y| z| distance| time| criteria1| criteria2| fitness1| fitness2| front| crowd_dist|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+
# |   1|3315| 10| A| i|      651|  641|       651|       641|      0.0|      1.0|     1|        Inf|
# |   1|7681|  8| A| j|      500|   97|       500|        97|    0.258|    0.057|     1|      0.998|
# |   1|  12|  1| C| k|       66|   66|        66|        66|      1.0|    0.003|     1|        Inf|
# |   1|-331|  4| B| k|      256|   64|       256|        64|    0.675|      0.0|     2|        Inf|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+

>>> new_pop.datasets.history.fitness_full
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+
# | gen|  id|  x| y| z| distance| time| criteria1| criteria2| fitness1| fitness2| front| crowd_dist|
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+
# |   1|3315| 10| A| i|      651|  641|       651|       641|      0.0|      1.0|     1|        Inf|
# |   1|7681|  8| A| j|      500|   97|       500|        97|    0.258|    0.057|     1|      0.998|
# |   1|  12|  1| C| k|       66|   66|        66|        66|      1.0|    0.003|     1|        Inf|
# |   1|-331|  4| B| k|      256|   64|       256|        64|    0.675|      0.0|     2|        Inf|
# |   2|3315| 10| A| i|      651|  641|       651|       641|         |         |      |           |
# |   2|8554| 10| C| k|         |     |          |          |         |         |      |           |
# |   2|  12|  1| C| k|       66|   66|        66|        66|         |         |      |           |
# |   2| 141|  1| A| i|         |     |          |          |         |         |      |           |
# +----+----+---+--+--+---------+-----+----------+----------+---------+---------+------+-----------+


# Simulate new population (only the untested)
# The part of the population that get to be tested is defined by the intersection of the population's "input" data and the "output" data
>>> pop_out.datasets.untested
# +----+----+---+--+--+
# | gen|  id|  x| y| z|
# +----+----+---+--+--+
# |   2|8554| 10| C| k|
# |   2| 141|  1| A| i|
# +----+----+---+--+--+

new_pop_out = env.simulate(pop_out, multi_threading=True)
