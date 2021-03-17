from copy import deepcopy
from tqdm import tqdm
import numpy as np

from lamarck.assistant import GenomeCreator
from lamarck.assistant import PopulationCreator
from lamarck import Environment


class Optimizer:
    """
    Single or Multi-Objective Genetic Algorithm.

    >>> opt = Optimizer()

    Basic Steps:
        1. Define the Variable Space by setting up the Genome Blueprint
            >>> opt.genome_creator.add_gene_specs.numeric(**kw) # for numeric variables
            >>> opt.genome_creator.add_gene_specs.categorical(**kw) # for categorical variables
            >>> opt.genome_creator.add_gene_specs.vectorial(**kw) # for vectorial variables

        2. Create a Population
            >>> opt.create_population(n_det, n_rand)

        3. Set the Environment: Define the Process Function
            >>> opt.set_process(my_function)

        4. Run
            - Single Objective:
            >>> simulated_pop = opt.run.single_objective(output, objective, **kw)
            - Multi Objective: Ranked
            >>> simulated_pop = opt.run.multi_objective.ranked(priorities, objectives, **kw)
            - Multi Objective: Pareto
            >>> simulated_pop = opt.run.multi_objective.pareto(outputs, objectives, **kw)
    """

    def __init__(self):
        self.genome_creator = GenomeCreator()
        self.pop = None
        self.simulated_pop = None
        self.env = Environment()
        self.run = Runner(self)
        self.set_genome_blueprint()

    @property
    def genome_blueprint(self):
        if self._gb is not None:
            return self._gb
        else:
            return self.genome_creator.get_genome_blueprint()

    @property
    def add_constraint(self):
        if self.pop is not None:
            return self.pop.apply_fitness.add_constraint

    def create_population(self, n_det=0, n_rand=0):
        """
        Create a Population by mixing evenly spaced and random solution ranges.
        
        For pure evenly spaced values, let n_rand be zero; and for pure random
        values, let n_det be zero.

        Parameters
        ----------
        :n_det:     `int` Number of evenly spaced values (subdivisions). The values
                    are selected according to each Gene's Blueprint.
                    Examples:
                        1. numeric linear: min=10, max=90, n_det=5
                            - values = (10, 30, 50, 70, 90)
                        2. numeric log: min=10, max=1000, n_det=3
                            - values = (10, 100, 1000)
                        3. categorical: domain=('A', 'B', 'C', 'D', 'E'), n_det=3
                            - values = ('A', 'C', 'E')
                        4. vectorial: domain=(1, 2, 3, 4), length=3, replace=True, ndet=4
                            - values = (1, 1, 1),
                                       (2, 1, 1),
                                       (3, 1, 1),
                                       (4, 1, 1)
                        5. vectorial: domain=(1, 2, 3, 4), length=3, replace=False, ndet=4
                            - values = (1, 2, 3),
                                       (2, 1, 3),
                                       (3, 1, 2),
                                       (4, 1, 2)
                    - The Population size will normally be the number n_det to the
                    power of the number of genes, except if some duplicate genomes
                    are generated (e.g. 4 genes and n_det=5 will end up generating
                    5**4=625 different combinations of solutions).

        :n_rand:    `int` Number of randomly generated solutions. The values will
                    be randomly generated and assigned to each Genome individually.
                    - The number 'n_rand' will normally end up being the Population
                    size, except if some duplicate genomes are generated.
        """
        pop_creator = PopulationCreator(self.genome_blueprint)
        self.pop = (pop_creator.create.det(n_det)
                    + pop_creator.create.rand(n_rand))
        self.pop.define()
    
    def set_genome_blueprint(self, genome_blueprint=None):
        """
        Manually set the Genome Blueprint. If `None`, the genome_creator's blueprint
        will be used.
        """
        self._gb = genome_blueprint

    def set_population(self, pop):
        """
        Manually insert a Population to be tested. The Population's Genome Blueprint
        will override the genome_creator's.
        """
        self._gb = deepcopy(pop.genome_blueprint)
        self.pop = pop
        self.pop.define()

    def set_process(self, process):
        """
        Define the Process (Python Function) in which the input parameters will be
        optimized by the algorithm.
        """
        self.env.config.set_process(process)


class Runner:
    def __init__(self, opt):
        self._opt = opt
        self.multi_objective = RunnerMulti(opt)

    def single_objective(self, output, objective,
                         max_generations=20, max_stall=None,
                         p_selection=0.5,
                         p_tournament=0.5, n_dispute=2,
                         p_elitism=0.5,
                         p_mutation=0.05, n_per_gene=1):
        """
        Run an Single-Objective Optimization.

        Parameters
        ----------
        :output:            `str` Name of the Variable to be optimized.
        :objective:         `str` Objective ('max', for "maximize"; 'min', for
                            "minimize").
        :max_generations:   `int` Maximum number of Generations (default: 20).
        :max_stall:         `int` Maximum number of times that the simulation
                            doesn't provide a better solution. If `None` is set, the
                            simulation will necessarily run until the
                            :max_generation: is reached (default: None).
        :p_selection:       `float` proportion of the population that will be
                            selected and will be potential Parents to the next
                            Generation (default: 0.5).
        :p_tournament:      `float` proportion of the population that will be
                            generated by the Tournament Selection Model (default:
                            0.5).
        :n_dispute:         `int` Number of selected Creatures that will compete for
                            one of the two Parent spots at any time in the
                            Tournament Selection (default: 2).
        :p_elitism:         `float` proportion of the population that will be
                            generated by the Elitism Selection Model (default: 0.5).
        :p_mutation:        `float` proportion of the population that will generate
                            new Creatures by mutating :n_per_gene: aspects of its
                            genome (default: 0.05).
        :n_per_gene:        `int` number of genes that are mutated (default: 1)
        """
        pop = self._opt.pop.copy()
        env = self._opt.env
        default_simulation('single', pop, env, output, objective,
                           max_generations, max_stall, p_selection,
                           p_tournament, n_dispute, p_elitism,
                           p_mutation, n_per_gene)
        self._opt.simulated_pop = pop
        return pop


class RunnerMulti:
    def __init__(self, opt):
        self._opt = opt

    def pareto(self, outputs, objectives,
               max_generations=20, max_stall=None,
               p_selection=0.5,
               p_tournament=0.5, n_dispute=2,
               p_elitism=0.5,
               p_mutation=0.05, n_per_gene=1):
        """
        Run a Multi-Objective Optimization using the Pareto Fronts to select the
        best range of solutions by finding their unique set of traits, giving the
        desired Outputs and Objectives.

        Parameters
        ----------
        :outputs:           `list` of Variable Names (`str`) to be optimized.
        :objectives:        `list` of Objectives (`str`); ('max', for "maximize";
                            'min', for "minimize").
        :max_generations:   `int` Maximum number of Generations (default: 20).
        :max_stall:         `int` Maximum number of times that the simulation
                            doesn't provide a better solution. If `None` is set, the
                            simulation will necessarily run until the
                            :max_generation: is reached (default: None).
        :p_selection:       `float` proportion of the population that will be
                            selected and will be potential Parents to the next
                            Generation (default: 0.5).
        :p_tournament:      `float` proportion of the population that will be
                            generated by the Tournament Selection Model (default:
                            0.5).
        :n_dispute:         `int` Number of selected Creatures that will compete for
                            one of the two Parent spots at any time in the
                            Tournament Selection (default: 2).
        :p_elitism:         `float` proportion of the population that will be
                            generated by the Elitism Selection Model (default: 0.5).
        :p_mutation:        `float` proportion of the population that will generate
                            new Creatures by mutating :n_per_gene: aspects of its
                            genome (default: 0.05).
        :n_per_gene:        `int` number of genes that are mutated (default: 1)
        """
        pop = self._opt.pop.copy()
        env = self._opt.env
        default_simulation('pareto', pop, env, outputs, objectives,
                           max_generations, max_stall, p_selection,
                           p_tournament, n_dispute, p_elitism,
                           p_mutation, n_per_gene)
        self._opt.simulated_pop = pop
        return pop

    def ranked(self, priorities, objectives,
               max_generations=20, max_stall=None,
               p_selection=0.5,
               p_tournament=0.5, n_dispute=2,
               p_elitism=0.5,
               p_mutation=0.05, n_per_gene=1):
        """
        Run a Multi-Objective Optimization using the Ranked Priority Method to
        select the best range of solutions by ordering them according to the
        "Priorities" list and its respective Objectives.

        Parameters
        ----------
        :priorities:        `list` of Variable Names (`str`) to be optimized. The
                            order of prioritization follows the order of the
                            Variable list.
        :objectives:        `list` of Objectives (`str`); ('max', for "maximize";
                            'min', for "minimize").
        :max_generations:   `int` Maximum number of Generations (default: 20).
        :max_stall:         `int` Maximum number of times that the simulation
                            doesn't provide a better solution. If `None` is set, the
                            simulation will necessarily run until the
                            :max_generation: is reached (default: None).
        :p_selection:       `float` proportion of the population that will be
                            selected and will be potential Parents to the next
                            Generation (default: 0.5).
        :p_tournament:      `float` proportion of the population that will be
                            generated by the Tournament Selection Model (default:
                            0.5).
        :n_dispute:         `int` Number of selected Creatures that will compete for
                            one of the two Parent spots at any time in the
                            Tournament Selection (default: 2).
        :p_elitism:         `float` proportion of the population that will be
                            generated by the Elitism Selection Model (default: 0.5).
        :p_mutation:        `float` proportion of the population that will generate
                            new Creatures by mutating :n_per_gene: aspects of its
                            genome (default: 0.05).
        :n_per_gene:        `int` number of genes that are mutated (default: 1)
        """
        pop = self._opt.pop.copy()
        env = self._opt.env
        default_simulation('ranked', pop, env, priorities, objectives,
                           max_generations, max_stall, p_selection,
                           p_tournament, n_dispute, p_elitism,
                           p_mutation, n_per_gene)
        self._opt.simulated_pop = pop
        return pop


def default_simulation(kind, pop, env, outputs, objectives, max_generations,
                       max_stall, p_selection, p_tournament, n_dispute,
                       p_elitism, p_mutation, n_per_gene):
    stall = 0
    pbar = tqdm(range(max_generations))
    for _ in pbar:
        env.simulate(pop)
        if kind == 'single':
            pop.apply_fitness.single_objective(outputs, objectives)
        elif kind == 'ranked':
            pop.apply_fitness.multi_objective.ranked(outputs, objectives)
        elif kind == 'pareto':
            pop.apply_fitness.multi_objective.pareto(outputs, objectives,
                                                     p_selection)
        if max_stall is not None:
            stall = compute_stall(pop)
            pbar.set_description(f'Stall: {stall}/{max_stall}   ')
            if stall >= max_stall:
                break
        pop.select(p_selection)
        if p_mutation > 0:
            pop.reproduce.mutate(p_mutation)
        if p_tournament > 0:
            pop.reproduce.tournament(n_dispute, p_children=p_tournament)
        if p_elitism > 0:
            pop.reproduce.elitism(p_children=p_elitism)


def compute_stall(pop):
    best_creatures = pop.get_creature.best()
    best_creatures_ids = np.array([creature.id for creature in best_creatures])
    best_ids_list = []
    for gen in range(pop.generation):
        gen_fitness = pop.datasets.get_generation_from_history(gen)
        gen_ranks = pop.fitness_rank_method(gen_fitness)
        f_best = gen_ranks == 1
        best_gen_ids = gen_ranks[f_best].index
        best_ids_list.append(best_gen_ids)
    comparison = [list_similarity(best_creatures_ids, best_ids) > 0.5
                  for best_ids in best_ids_list]
    return sum(comparison)


def list_similarity(list1, list2):
    n = len(list1)
    return sum([item in list2 for item in list1]) / n
