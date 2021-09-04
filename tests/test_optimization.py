import unittest

import numpy as np
from lamarck import BlueprintBuilder, Optimizer
from lamarck.population import Population

from salesman import TravelSalesman


class TestOptimization(unittest.TestCase):
    """
    E2E tests for the Lamarck Optimizer.
    """
    def test_readme_basic_example(self):
        # Defining the Process
        def my_process(x, y):
            val = np.sin(x)*x + np.sin(y)*y
            return {'val': val}

        # Building the Blueprint
        builder = BlueprintBuilder()
        # Adding gene specs
        builder.add_float_gene(name='x',
                               domain=(0, 12*np.pi))

        builder.add_float_gene(name='y',
                               domain=(0, 12*np.pi))

        blueprint = builder.get_blueprint()

        # Building the Population
        popdet = blueprint.populate.deterministic(n=20)
        poprnd = blueprint.populate.random(n=600)
        pop = popdet + poprnd

        # Setting up the Optimizer
        opt = Optimizer(pop, my_process)

        # Simulate (this will return an optimized population)
        opt.simulate.single_criteria(output='val', objective='max')

        # Check the best solution
        print(opt.datasets.get_best_criature(outputs='val', objectives='max'))

        # Assertions
        self.assertIsInstance(opt, Optimizer)
        self.assertIsInstance(opt.population, Population)

    def test_readme_salesman_example(self):
        builder = BlueprintBuilder()

        # Defining the Process
        # In order to persist the TravelSalesman object for the Process, we need to
        # wrap it in a function
        def process_deco(travel_salesman):
            def wrapper(route):
                return {'distance': travel_salesman.get_route_distance(route)}
            return wrapper

        # Genome Blueprint
        # (this will only have one variable that is a "Vector" of the particular order
        # of cities that the salesman should vist)
        number_of_cities = 20
        cities = tuple(range(number_of_cities))
        builder.add_set_gene(name='route',
                             domain=cities,
                             length=number_of_cities)

        # Creating the Population (5000 randomly generated 'route's)
        blueprint = builder.get_blueprint()
        pop = blueprint.populate.random(n=5000)

        # Setting up the Environment
        trav_salesman = TravelSalesman(number_of_cities, seed=123)
        process = process_deco(trav_salesman)

        # Setting up the Optimizer
        opt = Optimizer(population=pop, process=process)

        # Activate MultiThreading (for better performance)
        opt.config.multithread = True

        # Simulate (this will return an optimized population)
        opt.simulate.single_criteria(output='distance', objective='min')

        # Check the best solution
        print(opt.datasets.get_best_criature())

        # Assertions
        self.assertIsInstance(opt, Optimizer)
        self.assertIsInstance(opt.population, Population)
