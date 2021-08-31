import unittest
from typing import Callable
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from lamarck import Blueprint, Optimizer


class testOptimizerFitness(unittest.TestCase):
    """
    Optimizer Datasets tests.

    Tests
    -----
    1. Initial datasets sanity
        1.1. population dataset
        1.2. results dataset
        1.3. fitness dataset
        1.4. history dataset
        1.5. simulation dataset
    2. Results datasets after a simulation
    """
    blueprint: Blueprint
    process: Callable[[], dict]

    def setUp(self):
        blueprint_dict = {
            'min_size': {
                'type': 'integer',
                'specs': {'domain': [6, 10]}},
            'max_size': {
                'type': 'integer',
                'specs': {'domain': [8, 24]}},
            'price': {
                'type': 'float',
                'specs': {'domain': [100, 250]}},
            'brand': {
                'type': 'categorical',
                'specs': {'domain': ['AMTA', 'REPAL', 'NOSOR']}},
            'sick_pattern': {
                'type': 'array',
                'specs': {'domain': ['L', 'R'],
                          'length': 3}},
            'groove': {
                'type': 'set',
                'specs': {'domain': ['hh', 'bd', 'sn'],
                          'length': 3}},
            'is_loud': {
                'type': 'boolean',
                'specs': {}}
        }

        def process(min_size, max_size, price, brand, sick_pattern, groove, is_loud):
            brand_factor = {
                'AMTA': 3.7,
                'REPAL': 3.2,
                'NOSOR': 4.6
            }
            gear_sickness = (max_size/min_size) * brand_factor[brand] * (2+int(is_loud))
            price_unbelievebleness = (price / gear_sickness) * min_size
            pattern_val = hash(sick_pattern) % 1000
            groove_val = hash(groove) % 5000
            groovy = pattern_val + groove_val
            return {
                'gear_sickness': gear_sickness,
                'price_unbelievebleness': price_unbelievebleness,
                'grooveness': groovy
            }

        self.blueprint = Blueprint(blueprint_dict)
        self.process = process

    def test_single_objective_fitness(self):
        """
        Test if the Fitness dataset is correctly defined for the single objective
        criteria.

        Tests
        -----
        1. opt.dataset.fitness data
        2. opt.dataset.simulation data
        """
        pop = self.blueprint.populate.deterministic(n=3)
        opt = Optimizer(population=pop, process=self.process)
        opt.run(quiet=True)
        opt.apply_fitness.single_criteria(output='gear_sickness', objective='max')

        # Test 1
        rank = opt.datasets.results['gear_sickness']\
            .rank(method='min', ascending=False)\
            .astype(int)\
            .rename('Rank')
        concat_data = (
            opt.datasets.results['gear_sickness'].rename('Criteria'),
            rank
        )
        expected = pd.concat(concat_data, axis=1)
        actual = opt.datasets.fitness
        assert_frame_equal(expected, actual)

        # Test 2
        concat_data = (
            opt.datasets.population,
            opt.datasets.results,
            opt.datasets.results['gear_sickness'].rename('Criteria'),
            rank
        )
        expected = pd.concat(concat_data, axis=1)
        actual = opt.datasets.simulation
        assert_frame_equal(expected, actual)

    def test_multi_objective_ranked_fitness(self):
        """
        Test if the Fitness dataset is correctly defined for the multi objective
        "ranked" criteria.

        Tests
        -----
        1. opt.dataset.fitness data with 2 objectives
        2. opt.dataset.simulation data with 2 objective
        3. opt.dataset.fitness data with 3 objectives
        4. opt.dataset.simulation data with 3 objective
        """
        popdet = self.blueprint.populate.deterministic(n=3)
        poprnd = self.blueprint.populate.random(n=1000, seed=42)
        pop = pd.concat((popdet, poprnd)).reset_index(drop=True)
        opt = Optimizer(population=pop, process=self.process)
        ascmap = {'min': True, 'max': False}

        def multirank(df, priorities, objectives):
            ranks = []
            for priority, objective in zip(priorities, objectives):
                asc = ascmap[objective]
                r = df[priority]\
                    .rank(method='min', ascending=asc)
                ranks.append(r)
            rank = ranks[-1]
            for r in ranks[::-1]:
                order = int(np.log10(r.max())) + 1
                factor = 10**order
                rscore = r * factor + rank
                rank = rscore.rank(method='min')
            return rank.astype(int)

        opt.run(quiet=True)

        # Tests 1 & 2 - Setup
        outputs = ['gear_sickness', 'price_unbelievebleness']
        objectives = ['max', 'min']
        opt.apply_fitness.multi_criteria_ranked(outputs=outputs, objectives=objectives)
        rank = multirank(opt.datasets.results, outputs, objectives).rename('Rank')

        # Test 1
        concat_data = (
            [opt.datasets.results[col].rename(f'Criteria{i+1}')
             for i, col in enumerate(outputs)]
            + [rank]
        )
        expected = pd.concat(concat_data, axis=1)
        actual = opt.datasets.fitness
        assert_frame_equal(expected, actual)

        # Test 2
        concat_data = (
            [opt.datasets.population, opt.datasets.results]
            + [opt.datasets.results[col].rename(f'Criteria{i+1}')
               for i, col in enumerate(outputs)]
            + [rank]
        )
        expected = pd.concat(concat_data, axis=1)
        actual = opt.datasets.simulation
        assert_frame_equal(expected, actual)

        # Tests 3 & 4 - Setup
        outputs = ['gear_sickness', 'price_unbelievebleness', 'grooveness']
        objectives = ['max', 'min', 'max']
        opt.apply_fitness.multi_criteria_ranked(outputs=outputs, objectives=objectives)
        rank = multirank(opt.datasets.results, outputs, objectives).rename('Rank')

        # Test 3
        concat_data = (
            [opt.datasets.results[col].rename(f'Criteria{i+1}')
             for i, col in enumerate(outputs)]
            + [rank]
        )
        expected = pd.concat(concat_data, axis=1)
        actual = opt.datasets.fitness
        assert_frame_equal(expected, actual)

        # Test 4
        concat_data = (
            [opt.datasets.population, opt.datasets.results]
            + [opt.datasets.results[col].rename(f'Criteria{i+1}')
               for i, col in enumerate(outputs)]
            + [rank]
        )
        expected = pd.concat(concat_data, axis=1)
        actual = opt.datasets.simulation
        assert_frame_equal(expected, actual)

    def test_multi_objective_pareto_fitness(self):
        """
        Test if the Fitness dataset is correctly defined for the multi objective
        "pareto" criteria.
        """
        def process(x, y):
            return {'out1': x*y, 'out2': x-y}

        blueprint_dict = {
            'x': {
                'type': 'integer',
                'specs': {
                    'domain': [10, 60]}},
            'y': {
                'type': 'integer',
                'specs': {
                    'domain': [1, 20]}},
        }
        blueprint = Blueprint(blueprint_dict)
        pop = blueprint.populate.deterministic(n=3)
        opt = Optimizer(pop, process)
        opt.run()
        opt.apply_fitness.multi_criteria_pareto(['out1', 'out2'], ['max', 'min'])

        results_data = {
            'out1': [10, 100, 200, 35, 350, 700, 60, 600, 1200],
            'out2':   [9, 0,  -10, 34, 25,  15,  59, 50,  40]
        }
        fronts_data = [3, 2, 1, 3, 2, 1, 3, 2, 1]
        crowds_data = [np.inf, np.inf, np.inf, 100., 550., 1050., np.inf, np.inf, np.inf]

        index = opt.datasets.population.index
        results = pd.DataFrame(results_data, index=index)
        fronts = pd.Series(fronts_data, index=index, name='Front')
        crowds = pd.Series(crowds_data, index=index, name='Crowd')
        r1 = fronts.rank(method='dense', ascending=True)
        r2 = crowds.rank(method='dense', ascending=False)
        order1 = int(np.log10(r2.max())) + 1
        factor1 = 10**order1
        ranks = (r1 * factor1 + r2).rank(method='min').astype(int).rename('Rank')

        # Test Results Data
        expected = results
        actual = opt.datasets.results
        assert_frame_equal(expected, actual)

        # Test Fitness Data
        expected = pd.concat((fronts, crowds, ranks), axis=1)
        actual = opt.datasets.fitness
        assert_frame_equal(expected, actual)
