import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from lamarck import Blueprint, Optimizer
from lamarck.reproduce import (Populator,
                               select_parents_by_tournament)
from lamarck.utils import ParentOverloadException


class TestReproduction(unittest.TestCase):
    """
    Tests for the multiple reproduciton methods available.

    Reproduction Methods
    --------------------
    - Sexual
        - Cross-Over methods
        - Tournament
    - Asexual (mutation)
    """

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
        pop = self.blueprint.populate.random(n=10, seed=42)
        self.populator = Populator(self.blueprint)
        self.opt = Optimizer(pop, process)
        self.opt.run_multithread(quiet=True)

    def test_tournament(self):
        """
        Assert that tournament behaviours are sane.

        Tests
        -----
        1. Assert number of parents is equal to n_parents
        2. Assert it raises exception if total number of parents is lower then the number of
           solicited parents
        3. Assert if elitism lock works (that is if the number of n_parents and n_dispute end up
           creating a situation where the top n_parents are necessarily selected)
        """
        self.opt.run_multithread()
        self.opt.apply_fitness.single_criteria(output='gear_sickness', objective='max')
        ranked_data = self.opt.datasets.fitness
        n_selection = int(round(len(ranked_data) * 0.5))
        ranked_pop = ranked_data.sort_values('Rank')[0:n_selection]

        #  Test 1
        parents = select_parents_by_tournament(ranked_pop, n_dispute=2, n_parents=2, seed=42)
        expected = 2
        actual = len(parents)
        self.assertEqual(expected, actual)

        parents = select_parents_by_tournament(ranked_pop, n_dispute=2, n_parents=3, seed=42)
        expected = 3
        actual = len(parents)
        self.assertEqual(expected, actual)

        parents = select_parents_by_tournament(ranked_pop, n_dispute=3, n_parents=3, seed=42)
        expected = 3
        actual = len(parents)
        self.assertEqual(expected, actual)

        # Test 2
        with self.assertRaises(ParentOverloadException):
            select_parents_by_tournament(ranked_pop, n_dispute=4, n_parents=3, seed=42)

        with self.assertRaises(ParentOverloadException):
            select_parents_by_tournament(ranked_pop, n_dispute=3, n_parents=4, seed=42)

        # Test 3
        sortcols = ['Rank', 'id']
        expected = ranked_pop.reset_index().sort_values(by=sortcols)[0:3].reset_index(drop=True)
        actual = select_parents_by_tournament(ranked_pop, n_dispute=3, n_parents=3)\
            .reset_index()\
            .sort_values(by=sortcols)\
            .reset_index(drop=True)
        assert_frame_equal(expected, actual)

    def test_sexual_offspring_generation(self):
        """
        Assert that the sexual reproduction method is generating the offspring population
        correctly.

        Tests
        -----
        0. Validate schema
            0.1. Making 1 children
            0.2. Making 10 children
        1. Assert that the number of generated offspring correspond to the :n_offspring: param.
            1.1. 4 children (children_per_relation=2) - must result 4 children
            1.2. 5 children (children_per_relation=2) - must result 6 children
            1.3. 4 children (children_per_relation=3) - must result 6 children
            1.4. 5 children (children_per_relation=1) - must result 5 children
            1.5. 1 children (children_per_relation=3) - must result 3 children
            1.6. 1 children (children_per_relation=1) - must result 1 child
        2. Distinct children is equal or greater than the number of children_per_relation
            2.1. n_parents=2, children_per_relation=2
            2.2. n_parents=2, children_per_relation=3
            2.3. n_parents=3, children_per_relation=3
            2.4. n_parents=3, children_per_relation=4
        """
        self.opt.apply_fitness.single_criteria(output='gear_sickness', objective='max')
        ranked_data = self.opt.datasets.simulation
        n_selection = int(round(len(ranked_data) * 0.5))
        ranked_pop = ranked_data.sort_values('Rank')[0:n_selection]

        # Test 0
        expected_data = {
            'min_size': np.dtype(int),
            'max_size': np.dtype(int),
            'price': np.dtype(float),
            'brand': np.dtype(object),
            'sick_pattern': np.dtype(object),
            'groove': np.dtype(object),
            'is_loud': np.dtype(bool),
        }
        expected = pd.Series(expected_data)

        # Test 0.1
        offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                          n_offspring=1,
                                          n_dispute=2,
                                          n_parents=2,
                                          children_per_relation=1,
                                          rank_column='Rank',
                                          seed=42)
        actual = offspring.dtypes
        assert_series_equal(expected, actual)

        # Test 0.2
        offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                          n_offspring=10,
                                          n_dispute=2,
                                          n_parents=2,
                                          children_per_relation=2,
                                          rank_column='Rank',
                                          seed=42)
        actual = offspring.dtypes
        assert_series_equal(expected, actual)

        n_trials = 500
        for _ in range(n_trials):
            # Test 1
            # Test 1.1
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=4,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=2,
                                              rank_column='Rank',
                                              seed=42)
            expected = 4
            actual = len(offspring)
            self.assertEqual(expected, actual)

            # Test 1.2
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=5,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=2,
                                              rank_column='Rank',
                                              seed=42)
            expected = 6
            actual = len(offspring)
            self.assertEqual(expected, actual)

            # Test 1.3
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=4,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=3,
                                              rank_column='Rank',
                                              seed=42)
            expected = 6
            actual = len(offspring)
            self.assertEqual(expected, actual)

            # Test 1.4
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=5,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=1,
                                              rank_column='Rank',
                                              seed=42)
            expected = 5
            actual = len(offspring)
            self.assertEqual(expected, actual)

            # Test 1.5
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=1,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=3,
                                              rank_column='Rank',
                                              seed=42)
            expected = 3
            actual = len(offspring)
            self.assertEqual(expected, actual)

            # Test 1.6
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=1,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=1,
                                              rank_column='Rank',
                                              seed=42)
            expected = 1
            actual = len(offspring)
            self.assertEqual(expected, actual)

            # Test 2
            # Test 2.1
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=4,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=2,
                                              rank_column='Rank',
                                              seed=42)
            expected = 2
            actual = len(offspring.drop_duplicates())
            self.assertGreaterEqual(actual, expected)

            # Test 2.2
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=4,
                                              n_dispute=2,
                                              n_parents=2,
                                              children_per_relation=3,
                                              rank_column='Rank',
                                              seed=42)
            expected = 3
            actual = len(offspring.drop_duplicates())
            self.assertGreaterEqual(actual, expected)

            # Test 2.3
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=4,
                                              n_dispute=2,
                                              n_parents=3,
                                              children_per_relation=3,
                                              rank_column='Rank',
                                              seed=42)
            expected = 3
            actual = len(offspring.drop_duplicates())
            self.assertGreaterEqual(actual, expected)

            # Test 2.4
            offspring = self.populator.sexual(ranked_pop=ranked_pop,
                                              n_offspring=4,
                                              n_dispute=2,
                                              n_parents=3,
                                              children_per_relation=4,
                                              rank_column='Rank',
                                              seed=42)
            expected = 4
            actual = len(offspring.drop_duplicates())
            self.assertGreaterEqual(actual, expected)

    def test_asexual_offspring_generation(self):
        """
        Assert that the sexual reproduction method is generating the offspring population
        correctly.

        Tests
        -----
        0. Validate schema
            0.1. Making 1 children
            0.2. Making 10 children
        1. Assert that the number of generated offspring correspond to the :n_offspring: param.
            1.1. 1 child
            1.2. 5 children
            1.3. 10 children
        """
        self.opt.apply_fitness.single_criteria(output='gear_sickness', objective='max')
        ranked_data = self.opt.datasets.simulation
        n_selection = int(round(len(ranked_data) * 0.5))
        ranked_pop = ranked_data.sort_values('Rank')[0:n_selection]

        # Test 0
        expected_data = {
            'min_size': np.dtype(int),
            'max_size': np.dtype(int),
            'price': np.dtype(float),
            'brand': np.dtype(object),
            'sick_pattern': np.dtype(object),
            'groove': np.dtype(object),
            'is_loud': np.dtype(bool),
        }
        expected = pd.Series(expected_data)

        # Test 0.1
        offspring = self.populator.asexual(ranked_pop=ranked_pop,
                                           n_offspring=1,
                                           n_mutated_genes=1,
                                           children_per_creature=1,
                                           rank_column='Rank',
                                           seed=42)
        actual = offspring.dtypes
        assert_series_equal(expected, actual)

        # Test 0.2
        offspring = self.populator.asexual(ranked_pop=ranked_pop,
                                           n_offspring=10,
                                           n_mutated_genes=1,
                                           children_per_creature=1,
                                           rank_column='Rank',
                                           seed=42)
        actual = offspring.dtypes
        assert_series_equal(expected, actual)

        # Test 1.1
        offspring = self.populator.asexual(ranked_pop=ranked_pop,
                                           n_offspring=1,
                                           n_mutated_genes=1,
                                           rank_column='Rank',
                                           seed=42)
        expected = 1
        actual = len(offspring)
        self.assertEqual(expected, actual)

        # Test 1.2
        offspring = self.populator.asexual(ranked_pop=ranked_pop,
                                           n_offspring=5,
                                           n_mutated_genes=1,
                                           children_per_creature=1,
                                           rank_column='Rank',
                                           seed=42)
        expected = 5
        actual = len(offspring)
        self.assertEqual(expected, actual)

        # Test 1.3
        offspring = self.populator.asexual(ranked_pop=ranked_pop,
                                           n_offspring=10,
                                           n_mutated_genes=1,
                                           children_per_creature=1,
                                           rank_column='Rank',
                                           seed=42)
        expected = 10
        actual = len(offspring)
        self.assertEqual(expected, actual)

        # Test 1.4
        offspring = self.populator.asexual(ranked_pop=ranked_pop,
                                           n_offspring=1,
                                           n_mutated_genes=1,
                                           children_per_creature=3,
                                           rank_column='Rank',
                                           seed=42)
        expected = 3
        actual = len(offspring)
        self.assertEqual(expected, actual)
