import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from lamarck import Blueprint, Optimizer
from lamarck.reproduce import Reproductor


class TestReproduction(unittest.TestCase):
    """
    Tests for the multiple reproduciton methods available.

    Reproduction Methods
    --------------------
    - Tournament
    - Elitism
    - Mutations
    """
    blueprint: Blueprint
    reproductor: Reproductor
    opt = Optimizer

    def setUp(self):
        blueprint_dict = {
            'min_size': {
                'type': 'numeric',
                'specs': {'domain': int,
                          'range': [6, 10]}},
            'max_size': {
                'type': 'numeric',
                'specs': {'domain': int,
                          'range': [8, 24]}},
            'price': {
                'type': 'numeric',
                'specs': {'domain': float,
                          'range': [100, 250]}},
            'brand': {
                'type': 'categorical',
                'specs': {'domain': ['AMTA', 'REPAL', 'NOSOR']}},
            'sick_pattern': {
                'type': 'vectorial',
                'specs': {'domain': ['L', 'R'],
                          'replacement': True,
                          'length': 3}},
            'groove': {
                'type': 'vectorial',
                'specs': {'domain': ['hh', 'bd', 'sn'],
                          'replacement': False,
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
        self.reproductor = Reproductor(self.blueprint)
        self.opt = Optimizer(pop, process)

    def test_cross_over(self):
        """
        Test the Cross-Over method.

        Tests
        -----
        1.  2 parents, 2 numeric genes, 2 offspring
        2.  2 parents, 3 numeric genes, 2 offspring
        """
        # Test 1
        parents_data = {
            'min_size': [8, 10],
            'max_size': [10, 15]
        }
        parents = pd.DataFrame(parents_data)
        expected_data = {
            'min_size': [8, 15],
            'max_size': [10, 10]
        }
        expected = pd.DataFrame(expected_data)
        actual = self.reproductor.cross_over(parents, n_offspring=2)
        assert_frame_equal(expected, actual)

    def test_tournament_method(self):
        """
        The Tournament method consist on picking N (normally 2) Creatures randomly and assign
        the fittest of them to be the first parent, then repeating the same process again to
        combine 2 parents to generate the 2 offspring.

        Tests
        -----
        1. 2-way battle Tournament to generate 2 parents and 2 offspring
        2. 3-way battle Tournament to generate 2 parents and 2 offspring
        3. 4-way battle Tournament to generate 2 parents and 2 offspring
        """
        self.opt.run(quiet=True)
        self.opt.apply_fitness.single_criteria(output='gear_sickness', objective='max')
        sim_data = self.opt.datasets.simulation

        # Test 1
        # expected = pd.DataFrame()
        # selected_data = sim_data.sort_values('Rank')[0:5]
        # actual = self.reproductor.tournament(selected_data,
        #                                      total_offspring=6,
        #                                      n_dispute=2,
        #                                      n_parents=2,
        #                                      n_offspring=2)
        # assert_frame_equal(expected, actual)
