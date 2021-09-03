import unittest
from typing import Callable
import hashlib
import pandas as pd
from pandas.testing import assert_frame_equal

from lamarck import Blueprint, Optimizer


class testOptimizerDatasets(unittest.TestCase):
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

    def test_datasets_construction(self):
        """
        Test if the initial datasets are adequatetly set with the hashed id's as
        the index and the empty result columns in the :results:, :fitness: and
        :history: datasets.

        Also tests if the data concatenation that constitutes
        the :simulation: dataset is working es espected.
        """
        pop = self.blueprint.populate.deterministic(n=2)
        opt = Optimizer(population=pop, process=self.process)

        # population
        pop_data = {
            'min_size': [6]*5,
            'max_size': [8]*5,
            'price': [100.]*5,
            'brand': ['AMTA']*5,
            'sick_pattern': [('L', 'L', 'L')]*4 + [('R', 'R', 'R')],
            'groove': [('hh', 'bd', 'sn'), ('hh', 'bd', 'sn'),
                       ('sn', 'bd', 'hh'), ('sn', 'bd', 'hh'),
                       ('hh', 'bd', 'sn')],
            'is_loud': [False, True, False, True, False]
        }
        expected_index_encoded_strings = [
            str(tuple(arr[i] for arr in pop_data.values())).encode() for i in range(5)
        ]
        expected_index_strings = [hashlib.sha256(encoded).hexdigest()
                                  for encoded in expected_index_encoded_strings]
        expected_index = pd.Series(expected_index_strings)
        expected_index.name = 'id'
        expected = pd.DataFrame(pop_data, index=expected_index)
        actual = opt.datasets.population.head()
        assert_frame_equal(expected, actual)

        # results
        results_cols = ('gear_sickness', 'price_unbelievebleness', 'grooveness')
        expected = pd.DataFrame(columns=results_cols, index=expected_index)
        actual = opt.datasets.results.head()
        assert_frame_equal(expected, actual)

        # fitness
        expected = pd.DataFrame(index=expected_index)
        actual = opt.datasets.fitness.head()
        assert_frame_equal(expected, actual)

        # history
        expected = pd.DataFrame(columns=['generation'], dtype=int)
        actual = opt.datasets.history
        assert_frame_equal(expected, actual)

        # simulation
        datasets = (
            pd.DataFrame(pop_data, index=expected_index),
            pd.DataFrame(columns=results_cols, index=expected_index),
            pd.DataFrame(index=expected_index)
        )
        expected = pd.concat(datasets, axis=1)
        actual = opt.datasets.simulation.head()
        assert_frame_equal(expected, actual)

    def test_simulation_outputs(self):
        """
        Run the population and check if the output data is sane.
        """
        pop = self.blueprint.populate.deterministic(n=3)
        opt = Optimizer(population=pop, process=self.process)
        inputs = {
            'min_size': [6]*5,
            'max_size': [8]*5,
            'price': [100.]*5,
            'brand': ['AMTA']*5,
            'sick_pattern': [('L', 'L', 'L')]*5,
            'groove': [('hh', 'bd', 'sn'), ('hh', 'bd', 'sn'),
                       ('bd', 'hh', 'sn'), ('bd', 'hh', 'sn'),
                       ('sn', 'bd', 'hh')],
            'is_loud': [False, True, False, True, False]
        }
        expected_data = {
            'id': [],
            'gear_sickness': [],
            'price_unbelievebleness': [],
            'grooveness': [],
        }
        for i in range(5):
            params = {}
            for key, vals in inputs.items():
                params.update({key: vals[i]})
            encoded_index = str(tuple([v for v in params.values()])).encode()
            index_str = hashlib.sha256(encoded_index).hexdigest()
            out_dict = self.process(**params)
            expected_data['id'].append(index_str)
            expected_data['gear_sickness'].append(out_dict['gear_sickness'])
            expected_data['price_unbelievebleness'].append(out_dict['price_unbelievebleness'])
            expected_data['grooveness'].append(out_dict['grooveness'])
        out_data = pd.DataFrame(expected_data).set_index('id')

        expected = out_data.sort_index()
        opt.run(quiet=True)
        actual = opt.datasets.results.head().sort_index()
        assert_frame_equal(expected, actual)
