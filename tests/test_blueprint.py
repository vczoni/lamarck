import unittest
import itertools
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from lamarck import Blueprint


class TestBlueprint(unittest.TestCase):
    """
    Test cases for the Blueprint class.
    """
    blueprint_dict: dict

    def setUp(self):
        self.blueprint_dict = {
            'min_size': {
                'type': 'integer',
                'specs': {'domain': (6, 10)}},
            'max_size': {
                'type': 'integer',
                'specs': {'domain': (8, 24)}},
            'price': {
                'type': 'float',
                'specs': {'domain': (100, 250)}},
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

    def test_deterministic_population_creation(self):
        """
        Test if the deterministic combination of all sets of elements is being
        correctly generated.

        Tests
        -----
        1. N = 2 (2**7 = 128 combinations)
        2. N = 3 (3**6 * 2 = 1458 combinations)
        3. N = (3, 3, 5, 3, 6, 6, 2) (9720 combinations)
        """
        blueprint = Blueprint(self.blueprint_dict)
        input_cols = list(self.blueprint_dict.keys())

        # Test 1
        items = (
            [6, 10],
            [8, 24],
            [100., 250.],
            ['AMTA', 'REPAL'],
            [('L', 'L', 'L'), ('R', 'R', 'R')],
            [('hh', 'bd', 'sn'), ('sn', 'bd', 'hh')],
            [False, True],
        )
        items = [sorted(item) for item in items]
        expected_data = itertools.product(*items)
        expected = pd.DataFrame(data=expected_data, columns=input_cols)
        actual = blueprint.populate.deterministic(n=2)
        assert_frame_equal(expected, actual)
        self.assertEqual(len(actual), 128)

        # Test 2
        items = (
            [6, 8, 10],
            [8, 16, 24],
            [100., 175., 250.],
            ['AMTA', 'REPAL', 'NOSOR'],
            [('L', 'L', 'L'), ('L', 'R', 'R'), ('R', 'R', 'R')],
            [('hh', 'bd', 'sn'), ('bd', 'hh', 'sn'), ('sn', 'bd', 'hh')],
            [False, True],
        )
        items = [sorted(item) for item in items]
        expected_data = itertools.product(*items)
        expected = pd.DataFrame(data=expected_data, columns=input_cols)
        actual = blueprint.populate.deterministic(n=3)
        assert_frame_equal(expected, actual)
        self.assertEqual(len(actual), 1458)

        # Test 3
        items = (
            [6, 8, 10],
            [8, 16, 24],
            [100., 137.5, 175., 212.5, 250.],
            ['AMTA', 'REPAL', 'NOSOR'],
            [('L', 'L', 'L'), ('L', 'L', 'R'), ('L', 'R', 'L'),
             ('R', 'L', 'L'), ('R', 'L', 'R'), ('R', 'R', 'R')],
            [('hh', 'bd', 'sn'), ('hh', 'sn', 'bd'), ('bd', 'hh', 'sn'),
             ('bd', 'sn', 'hh'), ('sn', 'hh', 'bd'), ('sn', 'bd', 'hh')],
            [False, True],
        )
        items = [sorted(item) for item in items]
        expected_data = itertools.product(*items)
        expected = pd.DataFrame(data=expected_data, columns=input_cols)
        n_dict = {
            'min_size': 3,
            'max_size': 3,
            'price': 5,
            'brand': 3,
            'sick_pattern': 6,
            'groove': 6,
            'is_loud': 2
        }
        actual = blueprint.populate.deterministic(n=n_dict)
        assert_frame_equal(expected, actual)
        self.assertEqual(len(actual), 9720)

    def test_random_population_creation(self):
        """
        Test if the randomly generated population is correcly being created.

        Tests
        -----
        1. N = 1000
        2. N = 20000
        """
        blueprint = Blueprint(self.blueprint_dict)

        def get_expected_data(n, seed):
            expected_data = {}
            np.random.seed(seed)
            expected_data['min_size'] = np.random.randint(6, 11, n, dtype=int)
            np.random.seed(seed)
            expected_data['max_size'] = np.random.randint(8, 25, n, dtype=int)
            np.random.seed(seed)
            expected_data['price'] = np.random.uniform(100, 250, n)
            np.random.seed(seed)
            expected_data['brand'] = np.random.choice(['AMTA', 'REPAL', 'NOSOR'], n)
            np.random.seed(seed)
            expected_data['sick_pattern'] = [np.random.choice(['L', 'R'], 3) for _ in range(n)]
            np.random.seed(seed)
            expected_data['groove'] = [np.random.choice(['hh', 'bd', 'sn'], 3, replace=False)
                                       for _ in range(n)]
            np.random.seed(seed)
            expected_data['is_loud'] = np.random.randint(0, 2, n, dtype=bool)
            return pd.DataFrame(expected_data)

        # Test 1
        expected = get_expected_data(1000, 42)
        actual = blueprint.populate.random(n=1000, seed=42)
        assert_frame_equal(expected, actual)
        self.assertEqual(len(actual), 1000)

        # Test 2
        expected = get_expected_data(20000, 123)
        actual = blueprint.populate.random(n=20000, seed=123)
        assert_frame_equal(expected, actual)
        self.assertEqual(len(actual), 20000)

    def test_constraint_trimming(self):
        """
        Test adding user-defined constraints to the population creator.

        Tests
        -----
        1. Constraint `dict`
        2. Constraint enforcing

        Constraints
        -----------
        1. Variable 'min_size' is always less than 'max_size'
        2. Sum of the variables 'min_size' and 'max_size' is never greater than 32
        3. Product of the variables 'min_size' and 'max_size' is never greater
           than 180 if the 'brand' is 'NOSOR', else the same product operation
           is always greater than 100
        4. If is_loud then price > 200
        """
        blueprint = Blueprint(self.blueprint_dict)
        data = blueprint.populate.deterministic(3)

        # Constraint 1
        f = data['min_size'] < data['max_size']
        expected = data[f]
        def constraint(min_size, max_size): return min_size < max_size
        blueprint.add_constraint(constraint)
        actual = blueprint.populate.deterministic(3)
        assert_frame_equal(expected, actual)

        # Constraint 2
        f = f & ((data['min_size'] + data['max_size']) <= 32)
        expected = data[f]
        def constraint(min_size, max_size): return (min_size + max_size) <= 32
        blueprint.add_constraint(constraint)
        actual = blueprint.populate.deterministic(3)
        assert_frame_equal(expected, actual)

        # Constraint 3
        fc3 = (
            ((data['brand'] == 'NOSOR') & ((data['min_size'] * data['max_size']) <= 180))
            | ((data['brand'] != 'NOSOR') & ((data['min_size'] * data['max_size']) <= 100))
        )
        f = f & fc3
        expected = data[f]

        def constraint(min_size, max_size, brand):
            if brand == 'NOSOR':
                flag = (min_size * max_size) <= 180
            else:
                flag = (min_size * max_size) <= 100
            return flag
        blueprint.add_constraint(constraint)
        actual = blueprint.populate.deterministic(3)
        assert_frame_equal(expected, actual)

        # Constraint 4
        f = f & ~(data['is_loud'] & (data['price'] <= 200))
        expected = data[f]
        def constraint(is_loud, price): return price > 200 if is_loud else True
        blueprint.add_constraint(constraint)
        actual = blueprint.populate.deterministic(3)
        assert_frame_equal(expected, actual)
