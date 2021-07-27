import unittest
import itertools
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
                'type': 'numeric',
                'domain': int,
                'range': [6, 10]},
            'max_size': {
                'type': 'numeric',
                'domain': int,
                'range': [8, 24]},
            'price': {
                'type': 'numeric',
                'domain': float,
                'range': [100, 250]},
            'brand': {
                'type': 'categorical',
                'domain': ['AMTA', 'REPAL', 'NOSOR']},
            'sick_pattern': {
                'type': 'vectorial',
                'domain': ['L', 'R'],
                'replacement': True,
                'length': 3},
            'groove': {
                'type': 'vectorial',
                'domain': ['hh', 'bd', 'sn'],
                'replacement': False,
                'length': 3},
            'is_loud': {
                'type': 'boolean'}
        }

    def test_linearly_spaced_distribution(self):
        """
        Test the distribution of linearly spaced values.

        Tests
        -----
        1.  Numeric `int` values
        2.  Numeric `int` values with less available values than the number of
            subdivisions (n)
        3.  Numeric `float` values
        4.  Categorical values
        5.  Categorical values with less available values than the number of
            subdivisions (n)
        6.  Vectorial values with replacement
        7.  Vectorial values with replacement with less available values than
            the number of subdivisions (n)
        8.  Vectorial values without replacement
        9.  Vectorial values without replacement with less available values than
            the number of subdivisions (n)
        10. Boolean values
        """
        blueprint = Blueprint(self.blueprint_dict)

        # Test 1 (n=4)
        expected = (6, 7, 8, 10)
        actual_array = blueprint.genes.min_size.get_linspace(4)
        actual = tuple(actual_array)
        self.assertTupleEqual(expected, actual)

        # Test 2 (n=7)
        expected = (6, 7, 8, 9, 10)
        actual_array = blueprint.genes.min_size.get_linspace(7)
        actual = tuple(actual_array)
        self.assertTupleEqual(expected, actual)

        # Test 3 (n=7)
        expected = (100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0)
        actual_array = blueprint.genes.price.get_linspace(7)
        actual = tuple(actual_array)
        self.assertTupleEqual(expected, actual)

        # Test 4 (n=2)
        expected = ('AMTA', 'NOSOR')
        actual_array = blueprint.genes.brand.get_linspace(2)
        actual = tuple(actual_array)
        self.assertTupleEqual(expected, actual)

        # Test 5 (n=4)
        expected = ('AMTA', 'REPAL', 'NOSOR')
        actual_array = blueprint.genes.brand.get_linspace(4)
        actual = tuple(actual_array)
        self.assertTupleEqual(expected, actual)

        # Test 6 (n=3)
        expected = [
            ['L', 'L', 'L'],
            ['L', 'R', 'R'],
            ['R', 'R', 'R']
        ]
        actual = blueprint.genes.sick_pattern.get_linspace(3).tolist()
        self.assertListEqual(expected, actual)

        # Test 7 (n=10)
        expected = [
            ['L', 'L', 'L'],
            ['L', 'L', 'R'],
            ['L', 'R', 'L'],
            ['L', 'R', 'R'],
            ['R', 'L', 'L'],
            ['R', 'L', 'R'],
            ['R', 'R', 'L'],
            ['R', 'R', 'R']
        ]
        actual = blueprint.genes.sick_pattern.get_linspace(10).tolist()
        self.assertListEqual(expected, actual)

        # Test 8 (n=4)
        expected = [
            ['hh', 'bd', 'sn'],
            ['hh', 'sn', 'bd'],
            ['bd', 'sn', 'hh'],
            ['sn', 'bd', 'hh']
        ]
        actual = blueprint.genes.groove.get_linspace(4).tolist()
        self.assertListEqual(expected, actual)

        # Test 9 (n=10)
        expected = [
            ['hh', 'bd', 'sn'],
            ['hh', 'sn', 'bd'],
            ['bd', 'hh', 'sn'],
            ['bd', 'sn', 'hh'],
            ['sn', 'hh', 'bd'],
            ['sn', 'bd', 'hh']
        ]
        actual = blueprint.genes.groove.get_linspace(10).tolist()
        self.assertListEqual(expected, actual)

        # Test 10
        expected = [False, True]
        actual = blueprint.genes.is_loud.get_linspace().tolist()
        self.assertListEqual(expected, actual)

    def test_deterministic_population_creation(self):
        """
        Test if the deterministic combination of all sets of elements is being
        correctly generated.

        Tests
        -----
        1. N = 2 (2**7 = 128 combinations)
        2. N = 3 (3**6 * 2 = 1458 combinations)
        3. N = [3, 3, 5, 3, 6, 6, 2] (9720 combinations)
        """
        blueprint = Blueprint(self.blueprint_dict)
        input_cols = list(self.blueprint_dict.keys())

        # Test 1
        items = (
            [6, 10],
            [8, 24],
            [100., 250.],
            ['AMTA', 'NOSOR'],
            [('L', 'L', 'L'), ('R', 'R', 'R')],
            [('hh', 'bd', 'sn'), ('sn', 'bd', 'hh')],
            [False, True],
        )
        expected_data = itertools.product(*items)
        expected = pd.DataFrame(data=expected_data, columns=input_cols)
        actual = blueprint.get_pop_data.deterministic(n=2)
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
        expected_data = itertools.product(*items)
        expected = pd.DataFrame(data=expected_data, columns=input_cols)
        actual = blueprint.get_pop_data.deterministic(n=3)
        assert_frame_equal(expected, actual)
        self.assertEqual(len(actual), 1458)

        # Test 3

    def test_random_population_creation(self):
        """
        """

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
        """
        blueprint = Blueprint(self.blueprint_dict)
        # Constraint 1
        def minsize_lt_maxsize(min_size, max_size): min_size < max_size
        blueprint.add_constraint(name='minsize_lt_maxsize',
                                 func=minsize_lt_maxsize)
        # Testing constraints dict
        expected = {
            'minsize_lt_maxsize': minsize_lt_maxsize
        }
        actual = blueprint._constraints
        self.assertDictEqual(expected, actual)
        # Testing constraint application
