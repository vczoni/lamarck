import unittest
import numpy as np
from numpy.testing import assert_array_equal

from lamarck.genes import GeneCollection


class TestGene(unittest.TestCase):
    """
    Test cases for the Genes classes and their diversified methods of generating
    random and linearly spaced distributions.
    """
    genes: GeneCollection

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
        self.genes = GeneCollection(blueprint_dict)

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
        10. Boolean values (2)
        11. Boolean values (5)
        """
        def convert_to_array(ls):
            arr = np.empty(len(ls), dtype=object)
            arr[:] = ls
            return arr

        # Test 1 (n=4)
        expected = np.array((6, 7, 8, 10))
        actual = self.genes.min_size.get_linspace(4)
        assert_array_equal(expected, actual)

        # Test 2 (n=7)
        expected = np.array((6, 6, 7, 8, 8, 9, 10))
        actual = self.genes.min_size.get_linspace(7)
        assert_array_equal(expected, actual)

        # Test 3 (n=7)
        expected = np.array((100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0))
        actual = self.genes.price.get_linspace(7)
        assert_array_equal(expected, actual)

        # Test 4 (n=2)
        expected = np.array(('AMTA', 'REPAL'))
        actual = self.genes.brand.get_linspace(2)
        assert_array_equal(expected, actual)

        # Test 5 (n=4)
        expected = np.array(('AMTA', 'AMTA', 'REPAL', 'NOSOR'))
        actual = self.genes.brand.get_linspace(4)
        assert_array_equal(expected, actual)

        # Test 6 (n=3)
        expected = convert_to_array([
            ('L', 'L', 'L'),
            ('L', 'R', 'R'),
            ('R', 'R', 'R')
        ])
        actual = self.genes.sick_pattern.get_linspace(3)
        assert_array_equal(expected, actual)

        # Test 7 (n=10)
        expected = convert_to_array([
            ('L', 'L', 'L'), ('L', 'L', 'L'),
            ('L', 'L', 'R'), ('L', 'R', 'L'),
            ('L', 'R', 'R'), ('L', 'R', 'R'),
            ('R', 'L', 'L'), ('R', 'L', 'R'),
            ('R', 'R', 'L'), ('R', 'R', 'R')
        ])
        actual = self.genes.sick_pattern.get_linspace(10)
        assert_array_equal(expected, actual)

        # Test 8 (n=4)
        expected = convert_to_array([
            ('hh', 'bd', 'sn'),
            ('hh', 'sn', 'bd'),
            ('bd', 'sn', 'hh'),
            ('sn', 'bd', 'hh')
        ])
        actual = self.genes.groove.get_linspace(4)
        assert_array_equal(expected, actual)

        # Test 9 (n=10)
        expected = convert_to_array([
            ('hh', 'bd', 'sn'), ('hh', 'bd', 'sn'),
            ('hh', 'sn', 'bd'), ('hh', 'sn', 'bd'),
            ('bd', 'hh', 'sn'), ('bd', 'hh', 'sn'),
            ('bd', 'sn', 'hh'), ('bd', 'sn', 'hh'),
            ('sn', 'hh', 'bd'), ('sn', 'bd', 'hh')
        ])
        actual = self.genes.groove.get_linspace(10)
        assert_array_equal(expected, actual)

        # Test 10
        expected = np.array([False, True])
        actual = self.genes.is_loud.get_linspace(2)
        assert_array_equal(expected, actual)

        # Test 11
        expected = np.array([False, False, False, True, True])
        actual = self.genes.is_loud.get_linspace(5)
        assert_array_equal(expected, actual)
