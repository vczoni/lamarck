import unittest
import numpy as np
from numpy.testing import assert_array_equal

from lamarck.gene import GeneCollection, sexual_gene_reproductor


class TestGene(unittest.TestCase):
    """
    Test cases for the Genes classes and their diversified methods of generating
    random and linearly spaced distributions.
    """
    genes: GeneCollection

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
        10. Boolean values
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
        expected = np.array((6, 7, 8, 9, 10))
        actual = self.genes.min_size.get_linspace(7)
        assert_array_equal(expected, actual)

        # Test 3 (n=7)
        expected = np.array((100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0))
        actual = self.genes.price.get_linspace(7)
        assert_array_equal(expected, actual)

        # Test 4 (n=2)
        expected = np.array(('AMTA', 'NOSOR'))
        actual = self.genes.brand.get_linspace(2)
        assert_array_equal(expected, actual)

        # Test 5 (n=4)
        expected = np.array(('AMTA', 'REPAL', 'NOSOR'))
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
            ('L', 'L', 'L'),
            ('L', 'L', 'R'),
            ('L', 'R', 'L'),
            ('L', 'R', 'R'),
            ('R', 'L', 'L'),
            ('R', 'L', 'R'),
            ('R', 'R', 'L'),
            ('R', 'R', 'R')
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
            ('hh', 'bd', 'sn'),
            ('hh', 'sn', 'bd'),
            ('bd', 'hh', 'sn'),
            ('bd', 'sn', 'hh'),
            ('sn', 'hh', 'bd'),
            ('sn', 'bd', 'hh')
        ])
        actual = self.genes.groove.get_linspace(10)
        assert_array_equal(expected, actual)

        # Test 10
        expected = np.array([False, True])
        actual = self.genes.is_loud.get_linspace()
        assert_array_equal(expected, actual)

    def test_gene_reproduction_methods(self):
        """
        Assert that the sexual reproduction method behave correctly according to the type of the
        gene it's mixing.

        Cross-Over Illustration
        -----------------------
        - Parents genes (example)
        ┌──────────┬───┬───┬───┬─────────┬────────────┬─────┐
        │Parents   │ u │ v │ w │    x    │     y      │  z  │
        ├──────────┼───┼───┼───┼─────────┼────────────┼─────┤
        │Parent 1  │ 7 │ 5 │ A │(a, b, c)│(1, 2, 3, 4)│True │
        ├──────────┼───┼───┼───┼─────────┼────────────┼─────┤
        │Parent 2  │ 1 │ 8 │ C │(a, d, e)│(4, 1, 3, 2)│False│
        └──────────┴───┴───┴───┴─────────┴────────────┴─────┘

        - Childrens genes (example)
        ┌──────────┬───┬───┬───┬─────────┬────────────┬─────┐
        │Childrens │ u │ v │ w │    x    │     y*     │  z  │
        ├──────────┼───┼───┼───┼─────────┼────────────┼─────┤
        │Children 1│ 7 │ 8 │ A │(a, d, c)│(1, 2, 4, 3)│True │
        ├──────────┼───┼───┼───┼─────────┼────────────┼─────┤
        │Children 2│ 1 │ 5 │ C │(a, b, e)│(4, 1, 2, 3)│False│
        └──────────┴───┴───┴───┴─────────┴────────────┴─────┘
        *no replacement

        Tests
        -----
        1.  Mix numeric genes (2 parents, 2 children)
        2.  Mix numeric genes (3 parents, 2 children)
        3.  Mix categorical genes (2 parents, 3 children)
        4.  Mix boolean genes (2 parents, 2 children)
        5.  Mix boolean genes (3 parents, 2 children)
        6.  Mix boolean genes (3 parents, 3 children)
        7.  Mix vectorial genes (2 parents, 2 children)
        8.  Mix vectorial genes (3 parents, 2 children)
        9.  Cross vectorial genes (2 parents, 2 children)
        10. Cross vectorial genes (3 parents, 2 children)
        11. Cross vectorial genes without replacement (2 parents, 2 children)
        12. Cross vectorial genes without replacement (3 parents, 2 children)
        """
        n_tries = 1000

        # Test 1
        parent_genes = (7, 1)
        expected = [
            (1, 1), (1, 7),
            (7, 1), (7, 7)
        ]
        actual = sexual_gene_reproductor.scalar_mix(parent_genes)
        self.assertIn(actual, expected)

        # Test 2
        parent_genes = (8, 5, 2)
        expected = [
            (8, 8), (8, 5), (8, 2),
            (5, 8), (5, 5), (5, 2),
            (2, 8), (2, 5), (2, 2)
        ]
        actual = sexual_gene_reproductor.scalar_mix(parent_genes)
        self.assertIn(actual, expected)

        # Test 3
        parent_genes = ('A', 'C')
        expected = [
            ('A', 'A', 'A'), ('A', 'A', 'C'),
            ('A', 'C', 'A'), ('A', 'C', 'C'),
            ('C', 'A', 'A'), ('C', 'A', 'C'),
            ('C', 'C', 'A'), ('C', 'C', 'C')
        ]
        for _ in range(n_tries):
            actual = sexual_gene_reproductor.scalar_mix(parent_genes, n_children=3)
            self.assertIn(actual, expected)

        # Test 4
        parent_genes = (True, False)
        expected = [
            (True, True), (True, False),
            (False, True), (False, False)
        ]
        for _ in range(n_tries):
            actual = sexual_gene_reproductor.scalar_mix(parent_genes)
            self.assertIn(actual, expected)

        # Test 5
        parent_genes = (True, False, False)
        expected = [
            (True, True), (True, False),
            (False, True), (False, False)
        ]
        for _ in range(n_tries):
            actual = sexual_gene_reproductor.scalar_mix(parent_genes)
            self.assertIn(actual, expected)

        # Test 6
        parent_genes = (True, False, False)
        expected = [
            (True, True, True), (True, True, False),
            (True, False, True), (True, False, False),
            (False, True, True), (False, True, False),
            (False, False, True), (False, False, False),
        ]
        for _ in range(n_tries):
            actual = sexual_gene_reproductor.scalar_mix(parent_genes, n_children=3)
            self.assertIn(actual, expected)

        # Test 7
        parent_genes = (
            (1, 0, 2),
            (2, 1, 1)
        )
        expected = [
            (1, 0, 2), (1, 0, 1), (1, 1, 2), (1, 1, 1),
            (2, 0, 2), (2, 0, 1), (2, 1, 2), (2, 1, 1)
        ]
        for _ in range(n_tries):
            actuals = sexual_gene_reproductor.vectorial_mix(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 8
        parent_genes = (
            (1, 0, 2),
            (2, 1, 1),
            (7, 9, 6)
        )
        expected = [
            (1, 0, 2), (1, 0, 1), (1, 0, 6),
            (1, 1, 2), (1, 1, 1), (1, 1, 6),
            (1, 9, 2), (1, 9, 1), (1, 9, 6),
            (2, 0, 2), (2, 0, 1), (2, 0, 6),
            (2, 1, 2), (2, 1, 1), (2, 1, 6),
            (2, 9, 2), (2, 9, 1), (2, 9, 6),
            (7, 0, 2), (7, 0, 1), (7, 0, 6),
            (7, 1, 2), (7, 1, 1), (7, 1, 6),
            (7, 9, 2), (7, 9, 1), (7, 9, 6),
        ]
        for _ in range(n_tries):
            actuals = sexual_gene_reproductor.vectorial_mix(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 9
        parent_genes = (
            (1, 0, 2, 4, 3),
            (2, 3, 0, 1, 4)
        )
        expected = [
            (1, 0, 2, 4, 4),
            (1, 0, 2, 1, 4),
            (1, 0, 0, 1, 4),
            (1, 3, 0, 1, 4),
            (2, 3, 0, 1, 3),
            (2, 3, 0, 4, 3),
            (2, 3, 2, 4, 3),
            (2, 0, 2, 4, 3)
        ]
        for _ in range(n_tries):
            actuals = sexual_gene_reproductor.vectorial_cross(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 10
        parent_genes = (
            (1, 0, 2, 4, 3),
            (2, 3, 0, 1, 4),
            (0, 5, 6, 2, 1)
        )
        expected = [
            (1, 0, 2, 4, 4), (1, 0, 2, 1, 4), (1, 0, 0, 1, 4), (1, 3, 0, 1, 4),
            (1, 0, 2, 4, 1), (1, 0, 2, 2, 1), (1, 0, 6, 2, 1), (1, 5, 6, 2, 1),
            (2, 3, 0, 1, 3), (2, 3, 0, 4, 3), (2, 3, 2, 4, 3), (2, 0, 2, 4, 3),
            (2, 3, 0, 1, 1), (2, 3, 0, 2, 1), (2, 3, 6, 2, 1), (2, 5, 6, 2, 1),
            (0, 5, 6, 2, 3), (0, 5, 6, 4, 3), (0, 5, 2, 4, 3), (0, 0, 2, 4, 3),
            (0, 5, 6, 2, 4), (0, 5, 6, 1, 4), (0, 5, 0, 1, 4), (0, 3, 0, 1, 4)
        ]
        for _ in range(n_tries):
            actuals = sexual_gene_reproductor.vectorial_cross(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 11
        parent_genes = (
            (1, 0, 2, 4, 3),
            (2, 3, 0, 1, 4)
        )
        expected = [
            (1, 0, 2, 4, 3),
            (1, 0, 2, 3, 4),
            (1, 0, 2, 3, 4),
            (1, 2, 3, 0, 4),
            (2, 3, 0, 1, 4),
            (2, 3, 0, 1, 4),
            (2, 3, 1, 0, 4),
            (2, 1, 0, 4, 3)
        ]
        for _ in range(n_tries):
            actuals = sexual_gene_reproductor.vectorial_cross_unique(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 12
        parent_genes = (
            (1, 0, 2, 4, 3),
            (2, 3, 0, 1, 4),
            (0, 5, 6, 2, 1)
        )
        expected = [
            (1, 0, 2, 4, 3), (1, 0, 2, 3, 4), (1, 0, 2, 3, 4), (1, 2, 3, 0, 4),
            (1, 0, 2, 4, 5), (1, 0, 2, 5, 6), (1, 0, 5, 6, 2), (1, 0, 5, 6, 2),
            (2, 3, 0, 1, 4), (2, 3, 0, 1, 4), (2, 3, 1, 0, 4), (2, 1, 0, 4, 3),
            (2, 3, 0, 1, 5), (2, 3, 0, 5, 6), (2, 3, 0, 5, 6), (2, 0, 5, 6, 1),
            (0, 5, 6, 2, 1), (0, 5, 6, 1, 2), (0, 5, 1, 2, 4), (0, 1, 2, 4, 3),
            (0, 5, 6, 2, 3), (0, 5, 6, 2, 3), (0, 5, 2, 3, 1), (0, 2, 3, 1, 4)
        ]
        for _ in range(n_tries):
            actuals = sexual_gene_reproductor.vectorial_cross_unique(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)
