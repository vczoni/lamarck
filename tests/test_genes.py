import unittest
import numpy as np
from numpy.testing import assert_array_equal

from lamarck.genes.gene import (IntegerGene, FloatGene,
                                CategoricalGene, BooleanGene,
                                ArrayGene, SetGene)


class TestGene(unittest.TestCase):
    """
    Test cases for the Genes classes and their diversified methods of generating
    random and linearly spaced distributions.
    """

    def test_IntegerGene(self):
        """
        Test IntegerGene
        """
        int_gene = IntegerGene('int_gene', (0, 10))

        # Test 1: get_linspace
        expected = np.array([0, 2, 4, 6, 8, 10])
        actual = int_gene.get_linspace(6)
        assert_array_equal(expected, actual)

        # Test 2: get_random
        expected_range = list(range(0, 11))
        actual_vals = int_gene.get_random(100)
        self.assertIsInstance(actual_vals, np.ndarray)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)
        # (test reproduce.asexual, which is the same but generates tuples instead of numpy
        # arrays)
        actual_vals = int_gene.reproduce.asexual(100)
        self.assertIsInstance(actual_vals, tuple)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)

        # Test 3: reproduce.sexual
        n_trials = 100
        # (1 children)
        expected_range = [
            (2,), (6,),
        ]
        for _ in range(n_trials):
            actual = int_gene.reproduce.sexual(parent_genes=(6, 2), n_children=1)
            self.assertIn(actual, expected_range)
        # (2 children)
        expected_range = [
            (2, 2), (2, 6),
            (6, 2), (6, 6)
        ]
        for _ in range(n_trials):
            actual = int_gene.reproduce.sexual(parent_genes=(6, 2), n_children=2)
            self.assertIn(actual, expected_range)
        # (3 children)
        expected_range = [
            (2, 2, 2), (2, 2, 6),
            (2, 6, 2), (2, 6, 6),
            (6, 2, 2), (6, 2, 6),
            (6, 6, 2), (6, 6, 6),
        ]
        for _ in range(n_trials):
            actual = int_gene.reproduce.sexual(parent_genes=(6, 2), n_children=3)
            self.assertIn(actual, expected_range)

    def test_FloatGene(self):
        """
        Test FloatGene
        """
        float_gene = FloatGene('float_gene', (0, 10))

        # Test 1: get_linspace
        expected = np.array([0., 2., 4., 6., 8., 10.])
        actual = float_gene.get_linspace(6)
        assert_array_equal(expected, actual)

        # Test 2: get_random
        expected_range = (0, 11)
        actual_vals = float_gene.get_random(100)
        self.assertIsInstance(actual_vals, np.ndarray)
        for actual in actual_vals:
            self.assertGreater(actual, expected_range[0])
            self.assertLess(actual, expected_range[1])
        # (test reproduce.asexual, which is the same but generates tuples instead of numpy
        # arrays)
        actual_vals = float_gene.reproduce.asexual(100)
        self.assertIsInstance(actual_vals, tuple)
        for actual in actual_vals:
            self.assertGreater(actual, expected_range[0])
            self.assertLess(actual, expected_range[1])

        # Test 3: reproduce.sexual
        n_trials = 100
        # (1 children)
        expected_range = [
            (2.,), (6.,),
        ]
        for _ in range(n_trials):
            actual = float_gene.reproduce.sexual(parent_genes=(6., 2.), n_children=1)
            self.assertIn(actual, expected_range)
        # (2 children)
        expected_range = [
            (2., 2.), (2., 6.),
            (6., 2.), (6., 6.)
        ]
        for _ in range(n_trials):
            actual = float_gene.reproduce.sexual(parent_genes=(6., 2.), n_children=2)
            self.assertIn(actual, expected_range)
        # (3 children)
        expected_range = [
            (2., 2., 2.), (2., 2., 6.),
            (2., 6., 2.), (2., 6., 6.),
            (6., 2., 2.), (6., 2., 6.),
            (6., 6., 2.), (6., 6., 6.),
        ]
        for _ in range(n_trials):
            actual = float_gene.reproduce.sexual(parent_genes=(6., 2.), n_children=3)
            self.assertIn(actual, expected_range)

    def test_CategoricalGene(self):
        """
        Test CategoricalGene
        """
        cat_gene = CategoricalGene('cat_gene', ('A', 'B', 'C'))

        # Test 1: get_linspace
        # 3 genes
        expected = np.array(['A', 'B', 'C'])
        actual = cat_gene.get_linspace(3)
        assert_array_equal(expected, actual)
        # 5 genes
        expected = np.array(['A', 'A', 'B', 'B', 'C'])
        actual = cat_gene.get_linspace(5)
        assert_array_equal(expected, actual)
        # 6 genes
        expected = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        actual = cat_gene.get_linspace(6)
        assert_array_equal(expected, actual)
        # 6 genes
        expected = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        actual = cat_gene.get_linspace(9)
        assert_array_equal(expected, actual)

        # Test 2: get_random
        expected_range = ('A', 'B', 'C')
        actual_vals = cat_gene.get_random(100)
        self.assertIsInstance(actual_vals, np.ndarray)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)
        # (test reproduce.asexual, which is the same but generates tuples instead of numpy
        # arrays)
        actual_vals = cat_gene.reproduce.asexual(100)
        self.assertIsInstance(actual_vals, tuple)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)

        # Test 3: reproduce.sexual
        n_trials = 100
        # (1 children)
        expected_range = [
            ('A',), ('B',),
        ]
        for _ in range(n_trials):
            actual = cat_gene.reproduce.sexual(parent_genes=('A', 'B'), n_children=1)
            self.assertIn(actual, expected_range)
        # (2 children)
        expected_range = [
            ('A', 'A'), ('A', 'B'),
            ('B', 'A'), ('B', 'B')
        ]
        for _ in range(n_trials):
            actual = cat_gene.reproduce.sexual(parent_genes=('A', 'B'), n_children=2)
            self.assertIn(actual, expected_range)
        # (3 children)
        expected_range = [
            ('A', 'A', 'A'), ('A', 'A', 'B'),
            ('A', 'B', 'A'), ('A', 'B', 'B'),
            ('B', 'A', 'A'), ('B', 'A', 'B'),
            ('B', 'B', 'A'), ('B', 'B', 'B'),
        ]
        for _ in range(n_trials):
            actual = cat_gene.reproduce.sexual(parent_genes=('B', 'A'), n_children=3)
            self.assertIn(actual, expected_range)

    def test_BooleanGene(self):
        """
        Test BooleanGene
        """
        bool_gene = BooleanGene('bool_gene')

        # Test 1: get_linspace
        # 2 genes
        expected = np.array([False, True])
        actual = bool_gene.get_linspace(2)
        assert_array_equal(expected, actual)
        # 3 genes
        expected = np.array([False, False, True])
        actual = bool_gene.get_linspace(3)
        assert_array_equal(expected, actual)
        # 5 genes
        expected = np.array([False, False, False, True, True])
        actual = bool_gene.get_linspace(5)
        assert_array_equal(expected, actual)
        # 6 genes
        expected = np.array([False, False, False, True, True, True])
        actual = bool_gene.get_linspace(6)
        assert_array_equal(expected, actual)
        # 6 genes
        expected = np.array([False, False, False, False, False, True, True, True, True])
        actual = bool_gene.get_linspace(9)
        assert_array_equal(expected, actual)

        # Test 2: get_random
        expected_range = (False, True)
        actual_vals = bool_gene.get_random(100)
        self.assertIsInstance(actual_vals, np.ndarray)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)
        # (test reproduce.asexual, which is the same but generates tuples instead of numpy
        # arrays)
        actual_vals = bool_gene.reproduce.asexual(100)
        self.assertIsInstance(actual_vals, tuple)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)

        # Test 3: reproduce.sexual
        n_trials = 100
        # (1 children)
        expected_range = [
            (True,), (False,),
        ]
        for _ in range(n_trials):
            actual = bool_gene.reproduce.sexual(parent_genes=(False, True), n_children=1)
            self.assertIn(actual, expected_range)
        # (2 children)
        expected_range = [
            (False, False), (False, True),
            (True, False), (True, True)
        ]
        for _ in range(n_trials):
            actual = bool_gene.reproduce.sexual(parent_genes=(False, True), n_children=2)
            self.assertIn(actual, expected_range)
        # (3 children)
        expected_range = [
            (False, False, False), (False, False, True),
            (False, True, False), (False, True, True),
            (True, False, False), (True, False, True),
            (True, True, False), (True, True, True),
        ]
        for _ in range(n_trials):
            actual = bool_gene.reproduce.sexual(parent_genes=(True, False), n_children=3)
            self.assertIn(actual, expected_range)

    def test_ArrayGene(self):
        """
        Test ArrayGene
        """
        array_gene = ArrayGene('array_gene', ('A', 'B', 'C'), length=4)

        # Test 1: get_linspace
        # 2 genes
        expected_vals = [
            ('A', 'A', 'A', 'A'),
            ('C', 'C', 'C', 'C'),
        ]
        expected = np.empty(len(expected_vals), dtype=object)
        expected[:] = expected_vals
        actual = array_gene.get_linspace(2)
        assert_array_equal(expected, actual)
        # 3 genes
        expected_vals = [
            ('A', 'A', 'A', 'A'),
            ('B', 'B', 'B', 'B'),
            ('C', 'C', 'C', 'C'),
        ]
        expected = np.empty(len(expected_vals), dtype=object)
        expected[:] = expected_vals
        actual = array_gene.get_linspace(3)
        assert_array_equal(expected, actual)

        # Test 2: get_random
        expected_range = (
            ('A', 'A', 'A', 'A'),
            ('A', 'A', 'A', 'B'),
            ('A', 'A', 'A', 'C'),
            ('A', 'A', 'B', 'A'),
            ('A', 'A', 'B', 'B'),
            ('A', 'A', 'B', 'C'),
            ('A', 'A', 'C', 'A'),
            ('A', 'A', 'C', 'B'),
            ('A', 'A', 'C', 'C'),
            ('A', 'B', 'A', 'A'),
            ('A', 'B', 'A', 'B'),
            ('A', 'B', 'A', 'C'),
            ('A', 'B', 'B', 'A'),
            ('A', 'B', 'B', 'B'),
            ('A', 'B', 'B', 'C'),
            ('A', 'B', 'C', 'A'),
            ('A', 'B', 'C', 'B'),
            ('A', 'B', 'C', 'C'),
            ('A', 'C', 'A', 'A'),
            ('A', 'C', 'A', 'B'),
            ('A', 'C', 'A', 'C'),
            ('A', 'C', 'B', 'A'),
            ('A', 'C', 'B', 'B'),
            ('A', 'C', 'B', 'C'),
            ('A', 'C', 'C', 'A'),
            ('A', 'C', 'C', 'B'),
            ('A', 'C', 'C', 'C'),
            ('B', 'A', 'A', 'A'),
            ('B', 'A', 'A', 'B'),
            ('B', 'A', 'A', 'C'),
            ('B', 'A', 'B', 'A'),
            ('B', 'A', 'B', 'B'),
            ('B', 'A', 'B', 'C'),
            ('B', 'A', 'C', 'A'),
            ('B', 'A', 'C', 'B'),
            ('B', 'A', 'C', 'C'),
            ('B', 'B', 'A', 'A'),
            ('B', 'B', 'A', 'B'),
            ('B', 'B', 'A', 'C'),
            ('B', 'B', 'B', 'A'),
            ('B', 'B', 'B', 'B'),
            ('B', 'B', 'B', 'C'),
            ('B', 'B', 'C', 'A'),
            ('B', 'B', 'C', 'B'),
            ('B', 'B', 'C', 'C'),
            ('B', 'C', 'A', 'A'),
            ('B', 'C', 'A', 'B'),
            ('B', 'C', 'A', 'C'),
            ('B', 'C', 'B', 'A'),
            ('B', 'C', 'B', 'B'),
            ('B', 'C', 'B', 'C'),
            ('B', 'C', 'C', 'A'),
            ('B', 'C', 'C', 'B'),
            ('B', 'C', 'C', 'C'),
            ('C', 'A', 'A', 'A'),
            ('C', 'A', 'A', 'B'),
            ('C', 'A', 'A', 'C'),
            ('C', 'A', 'B', 'A'),
            ('C', 'A', 'B', 'B'),
            ('C', 'A', 'B', 'C'),
            ('C', 'A', 'C', 'A'),
            ('C', 'A', 'C', 'B'),
            ('C', 'A', 'C', 'C'),
            ('C', 'B', 'A', 'A'),
            ('C', 'B', 'A', 'B'),
            ('C', 'B', 'A', 'C'),
            ('C', 'B', 'B', 'A'),
            ('C', 'B', 'B', 'B'),
            ('C', 'B', 'B', 'C'),
            ('C', 'B', 'C', 'A'),
            ('C', 'B', 'C', 'B'),
            ('C', 'B', 'C', 'C'),
            ('C', 'C', 'A', 'A'),
            ('C', 'C', 'A', 'B'),
            ('C', 'C', 'A', 'C'),
            ('C', 'C', 'B', 'A'),
            ('C', 'C', 'B', 'B'),
            ('C', 'C', 'B', 'C'),
            ('C', 'C', 'C', 'A'),
            ('C', 'C', 'C', 'B'),
            ('C', 'C', 'C', 'C'),
        )
        actual_vals = array_gene.get_random(100)
        self.assertIsInstance(actual_vals, np.ndarray)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)
        # (test reproduce.asexual, which is the same but generates tuples instead of numpy
        # arrays)
        actual_vals = array_gene.reproduce.asexual(100)
        self.assertIsInstance(actual_vals, tuple)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)

        # Test 3: reproduce.sexual
        n_trials = 100
        parent_genes = (('A', 'B'), ('B', 'A'))
        # (1 children)
        expected_range = [
            (('A', 'A'),), (('A', 'B'),),
            (('B', 'A'),), (('B', 'B'),),
        ]
        for _ in range(n_trials):
            actual = array_gene.reproduce.sexual(parent_genes=parent_genes, n_children=1)
            self.assertIn(actual, expected_range)
        # (2 children)
        expected_range = [
            (('A', 'A'), ('A', 'A')),
            (('A', 'A'), ('A', 'B')),
            (('A', 'A'), ('B', 'A')),
            (('A', 'A'), ('B', 'B')),
            (('A', 'B'), ('A', 'A')),
            (('A', 'B'), ('A', 'B')),
            (('A', 'B'), ('B', 'A')),
            (('A', 'B'), ('B', 'B')),
            (('B', 'A'), ('A', 'A')),
            (('B', 'A'), ('A', 'B')),
            (('B', 'A'), ('B', 'A')),
            (('B', 'A'), ('B', 'B')),
            (('B', 'B'), ('A', 'A')),
            (('B', 'B'), ('A', 'B')),
            (('B', 'B'), ('B', 'A')),
            (('B', 'B'), ('B', 'B')),
        ]
        for _ in range(n_trials):
            actual = array_gene.reproduce.sexual(parent_genes=parent_genes, n_children=2)
            self.assertIn(actual, expected_range)

    def test_SetGene(self):
        """
        Test SetGene
        """
        set_gene = SetGene('set_gene', ('A', 'B', 'C'), length=2)

        # Test 1: get_linspace
        # 2 genes
        expected_vals = [
            ('A', 'B'),
            ('C', 'B'),
        ]
        expected = np.empty(len(expected_vals), dtype=object)
        expected[:] = expected_vals
        actual = set_gene.get_linspace(2)
        assert_array_equal(expected, actual)
        # 3 genes
        expected_vals = [
            ('A', 'B'),
            ('B', 'A'),
            ('C', 'B'),
        ]
        expected = np.empty(len(expected_vals), dtype=object)
        expected[:] = expected_vals
        actual = set_gene.get_linspace(3)
        assert_array_equal(expected, actual)

        # Test 2: get_random
        expected_range = (
            ('A', 'B'), ('A', 'C'),
            ('B', 'A'), ('B', 'C'),
            ('C', 'A'), ('C', 'B'),
        )
        actual_vals = set_gene.get_random(100)
        self.assertIsInstance(actual_vals, np.ndarray)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)
        # (test reproduce.asexual, which is the same but generates tuples instead of numpy
        # arrays)
        actual_vals = set_gene.reproduce.asexual(100)
        self.assertIsInstance(actual_vals, tuple)
        for actual in actual_vals:
            self.assertIn(actual, expected_range)

        # Test 3: reproduce.sexual
        n_trials = 100
        parent_genes = (('A', 'B'), ('B', 'A'))
        # (1 children)
        expected_range = [
            (('A', 'B'),),
            (('B', 'A'),)
        ]
        for _ in range(n_trials):
            actual = set_gene.reproduce.sexual(parent_genes=parent_genes, n_children=1)
            self.assertIn(actual, expected_range)
        # (2 children)
        expected_range = [
            (('A', 'B'), ('A', 'B')),
            (('A', 'B'), ('B', 'A')),
            (('B', 'A'), ('A', 'B')),
            (('B', 'A'), ('B', 'A')),
        ]
        for _ in range(n_trials):
            actual = set_gene.reproduce.sexual(parent_genes=parent_genes, n_children=2)
            self.assertIn(actual, expected_range)
