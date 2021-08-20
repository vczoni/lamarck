import unittest

from lamarck.sexual import sexual_reproduction


class TestSexualReproduction(unittest.TestCase):
    """
    Sexual Reproduciton Tests

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
    *no replacement (set)
    """

    def test_numeric_reproduction_methods(self):
        """
        Assert that the sexual reproduction method behave correctly according to the type of the
        gene it's mixing.

        Tests
        -----
        1. Mix numeric genes (2 parents, 2 children)
        2. Mix numeric genes (3 parents, 2 children)
        """
        # Test 1
        parent_genes = (7, 1)
        expected = [
            (1, 1), (1, 7),
            (7, 1), (7, 7)
        ]
        actual = sexual_reproduction.scalar_mix(parent_genes)
        self.assertIn(actual, expected)

        # Test 2
        parent_genes = (8, 5, 2)
        expected = [
            (8, 8), (8, 5), (8, 2),
            (5, 8), (5, 5), (5, 2),
            (2, 8), (2, 5), (2, 2)
        ]
        actual = sexual_reproduction.scalar_mix(parent_genes)
        self.assertIn(actual, expected)

    def test_categorical_reproduction_methods(self):
        """
        Assert that the sexual reproduction method behave correctly according to the type of the
        gene it's mixing.

        Tests
        -----
        1. Mix categorical genes (2 parents, 3 children)
        """
        n_tries = 1000

        # Test 1
        parent_genes = ('A', 'C')
        expected = [
            ('A', 'A', 'A'), ('A', 'A', 'C'),
            ('A', 'C', 'A'), ('A', 'C', 'C'),
            ('C', 'A', 'A'), ('C', 'A', 'C'),
            ('C', 'C', 'A'), ('C', 'C', 'C')
        ]
        for _ in range(n_tries):
            actual = sexual_reproduction.scalar_mix(parent_genes, n_children=3)
            self.assertIn(actual, expected)

    def test_boolean_reproduction_methods(self):
        """
        Assert that the sexual reproduction method behave correctly according to the type of the
        gene it's mixing.

        Tests
        -----
        1. Mix boolean genes (2 parents, 2 children)
        2. Mix boolean genes (3 parents, 2 children)
        3. Mix boolean genes (3 parents, 3 children)
        """
        n_tries = 1000

        # Test 1
        parent_genes = (True, False)
        expected = [
            (True, True), (True, False),
            (False, True), (False, False)
        ]
        for _ in range(n_tries):
            actual = sexual_reproduction.scalar_mix(parent_genes)
            self.assertIn(actual, expected)

        # Test 2
        parent_genes = (True, False, False)
        expected = [
            (True, True), (True, False),
            (False, True), (False, False)
        ]
        for _ in range(n_tries):
            actual = sexual_reproduction.scalar_mix(parent_genes)
            self.assertIn(actual, expected)

        # Test 3
        parent_genes = (True, False, False)
        expected = [
            (True, True, True), (True, True, False),
            (True, False, True), (True, False, False),
            (False, True, True), (False, True, False),
            (False, False, True), (False, False, False),
        ]
        for _ in range(n_tries):
            actual = sexual_reproduction.scalar_mix(parent_genes, n_children=3)
            self.assertIn(actual, expected)

    def test_vectorial_reproduction_methods(self):
        """
        Assert that the sexual reproduction method behave correctly according to the type of the
        gene it's mixing.

        Tests
        -----
        1. Mix vectorial genes (2 parents, 2 children)
        2. Mix vectorial genes (3 parents, 2 children)
        3. Cross vectorial genes (2 parents, 2 children)
        4. Cross vectorial genes (3 parents, 2 children)
        """
        n_tries = 1000

        # Test 1
        parent_genes = (
            (1, 0, 2),
            (2, 1, 1)
        )
        expected = [
            (1, 0, 2), (1, 0, 1), (1, 1, 2), (1, 1, 1),
            (2, 0, 2), (2, 0, 1), (2, 1, 2), (2, 1, 1)
        ]
        for _ in range(n_tries):
            actuals = sexual_reproduction.vectorial_mix(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 2
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
            actuals = sexual_reproduction.vectorial_mix(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 3
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
            actuals = sexual_reproduction.vectorial_cross(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 4
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
            actuals = sexual_reproduction.vectorial_cross(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

    def test_vectorial_reproduction_without_replacement_methods(self):
        """
        Assert that the sexual reproduction method behave correctly according to the type of the
        gene it's mixing.

        Tests
        -----
        1. Cross vectorial genes without replacement (2 parents, 2 children)
        2. Cross vectorial genes without replacement (3 parents, 2 children)
        """
        n_tries = 1000

        # Test 1
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
            actuals = sexual_reproduction.vectorial_cross_unique(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)

        # Test 2
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
            actuals = sexual_reproduction.vectorial_cross_unique(parent_genes, n_children=2)
            for actual in actuals:
                self.assertIn(actual, expected)
