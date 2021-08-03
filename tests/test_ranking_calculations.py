import unittest
import pandas as pd
from pandas.testing import assert_series_equal

from lamarck.rankcalculator import RankCalculator


class TestRankingMethods(unittest.TestCase):
    """
    Testing The Ranking methods and auxiliary data.

    Tests
    -----
    1. Single Objective Ranking
    2. Multi Objective Ranking - Ranked
    3. Multi Objective Ranking - Pareto
    """

    def test_single_objective_ranking(self):
        """
        Testing if the single criteria ranks the results according to its objective.

        Tests
        -----
        1. Ranking of results that must be 'maximum'
        2. Ranking of results that must be 'minimum'
        """
        results_data = {
            'x': [0, 0, 1, 10, 15, 2, 8],
            'y': [2, 40, 13, 2, 2, 40, 7],
        }
        results_df = pd.DataFrame(results_data)
        ranker = RankCalculator(results_df)

        # Test 1 (a) maximum for 'x'
        expected = pd.Series([6, 6, 5, 2, 1, 4, 3], name='Rank')
        ranker.update(out='x')
        actual = ranker.single('max')
        assert_series_equal(expected, actual)

        # Test 1 (b) maximum for 'y'
        expected = pd.Series([5, 1, 3, 5, 5, 1, 4], name='Rank')
        ranker.update(out='y')
        actual = ranker.single('max')
        assert_series_equal(expected, actual)

        # Test 2 (a) minimum for 'x'
        expected = pd.Series([1, 1, 3, 6, 7, 4, 5], name='Rank')
        ranker.update(out='x')
        actual = ranker.single('min')
        assert_series_equal(expected, actual)

        # Test 2 (b) minimum for 'y'
        expected = pd.Series([1, 6, 5, 1, 1, 6, 4], name='Rank')
        ranker.update(out='y')
        actual = ranker.single('min')
        assert_series_equal(expected, actual)

    def test_multi_objective_ranked_ranking(self):
        """
        Testing if the multi criteria (ranked) ranks the results according to its
        objectives.

        Tests
        -----
        1. Ranking of results of x, then y, that must be 'maximum'
        2. Ranking of results of y, then x, that must be 'maximum'
        3. Ranking of results of x, then y, that must be 'minimum'
        4. Ranking of results of y, then x, that must be 'minimum'
        5. Ranking of results of x, then y, where x must be maximum and y must be minimum
        6. Ranking of results of y, then x, where x must be maximum and y must be minimum
        """
        results_data = {
            'x': [-10, 10, 0, 1, 10, 15, 2, 8],
            'y': [2, 2, 40, 13, 2, 2, 40, 7]
        }
        results_df = pd.DataFrame(results_data)
        ranker = RankCalculator(results_df)

        # Test 1
        expected = pd.Series([8, 2, 7, 6, 2, 1, 5, 4], name='Rank')
        ranker.update(out=['x', 'y'])
        actual = ranker.ranked(['max', 'max'])
        assert_series_equal(expected, actual)

        # Test 2
        expected = pd.Series([8, 6, 2, 3, 6, 5, 1, 4], name='Rank')
        ranker.update(out=['y', 'x'])
        actual = ranker.ranked(['max', 'max'])
        assert_series_equal(expected, actual)

        # Test 3
        expected = pd.Series([1, 6, 2, 3, 6, 8, 4, 5], name='Rank')
        ranker.update(out=['x', 'y'])
        actual = ranker.ranked(['min', 'min'])
        assert_series_equal(expected, actual)

        # Test 4
        expected = pd.Series([1, 2, 7, 6, 2, 4, 8, 5], name='Rank')
        ranker.update(out=['y', 'x'])
        actual = ranker.ranked(['min', 'min'])
        assert_series_equal(expected, actual)

        # Test 5
        expected = pd.Series([8, 2, 7, 6, 2, 1, 5, 4], name='Rank')
        ranker.update(out=['x', 'y'])
        actual = ranker.ranked(['max', 'min'])
        assert_series_equal(expected, actual)

        # Test 6
        expected = pd.Series([4, 2, 8, 6, 2, 1, 7, 5], name='Rank')
        ranker.update(out=['y', 'x'])
        actual = ranker.ranked(['min', 'max'])
        assert_series_equal(expected, actual)

        # Test 7

    def test_multi_objective_pareto_fronts(self):
        """
        Testing if the multi criteria (pareto) `pareto_fronts` method is correctly
        generating the `front` values.

        Tests
        -----
        1. Max x then min y
        2. Min y then max x (result should be he same as 1)
        3. Min x then max y
        4. Max x then max y
        5. Min x then min y
        6. Max z then min w
        6. Max x then min y then min z then max w
        """
        results_data = {
            'x': [-10, 8, 0, 1, 10, 15, 2, 8, 15],
            'y': [2, 2, 40, 13, 2, 2, 40, 7, 2],
            'z': [10, 100, 200, 35, 350, 700, 60, 600, 1200],
            'w': [9, 0, -10, 34, 25, 15, 59, 50, 40]
        }
        results_df = pd.DataFrame(results_data)
        ranker = RankCalculator(results_df)

        # Test 1 (max x then min y)
        expected = pd.Series([4, 3, 6, 5, 2, 1, 5, 4, 1], name='Front')
        ranker.update(out=['x', 'y'])
        actual = ranker.pareto_fronts(objectives=['max', 'min'])
        assert_series_equal(expected, actual)

        # Test 2 (min y then max x) - should be the same
        expected = pd.Series([4, 3, 6, 5, 2, 1, 5, 4, 1], name='Front')
        ranker.update(out=['y', 'x'])
        actual = ranker.pareto_fronts(objectives=['min', 'max'])
        assert_series_equal(expected, actual)

        # Test 3 (min x then max y)
        expected = pd.Series([1, 4, 1, 2, 5, 6, 2, 3, 6], name='Front')
        ranker.update(out=['x', 'y'])
        actual = ranker.pareto_fronts(objectives=['min', 'max'])
        assert_series_equal(expected, actual)

        # Test 4 (max x then max y)
        expected = pd.Series([4, 3, 2, 2, 2, 1, 1, 1, 1], name='Front')
        ranker.update(out=['x', 'y'])
        actual = ranker.pareto_fronts(objectives=['max', 'max'])
        assert_series_equal(expected, actual)

        # Test 5 (min x then min y)
        expected = pd.Series([1, 2, 2, 2, 3, 4, 3, 3, 4], name='Front')
        ranker.update(out=['x', 'y'])
        actual = ranker.pareto_fronts(objectives=['min', 'min'])
        assert_series_equal(expected, actual)

        # Test 6 (max z then min w)
        expected = pd.Series([3, 2, 1, 3, 2, 1, 3, 2, 1], name='Front')
        ranker.update(out=['z', 'w'])
        actual = ranker.pareto_fronts(objectives=['max', 'min'])
        assert_series_equal(expected, actual)

        # Test 6 (max x then min y then min z then max w)
        expected = pd.Series([1, 1, 2, 1, 1, 1, 1, 1, 1], name='Front')
        ranker.update(out=['x', 'y', 'z', 'w'])
        actual = ranker.pareto_fronts(objectives=['max', 'min', 'min', 'max'])
        assert_series_equal(expected, actual)

    def test_multi_objective_pareto_ranking(self):
        """
        Testing if the multi criteria (pareto) ranks the results according to its
        objectives.

        Tests
        -----
        1. 2 variables
        2. 3 variables
        3. 5 variables
        """
        results_data = {
            'v': [6, 2, 12, -4, 2, 5, 5, 8, 11],
            'w': [-11, -125, 1001, 112, 12, 13, 31, 30, 0],
            'x': [-10, 10, 0, 1, 10, 15, 2, 8, -1],
            'y': [2, 2, 40, 13, 2, 2, 40, 7, 12],
            'z': [0, 0, 1, 1, 0, 0, 1, 0, 1]
        }
        results_df = pd.DataFrame(results_data)
        ranker = RankCalculator(results_df)

        # Test 1
        expected = pd.Series([9, 4, 4, 4, 4, 1, 1, 1, 8], name='Rank')
        ranker.update(out=['x', 'y'])
        actual = ranker.pareto(['max', 'max'])
        assert_series_equal(expected, actual)

        # Test 2
        expected = pd.Series([8, 4, 4, 4, 4, 1, 1, 1, 8], name='Rank')
        ranker.update(out=['x', 'y', 'z'])
        actual = ranker.pareto(['max', 'max', 'min'])
        assert_series_equal(expected, actual)

        # Test 3
        expected = pd.Series([7, 1, 7, 1, 7, 1, 1, 6, 1], name='Rank')
        ranker.update(out=['v', 'w', 'x', 'y', 'z'])
        actual = ranker.pareto(['min', 'min', 'max', 'max', 'min'])
        assert_series_equal(expected, actual)
