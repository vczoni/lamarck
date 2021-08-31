import unittest
import numpy as np


from lamarck import BlueprintBuilder
from lamarck.utils import VectorialOverloadException


class TestBlueprintBuilder(unittest.TestCase):
    """
    Testing the basic genome blueprint constructor.
    """

    def test_reset_blueprint(self):
        """
        Test if resetting the blueprint actually garantees that the :_blueprint:
        attribute is actually an empty `dict`

        Tests
        -----
        1. Test if newly created buider has an empty blueprint
        2. Test if buider effectively resets the blueprint
        """
        expected = {}

        # Test 1
        builder = BlueprintBuilder()
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)

        # Test 2
        builder._blueprint.update({'a': 1})
        builder.reset()
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)

    def test_get_blueprint(self):
        """
        Test geting the current blueprint from the builder.

        Genes:
            x: NUMERIC `int` number varying from 1 to 10
            y: CATEGORICAL value with domain ['a' , 'b', 'c']
            z: ARRAY value with domain (0, 1), with length 8
            v: SET value with domain (0, 1, 2), with length 3
            w: BOOLEAN
        """
        expected = {
            'x': {
                'type': 'integer',
                'specs': {'domain': (1, 10)}},
            'y': {
                'type': 'categorical',
                'specs': {'domain': ('a', 'b', 'c')}},
            'z': {
                'type': 'array',
                'specs': {'domain': (0, 1),
                          'length': 8}},
            'v': {
                'type': 'set',
                'specs': {'domain': (0, 1, 2),
                          'length': 3}},
            'w': {
                'type': 'boolean',
                'specs': {}}
        }
        builder = BlueprintBuilder()
        builder.add_integer_gene(name='x', domain=(1, 10))
        builder.add_categorical_gene(name='y', domain=('a', 'b', 'c'))
        builder.add_array_gene(name='z', domain=(0, 1), length=8)
        builder.add_set_gene(name='v', domain=(0, 1, 2), length=3)
        builder.add_boolean_gene(name='w')
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)

    def test_adding_numeric_gene(self):
        """
        Test adding numeric Genes to the Blueprint.

        Tests
        -----
        1. Adding an `int` gene and a `float` gene
        Genes:
            x: `int` number varying from 1 to 10
            y: `float` number varying from 0 to 6 pi
        """
        builder = BlueprintBuilder()

        # Test 1
        expected = {
            'x': {
                'type': 'integer',
                'specs': {'domain': (1, 10)}},
            'y': {
                'type': 'float',
                'specs': {'domain': (0, 6*np.pi)}}
        }
        builder.add_integer_gene(name='x',
                                 domain=(1, 10))
        builder.add_float_gene(name='y',
                               domain=(0, 6*np.pi))
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)

    def test_adding_categorical_gene(self):
        """
        Test adding categorical Genes to the Blueprint.

        Genes:
            names: domain = ['Jake', 'Amy', 'Raymond', 'Rosa']
            ages: domain = [35, 34, 53, 29]
        """
        expected = {
            'names': {
                'type': 'categorical',
                'specs': {'domain': ('Jake', 'Amy', 'Raymond', 'Rosa')}},
            'ages': {
                'type': 'categorical',
                'specs': {'domain': (35, 34, 53, 29)}}
        }
        builder = BlueprintBuilder()
        builder.add_categorical_gene(name='names',
                                     domain=('Jake', 'Amy', 'Raymond', 'Rosa'))
        builder.add_categorical_gene(name='ages',
                                     domain=(35, 34, 53, 29))
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)

    def test_adding_vectorial_gene(self):
        """
        Test adding vectorial Genes to the Blueprint
        Tests
        -----
        1. With valid specs
        Genes:
            vec_replacement: domain=[0, 1], replacement=True (array), lenght=5
            vec_no_replacement: domain=['X', 'Y', 'Z'], replacement=False (set), lenght=3
        2. With invalid specs
        Genes:
            vec_invalid: domain=['X', 'Y', 'Z'], replacement=False (set), lenght=5
        """
        # Test 1
        expected = {
            'vec_replacement': {
                'type': 'array',
                'specs': {'domain': (0, 1),
                          'length': 5}},
            'vec_no_replacement': {
                'type': 'set',
                'specs': {'domain': ('X', 'Y', 'Z'),
                          'length': 3}}
        }
        builder = BlueprintBuilder()
        builder.add_array_gene(name='vec_replacement',
                               domain=(0, 1),
                               length=5)
        builder.add_set_gene(name='vec_no_replacement',
                             domain=('X', 'Y', 'Z'),
                             length=3)
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)

        # Test 2
        with self.assertRaises(VectorialOverloadException):
            builder.add_set_gene(name='vec_invalid',
                                 domain=('X', 'Y', 'Z'),
                                 length=5)

    def test_adding_boolean_gene(self):
        """
        Test adding boolean Genes to the Blueprint
        Genes:
            flag1
            flag2
        """
        expected = {
            'flag1': {'type': 'boolean', 'specs': {}},
            'flag2': {'type': 'boolean', 'specs': {}},
        }
        builder = BlueprintBuilder()
        builder.add_boolean_gene(name='flag1')
        builder.add_boolean_gene(name='flag2')
        actual = builder.get_blueprint()._dict
        self.assertDictEqual(expected, actual)
