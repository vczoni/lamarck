import unittest
import doctest
import numpy as np


from lamarck import GenomeBlueprintBuilder
from lamarck.utils import VectorialOverloadException


class TestGenomeBlueprintBuilder(unittest.TestCase):
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
        builder = GenomeBlueprintBuilder()
        actual = builder.get_blueprint()
        self.assertDictEqual(expected, actual)

        # Test 2
        builder._blueprint.update({'a': 1})
        builder.reset()
        actual = builder.get_blueprint()
        self.assertDictEqual(expected, actual)
    
    def test_get_blueprint(self):
        """
        Test geting the current blueprint from the builder.

        Genes:
            x: NUMERIC `int` number varying from 1 to 10
            y: CATEGORICAL value with domain ['a' , 'b', 'c']
            z: VECTORIAL value with domain (0, 1), with replacement and length 8
            w: BOOLEAN
        """
        expected = {
            'x': {
                'type': 'numeric',
                'domain': int,
                'range': [1, 10]},
            'y': {
                'type': 'categorical',
                'domain': ['a', 'b', 'c']},
            'z': {
                'type': 'vectorial',
                'domain': (0, 1),
                'replacement': True,
                'length': 8},
            'w': {
                'type': 'boolean'}
        }
        builder = GenomeBlueprintBuilder()
        builder.add_numeric_gene(name='x', domain=int, range=[1, 10])
        builder.add_categorical_gene(name='y', domain=['a', 'b', 'c'])
        builder.add_vectorial_gene(name='z', domain=(0, 1), replacement=True,
                                   length=8)
        builder.add_boolean_gene(name='w')
        actual = builder.get_blueprint()
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
        2. Trying with a domain that is not `int` or `float`
        """
        builder = GenomeBlueprintBuilder()

        # Test 1
        expected = {
            'x': {
                'type': 'numeric',
                'domain': int,
                'range': [1, 10]},
            'y': {
                'type': 'numeric',
                'domain': float,
                'range': [0, 6*np.pi]}
        }
        builder.add_numeric_gene(name='x',
                                 domain=int,
                                 range=[1, 10])
        builder.add_numeric_gene(name='y',
                                 domain=float,
                                 range=[0, 6*np.pi])
        actual = builder.get_blueprint()
        self.assertDictEqual(expected, actual)

        # Test 2
        with self.assertRaises(TypeError):
            builder.add_numeric_gene(name='x',
                                     domain=bool,
                                     range=[False, True])
    
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
                'domain': ['Jake', 'Amy', 'Raymond', 'Rosa']},
            'ages': {
                'type': 'categorical',
                'domain': [35, 34, 53, 29]}
        }
        builder = GenomeBlueprintBuilder()
        builder.add_categorical_gene(name='names',
                                     domain=['Jake', 'Amy', 'Raymond', 'Rosa'])
        builder.add_categorical_gene(name='ages',
                                     domain=[35, 34, 53, 29])
        actual = builder.get_blueprint()
        self.assertDictEqual(expected, actual)
    
    def test_adding_vectorial_gene(self):
        """
        Test adding vectorial Genes to the Blueprint
        Tests
        -----
        1. With valid specs
        Genes:
            vec_replacement: domain=[0, 1], replacement=True, lenght=5
            vec_no_replacement: domain=['X', 'Y', 'Z'], replacement=False, lenght=3
        2. With invalid specs
        Genes:
            vec_invalid: domain=['X', 'Y', 'Z'], replacement=False, lenght=5
        """
        # Test 1
        expected = {
            'vec_replacement': {
                'type': 'vectorial',
                'domain': [0, 1],
                'replacement': True,
                'length': 5},
            'vec_no_replacement': {
                'type': 'vectorial',
                'domain': ['X', 'Y', 'Z'],
                'replacement': False,
                'length': 3}
        }
        builder = GenomeBlueprintBuilder()
        builder.add_vectorial_gene(name='vec_replacement',
                                   domain=[0, 1],
                                   replacement=True,
                                   length=5)
        builder.add_vectorial_gene(name='vec_no_replacement',
                                   domain=['X', 'Y', 'Z'],
                                   replacement=False,
                                   length=3)
        actual = builder.get_blueprint()
        self.assertDictEqual(expected, actual)

        # Test 2
        with self.assertRaises(VectorialOverloadException):
            builder.add_vectorial_gene(name='vec_invalid',
                                    domain=['X', 'Y', 'Z'],
                                    replacement=False,
                                    length=5)
        
    def test_adding_boolean_gene(self):
        """
        Test adding boolean Genes to the Blueprint
        Genes:
            flag1
            flag2
        """
        expected = {
            'flag1': {'type': 'boolean'},
            'flag2': {'type': 'boolean'},
        }
        builder = GenomeBlueprintBuilder()
        builder.add_boolean_gene(name='flag1')
        builder.add_boolean_gene(name='flag2')
        actual = builder.get_blueprint()
        self.assertDictEqual(expected, actual)
