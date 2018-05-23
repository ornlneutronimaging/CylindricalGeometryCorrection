import unittest
import numpy as np

from cgc.geometry_correction import GeometryCorrection


class TestNothing(unittest.TestCase):

    def setUp(self):
        pass

    def test_list_files_has_correct_format(self):
        '''assert the list of files is a non empty and existing list of files'''

        # string not allowed
        list_files = ''
        self.assertRaises(TypeError, GeometryCorrection, list_files=list_files)

        # name of files should exist
        list_files = ['i_do_not_exist.tiff']
        self.assertRaises(ValueError, GeometryCorrection, list_files=list_files)