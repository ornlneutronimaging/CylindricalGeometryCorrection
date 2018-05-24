import unittest
import glob
import os
import numpy as np

from cgc.geometry_correction import GeometryCorrection


class TestInitialization(unittest.TestCase):

    def setUp(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, "../../notebooks/data/"))

    def test_list_files_has_correct_format(self):
        '''assert the list of files is a non empty and existing list of files'''

        # string not allowed during initialization
        list_files = ''
        self.assertRaises(TypeError, GeometryCorrection, list_files=list_files)

        # name of files should exist
        list_files = ['i_do_not_exist.tiff']
        self.assertRaises(ValueError, GeometryCorrection, list_files=list_files)

        # files exist
        list_files = glob.glob(self.data_path + '/*.fits')
        o_cgc = GeometryCorrection(list_files=list_files)
        list_files_returned = o_cgc.list_files
        self.assertTrue(list_files == list_files_returned)


class TestLoading(unittest.TestCase):

    def setUp(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, "../../notebooks/data/"))

    def test_loading_tiff_works(self):
        '''assert loading tiff of same size works'''
        list_tiff = glob.glob(self.data_path + '/tiff/homogeneous*.tif')
        o_cgc = GeometryCorrection(list_files=list_tiff)
        o_cgc.load_files()

        # we have correct number of arrays loaded
        len_expected = len(list_tiff)
        len_loaded = len(o_cgc.list_data)
        self.assertEqual(len_expected, len_loaded)

        # size of array is correct
        [height_loaded, width_loaded] = np.shape(o_cgc.list_data[0])
        [height_expected, width_expected] = [512, 512]
        self.assertTrue(height_expected, height_loaded)
        self.assertTrue(width_expected, width_loaded)

    def test_loading_fits_works(self):
        '''assert loading fits of same size works'''
        list_fits = glob.glob(self.data_path + '/fits/homogeneous*.fits')
        o_cgc = GeometryCorrection(list_files=list_fits)
        o_cgc.load_files()

        # we have correct number of arrays loaded
        len_expected = len(list_fits)
        len_loaded = len(o_cgc.list_data)
        self.assertEqual(len_expected, len_loaded)

        # size of array is correct
        [height_loaded, width_loaded] = np.shape(o_cgc.list_data[0])
        [height_expected, width_expected] = [512, 512]
        self.assertTrue(height_expected, height_loaded)
        self.assertTrue(width_expected, width_loaded)

