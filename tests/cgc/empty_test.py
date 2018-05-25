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

    def test_parameters_should_be_defined_after_loading_data(self):
        '''assert center and radius are defined after loading the data'''
        list_tiff = glob.glob(self.data_path + '/tiff/homogeneous*.tif')
        o_cgc = GeometryCorrection(list_files=list_tiff)
        self.assertRaises(AttributeError, o_cgc.define_parameters, pixel_center=10)

    def test_parameters_should_be_correctly_defined(self):
        '''assert pixel_center, radius1 and radius2 (if defined) have correct format'''
        list_tiff = glob.glob(self.data_path + '/tiff/homogeneous*.tif')
        o_cgc = GeometryCorrection(list_files=list_tiff)

        ## pixel
        # pixel center should be integer, >0 and withing the image size
        o_cgc.load_files()
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=2.5)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=-3)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=600)

        # pixel in right range correctly saved
        pixel_center_expected = 50
        o_cgc.define_parameters(pixel_center=pixel_center_expected, radius1=10)
        pixel_center_returned = o_cgc.pixel_center
        self.assertEqual(pixel_center_expected, pixel_center_returned)

        ## radius1
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50, radius1=-3.5)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50, radius1=-3)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=10, radius1=20)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50, radius1=800)

        # correct radius1 correctly saved
        radius1_expected = 100
        pixel_center = 250
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius1_expected)
        radius1_returned = o_cgc.radius1
        self.assertEqual(radius1_expected, radius1_returned)

        ## radius2
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=100, radius1=50, radius2=0.5)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=100, radius1=50, radius2=-20)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=100, radius1=50, radius2=200)

        # correct radius1 correctly saved
        radius1 = 100
        radius2_expected = 120
        pixel_center = 250
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius1, radius2=radius2_expected)
        radius2_returned = o_cgc.radius2
        self.assertEqual(radius2_expected, radius2_returned)

        # make sure program sort the radius1 and 2 (radius1 < radius2)
        radius1 = 150
        radius2 = 100
        pixel_center = 250
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius1, radius2=radius2)
        radius1_saved = o_cgc.radius1
        radius2_saved = o_cgc.radius2
        self.assertEqual(radius1_saved, radius2)
        self.assertEqual(radius2_saved, radius1)


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


class testCorrection(unittest.TestCase):

    def setUp(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, "../../notebooks/data/"))

    def test_calculate_sample_thickness(self):
        """assert calculation of thickness is correct for homo and inhomogeneous"""
        radius1 = 50
        pixel_center = 100

        list_fits = glob.glob(self.data_path + '/fits/homogeneous*.fits')
        o_cgc = GeometryCorrection(list_files=list_fits)
        o_cgc.load_files()

        # homogeneous sample
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius1)
        thickness_calculated = o_cgc.get_sample_thickness_at_center()
        thickness_expected = 100
        self.assertEqual(thickness_calculated, thickness_expected)

        # inhomogeneous sample
        radius2 = 70
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius1, radius2=radius2)
        thickness_calculated = o_cgc.get_sample_thickness_at_center()
        thickness_expected = 40
        self.assertEqual(thickness_calculated, thickness_expected)

