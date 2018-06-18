import unittest
import glob
import os
import numpy as np

from cgc.geometry_correction import GeometryCorrection

ERR_OFFSET = 0.01


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
        o_cgc.define_parameters(pixel_center=50, radius1=10)
        self.assertEqual(50, o_cgc.pixel_center)

        ## radius1
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50, radius1=-3.5)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50, radius1=-3)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=10, radius1=20)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=50, radius1=800)

        # correct radius1 correctly saved
        o_cgc.define_parameters(pixel_center=250, radius1=100)
        self.assertEqual(100, o_cgc.radius1)

        ## radius2
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=100, radius1=50, radius2=0.5)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=100, radius1=50, radius2=-20)
        self.assertRaises(ValueError, o_cgc.define_parameters, pixel_center=100, radius1=50, radius2=200)

        # correct radius1 correctly saved
        o_cgc.define_parameters(pixel_center=250, radius1=120, radius2=100)
        self.assertEqual(100, o_cgc.radius2)
        self.assertEqual(120, o_cgc.radius1)

        # correct radius1 correctly saved when no radius2
        o_cgc = GeometryCorrection(list_files=list_tiff)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=250, radius1=120)
        self.assertEqual(120, o_cgc.radius1)

        # make sure program sort the radius1 and 2 (radius1 being always the outside radius)
        o_cgc = GeometryCorrection(list_files=list_tiff)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=250, radius1=100, radius2=150)
        self.assertEqual(o_cgc.radius1, 150)
        self.assertEqual(o_cgc.radius2, 100)


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


class testHomogeneousCorrection(unittest.TestCase):

    def setUp(self):
        _file_path = os.path.dirname(__file__)
        self.data_path = os.path.abspath(os.path.join(_file_path, "../../notebooks/data/"))

    def test_calculate_sample_thickness(self):
        """assert calculation of thickness is correct for homo and inhomogeneous"""

        list_fits = glob.glob(self.data_path + '/fits/homogeneous*.fits')
        o_cgc = GeometryCorrection(list_files=list_fits)
        o_cgc.load_files()

        # homogeneous sample
        o_cgc.define_parameters(pixel_center=100, radius1=50)
        self.assertEqual(o_cgc.get_sample_thickness_at_center(), 100)

        # inhomogeneous sample
        o_cgc.define_parameters(pixel_center=100, radius1=50, radius2=70)
        self.assertEqual(o_cgc.get_sample_thickness_at_center(), 40)

    def test_calculate_pixel_intensity(self):
        """assert calculation of pixel_intensity works"""
        list_fits = glob.glob(self.data_path + '/tiff/homogeneous*.tif')
        o_cgc = GeometryCorrection(list_files=list_fits)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=256, radius1=200)
        _image_0 = o_cgc.list_data[0]
        _slice_50 = _image_0[50, :]
        self.assertAlmostEqual(2.00,
                               o_cgc.calculate_pixel_intensity(slice=_slice_50),
                               delta=ERR_OFFSET)

    def test_isolate_cylinder_from_image(self):
        """assert isolation of cylinder works for homogeneous and inhomogeneous"""
        list_tiff = glob.glob(self.data_path + '/tiff/homogeneous*.tif')
        o_cgc = GeometryCorrection(list_files=list_tiff)
        o_cgc.load_files()
        pixel_center = 256
        radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius)
        _isolated_cylinder_calculated = o_cgc.isolate_cylinder_from_image(index=0)

        # what we expect
        image_0 = o_cgc.list_data[0]
        _isolated_cylinder_expected = image_0[:, pixel_center-radius:pixel_center+radius+1]
        self.assertTrue((_isolated_cylinder_calculated == _isolated_cylinder_expected).all())

    def test_correction_intensity_2(self):
        """assert the correction works"""
        tiff_image = glob.glob(self.data_path + '/tiff/homogeneous_image_px_intensity_2.tif')
        o_cgc = GeometryCorrection(list_files=tiff_image)
        o_cgc.load_files()
        pixel_center = 256
        radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius)
        o_cgc.correct()
        first_image_corrected = o_cgc.list_data_corrected[0]
        row_10_returned = first_image_corrected[10, :]
        row_10_expected = np.ones(400)*2
        for _returned, _expected in zip(row_10_returned, row_10_expected):
            self.assertAlmostEqual(_returned, _expected, delta=0.1)

    def test_correction_intensity_4(self):
        """assert the correction works"""
        tiff_image = glob.glob(self.data_path + '/tiff/homogeneous_image_px_intensity_4.tif')
        o_cgc = GeometryCorrection(list_files=tiff_image)
        o_cgc.load_files()
        pixel_center = 256
        radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, radius1=radius)
        o_cgc.correct()
        first_image_corrected = o_cgc.list_data_corrected[0]
        row_10_returned = first_image_corrected[10, :]
        row_10_expected = np.ones(400)*4
        for _returned, _expected in zip(row_10_returned, row_10_expected):
            self.assertAlmostEqual(_returned, _expected, delta=0.1)

