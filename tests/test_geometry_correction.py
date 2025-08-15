import numpy as np
import pytest

from neutron_geomcorr.geometry_correction import GeometryCorrection

ERR_OFFSET = 0.01


class TestInitialization:
    def test_list_files_has_correct_format(self, sample_fits_file):
        """assert the list of files is a non empty and existing list of files"""

        # string not allowed during initialization
        list_files = ""
        with pytest.raises(TypeError):
            GeometryCorrection(list_files=list_files)

        # name of files should exist
        list_files = ["i_do_not_exist.tiff"]
        with pytest.raises(ValueError):
            GeometryCorrection(list_files=list_files)

        # files exist - use the fixture-provided file
        list_files = [sample_fits_file]
        o_cgc = GeometryCorrection(list_files=list_files)
        list_files_returned = o_cgc.list_files
        assert list_files == list_files_returned

    def test_parameters_should_be_defined_after_loading_data(self, homogeneous_tiff_files):
        """assert center and radius are defined after loading the data"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        with pytest.raises(AttributeError):
            o_cgc.define_parameters(pixel_center=10)

    def test_parameters_should_be_correctly_defined(self, homogeneous_tiff_files):
        """assert pixel_center, outer_radius and inner_radius (if defined) have correct format"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)

        ## pixel
        # pixel center should be integer, >0 and within the image size
        o_cgc.load_files()
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=2.5)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=-3)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=600)

        # pixel in right range correctly saved
        o_cgc.define_parameters(pixel_center=50, outer_radius=10)
        assert o_cgc.pixel_center == 50

        ## outer_radius
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=50)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=50, outer_radius=-3.5)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=50, outer_radius=-3)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=10, outer_radius=20)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=50, outer_radius=800)

        # correct outer_radius correctly saved
        o_cgc.define_parameters(pixel_center=250, outer_radius=100)
        assert o_cgc.outer_radius == 100

        ## inner_radius
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=100, outer_radius=50, inner_radius=0.5)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=100, outer_radius=50, inner_radius=-20)
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=100, outer_radius=50, inner_radius=200)

        # correct outer_radius correctly saved
        o_cgc.define_parameters(pixel_center=250, outer_radius=120, inner_radius=100)
        assert o_cgc.inner_radius == 100
        assert o_cgc.outer_radius == 120

        # correct outer_radius correctly saved when no inner_radius
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=250, outer_radius=120)
        assert o_cgc.outer_radius == 120

        # make sure program sort the outer_radius and 2 (outer_radius being always the outside radius)
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=250, outer_radius=100, inner_radius=150)
        assert o_cgc.outer_radius == 150
        assert o_cgc.inner_radius == 100


class TestLoading:
    def test_loading_tiff_works(self, homogeneous_tiff_files):
        """assert loading tiff of same size works"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()

        # we have correct number of arrays loaded
        len_expected = len(homogeneous_tiff_files)
        len_loaded = len(o_cgc.list_data)
        assert len_expected == len_loaded

        # size of array is correct
        [height_loaded, width_loaded] = np.shape(o_cgc.list_data[0])
        [height_expected, width_expected] = [512, 512]
        assert height_expected == height_loaded
        assert width_expected == width_loaded

    def test_loading_fits_works(self, homogeneous_fits_files):
        """assert loading fits of same size works"""
        o_cgc = GeometryCorrection(list_files=homogeneous_fits_files)
        o_cgc.load_files()

        # we have correct number of arrays loaded
        len_expected = len(homogeneous_fits_files)
        len_loaded = len(o_cgc.list_data)
        assert len_expected == len_loaded

        # size of array is correct
        [height_loaded, width_loaded] = np.shape(o_cgc.list_data[0])
        [height_expected, width_expected] = [512, 512]
        assert height_expected == height_loaded
        assert width_expected == width_loaded


class TestHomogeneousCorrection:
    def test_calculate_sample_thickness(self, homogeneous_fits_files):
        """assert calculation of thickness is correct for homogeneous sample"""
        o_cgc = GeometryCorrection(list_files=homogeneous_fits_files)
        o_cgc.load_files()

        # homogeneous sample
        o_cgc.define_parameters(pixel_center=100, outer_radius=50)
        assert o_cgc.get_sample_thickness_at_center() == 100

    def test_calculate_pixel_intensity(self, tiff_data_dir):
        """assert calculation of pixel_intensity works"""
        # Use specific file for this test
        list_fits = [str(tiff_data_dir / "homogeneous_image_px_intensity_2.tif")]
        o_cgc = GeometryCorrection(list_files=list_fits)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=256, outer_radius=200)
        _image_0 = o_cgc.list_data[0]
        _slice_50 = _image_0[50, :]
        assert o_cgc.calculate_pixel_intensity(slice=_slice_50) == pytest.approx(2.00, abs=ERR_OFFSET)

    def test_isolate_cylinder_from_image(self, homogeneous_tiff_files):
        """assert isolation of cylinder works for homogeneous and inhomogeneous"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        pixel_center = 256
        radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=radius)
        _isolated_cylinder_calculated = o_cgc.isolate_cylinder_from_image(index=0)

        # what we expect
        image_0 = o_cgc.list_data[0]
        _isolated_cylinder_expected = image_0[:, pixel_center - radius : pixel_center + radius + 1]
        assert (_isolated_cylinder_calculated == _isolated_cylinder_expected).all()
        del o_cgc

    def test_correction_intensity_4(self, tiff_data_dir):
        """assert the correction works"""
        tiff_image = [str(tiff_data_dir / "homogeneous_image_px_intensity_4.tif")]
        o_cgc_4 = GeometryCorrection(list_files=tiff_image)
        o_cgc_4.load_files()
        pixel_center = 256
        radius = 200
        o_cgc_4.define_parameters(pixel_center=pixel_center, outer_radius=radius)
        o_cgc_4.correct()
        first_image_corrected4 = o_cgc_4.list_data_corrected[0]
        row_10_returned = first_image_corrected4[10, :]
        row_10_expected = np.ones(400) * 4
        for _returned4, _expected4 in zip(row_10_returned, row_10_expected):
            assert _returned4 == pytest.approx(_expected4, abs=0.1)
        del o_cgc_4

    def test_correction_intensity_2(self, tiff_data_dir):
        """assert the correction works"""
        tiff_image = [str(tiff_data_dir / "homogeneous_image_px_intensity_2.tif")]
        o_cgc = GeometryCorrection(list_files=tiff_image)
        o_cgc.load_files()
        pixel_center = 256
        radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=radius)
        o_cgc.correct()
        first_image_corrected = o_cgc.list_data_corrected[0]
        row_10_returned = first_image_corrected[10, :]
        row_10_expected = np.ones(400) * 2
        for _returned, _expected in zip(row_10_returned, row_10_expected):
            assert _returned == pytest.approx(_expected, abs=0.1)


class TestInhomogeneousCorrection:
    def test_calculate_sample_thickness(self, inhomogeneous_fits_files):
        """assert calculation of thickness is correct for inhomogeneous"""
        o_cgc = GeometryCorrection(list_files=inhomogeneous_fits_files)
        o_cgc.load_files()

        # inhomogeneous sample
        o_cgc.define_parameters(pixel_center=100, outer_radius=50, inner_radius=70)
        assert o_cgc.get_sample_thickness_at_center() == 40

    def test_calculate_pixel_intensity(self, tiff_data_dir):
        """assert calculation of pixel_intensity works"""
        list_fits = [str(tiff_data_dir / "inhomogeneous_image_px_intensity_2.tif")]
        o_cgc = GeometryCorrection(list_files=list_fits)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=256, inner_radius=150, outer_radius=200)
        _image_0 = o_cgc.list_data[0]
        _slice_50 = _image_0[50, :]
        assert o_cgc.calculate_pixel_intensity(slice=_slice_50) == pytest.approx(2.00, abs=ERR_OFFSET)

    def test_isolate_cylinder_from_image(self, inhomogeneous_tiff_files):
        """assert isolation of cylinder works for inhomogeneous"""
        o_cgc = GeometryCorrection(list_files=inhomogeneous_tiff_files)
        o_cgc.load_files()
        pixel_center = 256
        inner_radius = 150
        outer_radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=outer_radius, inner_radius=inner_radius)
        _isolated_cylinder_calculated = o_cgc.isolate_cylinder_from_image(index=0)

        # what we expect
        image_0 = o_cgc.list_data[0]
        _isolated_cylinder_expected = image_0[:, pixel_center - outer_radius : pixel_center + outer_radius + 1]
        assert (_isolated_cylinder_calculated == _isolated_cylinder_expected).all()

    def test_correction_intensity_2(self, tiff_data_dir):
        """assert the correction works for inhomogeneous sample of intensity 2"""
        tiff_image = [str(tiff_data_dir / "inhomogeneous_image_px_intensity_2.tif")]
        o_cgc = GeometryCorrection(list_files=tiff_image)
        o_cgc.load_files()
        pixel_center = 256
        outer_radius = 200
        inner_radius = 150
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=outer_radius, inner_radius=inner_radius)
        o_cgc.correct()
        first_image_corrected = o_cgc.list_data_corrected[0]
        row_10_returned = first_image_corrected[10, :]
        row_10_expected = np.ones(400) * 2
        for _returned, _expected in zip(row_10_returned, row_10_expected):
            assert _returned == pytest.approx(_expected, abs=0.1)

    def test_correction_intensity_4(self, tiff_data_dir):
        """assert the correction works for inhomogeneous sample of intensity 4"""
        tiff_image = [str(tiff_data_dir / "inhomogeneous_image_px_intensity_4.tif")]
        o_cgc = GeometryCorrection(list_files=tiff_image)
        o_cgc.load_files()
        pixel_center = 256
        outer_radius = 200
        inner_radius = 150
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=outer_radius, inner_radius=inner_radius)
        o_cgc.correct()
        first_image_corrected = o_cgc.list_data_corrected[0]
        row_10_returned = first_image_corrected[10, :]
        row_10_expected = np.ones(400) * 4
        for _returned, _expected in zip(row_10_returned, row_10_expected):
            assert _returned == pytest.approx(_expected, abs=0.1)
