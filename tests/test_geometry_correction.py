import inspect

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

    @pytest.mark.parametrize("subdir, ext", [("tiff", ".tif"), ("fits", ".fits")])
    @pytest.mark.parametrize("intensity", [2, 4, 6, 8])
    def test_correction_flattens_homogeneous(self, test_data_dir, subdir, ext, intensity):
        """assert the correction flattens homogeneous cylinders for both formats

        Also encodes the output-shape contract: isolate_cylinder_from_image
        extracts 2R+1 columns and _correct_file_index trims one column on each
        side, so the corrected width is 2R-1. The previous zip()-based
        comparison silently truncated and could not detect a width change.
        """
        path = test_data_dir / subdir / f"homogeneous_image_px_intensity_{intensity}{ext}"
        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        pixel_center = 256
        radius = 200
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=radius)
        o_cgc.correct()
        corrected = o_cgc.list_data_corrected[0]

        assert corrected.shape == (512, 2 * radius - 1)
        # edge deviation scales with intensity (~2.5% of the flat value at the
        # cylinder rim), so the tolerance must be relative; the historical
        # abs=0.1 only ever held for intensities 2 and 4
        np.testing.assert_allclose(corrected[10, :], np.full(2 * radius - 1, float(intensity)), rtol=0.03)


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

    @pytest.mark.parametrize("subdir, ext", [("tiff", ".tif"), ("fits", ".fits")])
    @pytest.mark.parametrize("intensity", [2, 4, 6, 8])
    def test_correction_flattens_inhomogeneous(self, test_data_dir, subdir, ext, intensity):
        """assert the correction flattens hollow cylinders for both formats"""
        path = test_data_dir / subdir / f"inhomogeneous_image_px_intensity_{intensity}{ext}"
        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        pixel_center = 256
        outer_radius = 200
        inner_radius = 150
        o_cgc.define_parameters(pixel_center=pixel_center, outer_radius=outer_radius, inner_radius=inner_radius)
        o_cgc.correct()
        corrected = o_cgc.list_data_corrected[0]

        assert corrected.shape == (512, 2 * outer_radius - 1)
        np.testing.assert_allclose(corrected[10, :], np.full(2 * outer_radius - 1, float(intensity)), rtol=0.03)


class TestRun:
    def test_run_and_load_files_keep_notebook_kwarg(self):
        """assert the notebook progress-bar kwarg stays part of the public API

        All six tutorial notebooks call load_files(notebook=True); no other
        test exercises the kwarg, so a loader rewrite could silently drop it.
        """
        assert "notebook" in inspect.signature(GeometryCorrection.load_files).parameters
        assert "notebook" in inspect.signature(GeometryCorrection.run).parameters

    def test_load_files_notebook_kwarg_works_headless(self, homogeneous_tiff_files):
        """assert notebook=True works outside Jupyter

        The progress bar is tqdm.auto, which falls back to a console bar
        when no notebook frontend is available, so notebook=True must not
        raise in a headless environment.
        """
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files[:1])
        o_cgc.load_files(notebook=True)
        assert len(o_cgc.list_data) == 1

    def test_run_loads_defines_parameters_and_corrects(self, homogeneous_tiff_files):
        """assert run() performs the full load + define + correct pipeline

        Previously run() silently skipped the correction step despite its
        docstring; list_data_corrected stayed empty.
        """
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.run(pixel_center=256, outer_radius=200)

        assert len(o_cgc.list_data) == len(homogeneous_tiff_files)
        assert o_cgc.pixel_center == 256
        assert o_cgc.outer_radius == 200
        assert len(o_cgc.list_data_corrected) == len(homogeneous_tiff_files)
        assert o_cgc.list_data_corrected[0].shape == (512, 2 * 200 - 1)


class TestValidation:
    """Regression tests for the correctness batch (audit M2-M5, L1-L3)."""

    def test_correct_requires_loaded_data(self, homogeneous_tiff_files):
        """M5: correct() before load_files() must raise, not silently no-op"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        with pytest.raises(AttributeError, match="load_files"):
            o_cgc.correct()

    def test_correct_requires_defined_parameters(self, homogeneous_tiff_files):
        """M5: correct() after load but before define_parameters() must raise
        a descriptive error instead of an opaque TypeError from slicing"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        with pytest.raises(AttributeError, match="define_parameters"):
            o_cgc.correct()

    def test_failed_define_parameters_does_not_unlock_correct(self, homogeneous_tiff_files):
        """M5: a define_parameters() call that raises must leave correct() locked"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        with pytest.raises(ValueError):
            o_cgc.define_parameters(pixel_center=256, outer_radius=800)
        with pytest.raises(AttributeError, match="define_parameters"):
            o_cgc.correct()

    def test_zero_wall_thickness_rejected(self, homogeneous_tiff_files):
        """M3: inner_radius == outer_radius (zero wall) must raise instead of
        producing NaN/inf garbage downstream"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        with pytest.raises(ValueError, match="zero wall thickness"):
            o_cgc.define_parameters(pixel_center=256, outer_radius=100, inner_radius=100)

        # same guard on the outer_radius setter (hollow state already defined)
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        o_cgc.define_parameters(pixel_center=256, outer_radius=120, inner_radius=100)
        with pytest.raises(ValueError, match="zero wall thickness"):
            o_cgc.outer_radius = 100

    def test_inner_radius_requires_outer_radius_first(self, homogeneous_tiff_files):
        """M4: setting inner_radius while outer_radius is unset previously
        swap-assigned the value as the OUTER radius — silently a solid cylinder"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        o_cgc.pixel_center = 256
        with pytest.raises(ValueError, match="outer_radius first"):
            o_cgc.inner_radius = 100
        # the failed assignment must not have corrupted the geometry
        assert np.isnan(o_cgc.outer_radius)
        assert np.isnan(o_cgc.inner_radius)

    def test_numpy_integers_accepted(self, homogeneous_tiff_files):
        """L3: np.int64 (natural output of argmax/center-of-mass) must pass
        the integer validation on all three geometry setters"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        o_cgc.define_parameters(
            pixel_center=np.int64(256),
            outer_radius=np.int64(200),
            inner_radius=np.int64(150),
        )
        assert o_cgc.pixel_center == 256
        assert o_cgc.outer_radius == 200
        assert o_cgc.inner_radius == 150

    def test_booleans_rejected_as_integers(self, homogeneous_tiff_files):
        """L3: True passes isinstance(_, int) but is not a valid pixel index"""
        o_cgc = GeometryCorrection(list_files=homogeneous_tiff_files)
        o_cgc.load_files()
        with pytest.raises(ValueError, match="integer"):
            o_cgc.define_parameters(pixel_center=True)

    def test_empty_file_list_raises_on_load(self):
        """L1: an empty list_files previously sailed through load_files() and
        crashed later in define_parameters with an unrelated error"""
        o_cgc = GeometryCorrection(list_files=[])
        with pytest.raises(OSError, match="empty"):
            o_cgc.load_files()

    def test_default_constructor_works(self):
        """L2: GeometryCorrection() previously raised TypeError despite the
        documented 'Default is empty list'"""
        o_cgc = GeometryCorrection()
        assert o_cgc.list_files == []

    def test_instances_do_not_share_state(self, homogeneous_tiff_files):
        """M1 hazard: list_data/list_data_corrected were class-level lists
        shared across instances"""
        a = GeometryCorrection(list_files=homogeneous_tiff_files)
        a.load_files()
        b = GeometryCorrection()
        assert b.list_data == []
        assert b.list_data is not a.list_data
        assert b.list_data_corrected is not a.list_data_corrected

    def test_homogeneous_correction_edge_symmetry(self):
        """M2: the correction factor must behave identically at both tangent
        edges; sin(arccos(-1.0)) == 1.2e-16 previously made x=-R return a
        finite 8e15 while x=+R returned NaN"""
        assert np.isnan(GeometryCorrection.homogeneous_correction(x=200, radius=200))
        assert np.isnan(GeometryCorrection.homogeneous_correction(x=-200, radius=200))
        assert GeometryCorrection.homogeneous_correction(x=201, radius=200) == 0
        assert GeometryCorrection.homogeneous_correction(x=-201, radius=200) == 0
        assert GeometryCorrection.homogeneous_correction(x=0, radius=200) == 1
        # interior values stay finite and mirror-symmetric
        left = GeometryCorrection.homogeneous_correction(x=-199, radius=200)
        right = GeometryCorrection.homogeneous_correction(x=199, radius=200)
        assert np.isfinite(left) and left == pytest.approx(right)
