"""Unit tests for the file_handler module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neutron_geomcorr import file_handler


class TestFileHandler:
    """Test suite for file I/O operations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample 2D data for testing."""
        np.random.seed(42)
        return np.random.rand(100, 100)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)


class TestFitsOperations(TestFileHandler):
    """Test FITS file operations."""

    def test_make_fits_creates_file(self, sample_data, temp_dir):
        """Test that make_fits creates a file."""
        filename = temp_dir / "test.fits"

        # Skip if astropy not installed
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")

        file_handler.make_fits(sample_data, filename)
        assert filename.exists()

    def test_make_fits_overwrite(self, sample_data, temp_dir):
        """Test overwrite behavior of make_fits."""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")

        filename = temp_dir / "test.fits"

        # Create file
        file_handler.make_fits(sample_data, filename)

        # Overwrite should work by default
        file_handler.make_fits(sample_data * 2, filename, overwrite=True)
        assert filename.exists()

        # Overwrite=False should raise error
        with pytest.raises(FileExistsError):
            file_handler.make_fits(sample_data, filename, overwrite=False)

    def test_load_fits_returns_correct_data(self, sample_data, temp_dir):
        """Test that load_fits returns the correct data."""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")

        filename = temp_dir / "test.fits"
        file_handler.make_fits(sample_data, filename)

        loaded_data = file_handler.load_fits(filename)

        assert loaded_data.shape == sample_data.shape
        assert np.allclose(loaded_data, sample_data)
        assert loaded_data.dtype == np.float64

    def test_load_fits_file_not_found(self, temp_dir):
        """Test load_fits with non-existent file."""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")

        with pytest.raises(FileNotFoundError):
            file_handler.load_fits(temp_dir / "nonexistent.fits")

    def test_load_fits_squeezes_singleton_cube(self, temp_dir):
        """L4: a (1, H, W) cube is a single 2D image and must load as one"""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")
        from astropy.io import fits

        filename = temp_dir / "cube1.fits"
        fits.PrimaryHDU(np.random.rand(1, 16, 16)).writeto(filename)

        loaded = file_handler.load_fits(filename)
        assert loaded.shape == (16, 16)

    def test_load_fits_rejects_multiframe_cube(self, temp_dir):
        """L4: the documented contract is a single 2D image"""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")
        from astropy.io import fits

        filename = temp_dir / "cube3.fits"
        fits.PrimaryHDU(np.random.rand(3, 16, 16)).writeto(filename)

        with pytest.raises(ValueError, match="2D"):
            file_handler.load_fits(filename)

    def test_make_fits_invalid_data(self, temp_dir):
        """Test make_fits with invalid data."""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")

        filename = temp_dir / "test.fits"

        # 1D array should raise error
        with pytest.raises(ValueError):
            file_handler.make_fits(np.array([1, 2, 3]), filename)

        # 3D array should raise error
        with pytest.raises(ValueError):
            file_handler.make_fits(np.random.rand(10, 10, 3), filename)


class TestTiffOperations(TestFileHandler):
    """Test TIFF file operations."""

    def test_make_tiff_creates_file(self, sample_data, temp_dir):
        """Test that make_tiff creates a file."""
        filename = temp_dir / "test.tif"
        file_handler.make_tiff(sample_data, filename)
        assert filename.exists()

    def test_make_tiff_overwrite(self, sample_data, temp_dir):
        """Test overwrite behavior of make_tiff."""
        filename = temp_dir / "test.tif"

        # Create file
        file_handler.make_tiff(sample_data, filename)

        # Overwrite should work by default
        file_handler.make_tiff(sample_data * 2, filename, overwrite=True)
        assert filename.exists()

        # Overwrite=False should raise error
        with pytest.raises(FileExistsError):
            file_handler.make_tiff(sample_data, filename, overwrite=False)

    def test_load_tiff_returns_correct_data(self, sample_data, temp_dir):
        """Test that load_tiff returns the written pixel values."""
        filename = temp_dir / "test.tif"
        file_handler.make_tiff(sample_data, filename)

        loaded_data = file_handler.load_tiff(filename)

        assert loaded_data.shape == sample_data.shape
        assert loaded_data.dtype == np.float64
        # float data is stored as float32, so the round trip is exact to
        # float32 precision — previously it came back min-max stretched
        # to [0, 65535]
        np.testing.assert_allclose(loaded_data, sample_data, rtol=1e-6)

    def test_load_tiff_file_not_found(self, temp_dir):
        """Test load_tiff with non-existent file."""
        with pytest.raises(FileNotFoundError):
            file_handler.load_tiff(temp_dir / "nonexistent.tif")

    def test_load_tiff_rejects_multiframe_stack(self, temp_dir):
        """L4: the documented contract is a single 2D image"""
        import tifffile

        filename = temp_dir / "stack.tif"
        # photometric: keep the 3-frame stack grayscale; tifffile would
        # otherwise guess RGB from the leading 3 and warn
        tifffile.imwrite(filename, np.random.rand(3, 16, 16).astype(np.float32), photometric="minisblack")
        with pytest.raises(ValueError, match="2D"):
            file_handler.load_tiff(filename)

    def test_make_tiff_invalid_data(self, temp_dir):
        """Test make_tiff with invalid data."""
        filename = temp_dir / "test.tif"

        # 1D array should raise error
        with pytest.raises(ValueError):
            file_handler.make_tiff(np.array([1, 2, 3]), filename)

        # 3D array should raise error
        with pytest.raises(ValueError):
            file_handler.make_tiff(np.random.rand(10, 10, 3), filename)

    def test_make_tiff_different_dtypes(self, temp_dir):
        """Test make_tiff with different data types."""
        filename = temp_dir / "test.tif"

        # uint8 round-trips exactly with its dtype unchanged on disk
        data_uint8 = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        file_handler.make_tiff(data_uint8, filename)
        np.testing.assert_array_equal(file_handler.load_tiff(filename), data_uint8.astype(np.float64))

        # float64 round-trips to float32 precision
        data_float = np.random.rand(50, 50)
        file_handler.make_tiff(data_float, filename, overwrite=True)
        np.testing.assert_allclose(file_handler.load_tiff(filename), data_float, rtol=1e-6)


class TestUtilityFunctions(TestFileHandler):
    """Test utility functions."""

    def test_get_supported_formats(self):
        """Test get_supported_formats returns correct dict."""
        formats = file_handler.get_supported_formats()

        assert isinstance(formats, dict)
        assert "fits" in formats
        assert "tiff" in formats
        assert isinstance(formats["fits"], bool)
        assert isinstance(formats["tiff"], bool)

    def test_validate_image_data_valid(self, sample_data):
        """Test validate_image_data with valid data."""
        # Should not raise any exception
        file_handler.validate_image_data(sample_data)

    def test_validate_image_data_invalid(self):
        """Test validate_image_data with invalid data."""
        # Not a numpy array
        with pytest.raises(ValueError, match="must be a numpy array"):
            file_handler.validate_image_data([[1, 2], [3, 4]])

        # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            file_handler.validate_image_data(np.array([1, 2, 3]))

        # 3D array
        with pytest.raises(ValueError, match="must be 2D"):
            file_handler.validate_image_data(np.random.rand(10, 10, 3))

        # Empty array
        with pytest.raises(ValueError, match="is empty"):
            file_handler.validate_image_data(np.array([]).reshape(0, 0))

        # Array with NaN
        data_with_nan = np.ones((10, 10))
        data_with_nan[5, 5] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            file_handler.validate_image_data(data_with_nan)

        # Array with inf
        data_with_inf = np.ones((10, 10))
        data_with_inf[5, 5] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            file_handler.validate_image_data(data_with_inf)

    def test_validate_image_data_allow_nan(self):
        """L4: non-finite values pass when explicitly allowed"""
        data_with_nan = np.ones((10, 10))
        data_with_nan[5, 5] = np.nan
        file_handler.validate_image_data(data_with_nan, allow_nan=True)  # no error


class TestRoundTrip(TestFileHandler):
    """Test round-trip save and load operations."""

    def test_fits_round_trip(self, sample_data, temp_dir):
        """Test FITS save and load round trip."""
        if not file_handler.HAS_ASTROPY:
            pytest.skip("astropy not installed")

        filename = temp_dir / "roundtrip.fits"

        # Save data
        file_handler.make_fits(sample_data, filename)

        # Load data
        loaded_data = file_handler.load_fits(filename)

        # Check data integrity
        assert np.allclose(sample_data, loaded_data)
        assert loaded_data.dtype == np.float64

    def test_tiff_round_trip_uint8(self, temp_dir):
        """Test TIFF save and load round trip with uint8 data."""
        # Create uint8 data
        data_uint8 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        filename = temp_dir / "roundtrip.tif"

        # Save data
        file_handler.make_tiff(data_uint8, filename)

        # Load data
        loaded_data = file_handler.load_tiff(filename)

        # Values survive exactly; load_tiff converts to float64
        assert loaded_data.dtype == np.float64
        np.testing.assert_array_equal(loaded_data, data_uint8.astype(np.float64))

    def test_nan_bearing_data_round_trips(self, temp_dir):
        """L4: legitimately corrected data can contain NaN pixels and must
        be saveable; FITS and float TIFF both store NaN natively. The
        previous validation rejected it unconditionally."""
        data = np.random.rand(32, 32)
        data[3, 7] = np.nan

        tiff_name = temp_dir / "nan.tif"
        file_handler.make_tiff(data, tiff_name)
        loaded_tiff = file_handler.load_tiff(tiff_name)
        np.testing.assert_allclose(loaded_tiff, data, rtol=1e-6)
        assert np.isnan(loaded_tiff[3, 7])

        if file_handler.HAS_ASTROPY:
            fits_name = temp_dir / "nan.fits"
            file_handler.make_fits(data, fits_name)
            loaded_fits = file_handler.load_fits(fits_name)
            np.testing.assert_allclose(loaded_fits, data)
            assert np.isnan(loaded_fits[3, 7])

        # opting back into strict validation still works
        with pytest.raises(ValueError, match="non-finite"):
            file_handler.make_tiff(data, temp_dir / "strict.tif", allow_nan=False)


class TestErrorHandling(TestFileHandler):
    """Test error handling in file operations."""

    def test_missing_astropy_import(self, temp_dir, monkeypatch):
        """Test error when astropy is not available."""
        monkeypatch.setattr(file_handler, "HAS_ASTROPY", False)

        with pytest.raises(ImportError, match="astropy is required"):
            file_handler.make_fits(np.random.rand(10, 10), temp_dir / "test.fits")

        with pytest.raises(ImportError, match="astropy is required"):
            file_handler.load_fits(temp_dir / "test.fits")

    def test_missing_tifffile_import(self, temp_dir, monkeypatch):
        """Test error when tifffile is not available."""
        monkeypatch.setattr(file_handler, "HAS_TIFFFILE", False)

        with pytest.raises(ImportError, match="tifffile is required"):
            file_handler.make_tiff(np.random.rand(10, 10), temp_dir / "test.tif")

        with pytest.raises(ImportError, match="tifffile is required"):
            file_handler.load_tiff(temp_dir / "test.tif")


class TestLoadTiffFloatData(TestFileHandler):
    """load_tiff must return pixel data for the float TIFFs this project
    itself produces.

    The previous matplotlib.image.imread path returned an RGBA uint8
    render — not the data — for float (mode 'F') TIFFs, including the
    repo's own fixtures (pinned as a characterization test before this
    fix)."""

    def test_float_fixture_loads_as_2d_data(self, tiff_data_dir):
        filename = tiff_data_dir / "homogeneous_image_px_intensity_4.tif"
        loaded = file_handler.load_tiff(filename)

        assert loaded.ndim == 2
        assert loaded.dtype == np.float64
        # the fixture encodes a thickness-proportional cylinder with center
        # value intensity * diameter; RGBA renders are capped at 255
        assert loaded.max() > 255

    def test_pil_written_float_tiff_round_trips(self, temp_dir):
        """cross-engine check: a mode-'F' TIFF written by an independent
        library (Pillow) comes back through load_tiff as the true data"""
        pil_image = pytest.importorskip("PIL.Image", reason="Pillow writes the cross-check fixture")

        data = np.linspace(0.5, 2.5, 64 * 64, dtype=np.float32).reshape(64, 64)
        filename = temp_dir / "pil_float.tif"
        pil_image.fromarray(data, mode="F").save(str(filename))

        loaded = file_handler.load_tiff(filename)
        np.testing.assert_allclose(loaded, data, rtol=1e-6)


class TestMakeTiffFloatPreservesScale(TestFileHandler):
    """make_tiff must preserve the quantitative scale of float data.

    The previous implementation min-max rescaled every float image to
    uint16 by its own extrema, silently destroying scale and offset —
    per-TOF-bin corrected stacks each got a different undisclosed
    stretch (audit M6)."""

    def test_float_data_round_trips_with_scale(self, temp_dir):
        data = np.linspace(0.5, 2.5, 64 * 64, dtype=np.float64).reshape(64, 64)
        filename = temp_dir / "float_native.tif"
        file_handler.make_tiff(data, filename)
        loaded = file_handler.load_tiff(filename)

        # native float32 on disk: values keep their physical scale
        np.testing.assert_allclose(loaded, data, rtol=1e-6)

    def test_constant_float_image_keeps_its_value(self, temp_dir):
        """the degenerate constant-image branch previously zeroed everything"""
        data = np.full((32, 32), 7.5, dtype=np.float64)
        filename = temp_dir / "constant.tif"
        file_handler.make_tiff(data, filename)
        loaded = file_handler.load_tiff(filename)

        np.testing.assert_array_equal(loaded, data)
