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
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        filename = temp_dir / "test.tif"
        file_handler.make_tiff(sample_data, filename)
        assert filename.exists()

    def test_make_tiff_overwrite(self, sample_data, temp_dir):
        """Test overwrite behavior of make_tiff."""
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        filename = temp_dir / "test.tif"

        # Create file
        file_handler.make_tiff(sample_data, filename)

        # Overwrite should work by default
        file_handler.make_tiff(sample_data * 2, filename, overwrite=True)
        assert filename.exists()

        # Overwrite=False should raise error
        with pytest.raises(FileExistsError):
            file_handler.make_tiff(sample_data, filename, overwrite=False)

    def test_load_tiff_returns_data(self, sample_data, temp_dir):
        """Test that load_tiff returns data."""
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        filename = temp_dir / "test.tif"
        file_handler.make_tiff(sample_data, filename)

        loaded_data = file_handler.load_tiff(filename)

        assert loaded_data.shape == sample_data.shape
        assert loaded_data.dtype == np.float64
        # Note: TIFF conversion may lose precision due to uint16 conversion

    def test_load_tiff_file_not_found(self, temp_dir):
        """Test load_tiff with non-existent file."""
        if not (file_handler.HAS_MATPLOTLIB or file_handler.HAS_PIL):
            pytest.skip("Neither matplotlib nor Pillow installed")

        with pytest.raises(FileNotFoundError):
            file_handler.load_tiff(temp_dir / "nonexistent.tif")

    def test_make_tiff_invalid_data(self, temp_dir):
        """Test make_tiff with invalid data."""
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        filename = temp_dir / "test.tif"

        # 1D array should raise error
        with pytest.raises(ValueError):
            file_handler.make_tiff(np.array([1, 2, 3]), filename)

        # 3D array should raise error
        with pytest.raises(ValueError):
            file_handler.make_tiff(np.random.rand(10, 10, 3), filename)

    def test_make_tiff_different_dtypes(self, temp_dir):
        """Test make_tiff with different data types."""
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        filename = temp_dir / "test.tif"

        # Test with uint8
        data_uint8 = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        file_handler.make_tiff(data_uint8, filename)
        assert filename.exists()

        # Test with float64
        data_float = np.random.rand(50, 50)
        file_handler.make_tiff(data_float, filename, overwrite=True)
        assert filename.exists()


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
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        # Create uint8 data
        data_uint8 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        filename = temp_dir / "roundtrip.tif"

        # Save data
        file_handler.make_tiff(data_uint8, filename)

        # Load data
        loaded_data = file_handler.load_tiff(filename)

        # Check shape and type
        assert loaded_data.shape == data_uint8.shape
        assert loaded_data.dtype == np.float64


class TestErrorHandling(TestFileHandler):
    """Test error handling in file operations."""

    def test_missing_astropy_import(self, temp_dir, monkeypatch):
        """Test error when astropy is not available."""
        monkeypatch.setattr(file_handler, "HAS_ASTROPY", False)

        with pytest.raises(ImportError, match="astropy is required"):
            file_handler.make_fits(np.random.rand(10, 10), temp_dir / "test.fits")

        with pytest.raises(ImportError, match="astropy is required"):
            file_handler.load_fits(temp_dir / "test.fits")

    def test_missing_pil_import(self, temp_dir, monkeypatch):
        """Test error when PIL is not available."""
        monkeypatch.setattr(file_handler, "HAS_PIL", False)

        with pytest.raises(ImportError, match="Pillow is required"):
            file_handler.make_tiff(np.random.rand(10, 10), temp_dir / "test.tif")

    def test_missing_both_tiff_libs(self, temp_dir, monkeypatch):
        """Test error when neither matplotlib nor PIL is available for TIFF."""
        # Create a dummy file first
        dummy_file = temp_dir / "test.tif"
        dummy_file.write_text("dummy")

        monkeypatch.setattr(file_handler, "HAS_MATPLOTLIB", False)
        monkeypatch.setattr(file_handler, "HAS_PIL", False)

        with pytest.raises(ImportError, match="matplotlib or Pillow"):
            file_handler.load_tiff(dummy_file)


class TestLoadTiffPilBranch(TestFileHandler):
    """Pin the relationship between load_tiff's two code paths."""

    def test_pil_fallback_matches_matplotlib_for_uint16(self, temp_dir, monkeypatch):
        """assert both load_tiff code paths agree on integer TIFFs"""
        if not (file_handler.HAS_MATPLOTLIB and file_handler.HAS_PIL):
            pytest.skip("both matplotlib and Pillow required for this comparison")
        from PIL import Image

        data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        filename = temp_dir / "u16.tif"
        Image.fromarray(data).save(str(filename))

        via_matplotlib = file_handler.load_tiff(filename)
        monkeypatch.setattr(file_handler, "HAS_MATPLOTLIB", False)
        via_pil = file_handler.load_tiff(filename)

        assert via_pil.dtype == np.float64
        np.testing.assert_allclose(via_pil, via_matplotlib, rtol=1e-6)

    def test_matplotlib_path_mangles_float_tiff(self, tiff_data_dir, monkeypatch):
        """Characterization of a real bug: the default (matplotlib) path
        returns an RGBA uint8 render -- not the data -- for the float
        (mode 'F') TIFFs this project itself produces and ships as fixtures.
        The PIL fallback returns the true 2D float data. Tracked for the
        file_handler fix PR; this test pins today's broken behavior so the
        fix is explicit.
        """
        if not (file_handler.HAS_MATPLOTLIB and file_handler.HAS_PIL):
            pytest.skip("both matplotlib and Pillow required for this comparison")

        filename = tiff_data_dir / "homogeneous_image_px_intensity_4.tif"
        via_matplotlib = file_handler.load_tiff(filename)
        monkeypatch.setattr(file_handler, "HAS_MATPLOTLIB", False)
        via_pil = file_handler.load_tiff(filename)

        # the PIL path returns the documented contract (2D data)...
        assert via_pil.ndim == 2
        # ...while the default path currently violates it (RGBA, 0-255)
        assert via_matplotlib.ndim == 3
        assert via_matplotlib.shape[-1] == 4


class TestMakeTiffFloatNormalization(TestFileHandler):
    """Characterization: make_tiff min-max rescales float data to uint16.

    The original scale and offset are discarded (every image gets its own
    stretch), which silently destroys the quantitative scale of corrected
    transmission data. Pinned here so the planned fix (native float TIFF)
    changes this consciously.
    """

    def test_float_data_is_minmax_rescaled_to_uint16(self, temp_dir):
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        data = np.linspace(0.5, 2.5, 64 * 64, dtype=np.float64).reshape(64, 64)
        filename = temp_dir / "float_rescaled.tif"
        file_handler.make_tiff(data, filename)
        loaded = file_handler.load_tiff(filename)

        # quantitative values are gone: full min-max stretch to [0, 65535]
        assert loaded.min() == 0.0
        assert loaded.max() == 65535.0
        expected = ((data - data.min()) / (data.max() - data.min()) * 65535).astype(np.uint16)
        np.testing.assert_allclose(loaded, expected.astype(np.float64), atol=1.0)

    def test_constant_float_image_becomes_all_zeros(self, temp_dir):
        """assert the degenerate constant-image branch zeroes everything"""
        if not file_handler.HAS_PIL:
            pytest.skip("Pillow not installed")

        data = np.full((32, 32), 7.5, dtype=np.float64)
        filename = temp_dir / "constant.tif"
        file_handler.make_tiff(data, filename)
        loaded = file_handler.load_tiff(filename)

        np.testing.assert_array_equal(loaded, np.zeros((32, 32), dtype=np.float64))
