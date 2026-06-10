"""Characterization tests pinning the load contract before the NeuNorm removal.

These tests document the exact observable behavior of
``GeometryCorrection.load_files()`` under NeuNorm 1.x so that the planned
loader replacement (NeuNorm -> direct tifffile/astropy I/O) can demonstrate
behavior parity. Tests that pin known 1.x quirks (per-format dtypes, the
auto gamma filter on saturated integer data) are expected to be revisited
*consciously* by the loader-replacement PR rather than silently inherited.
"""

import numpy as np
import pytest
import tifffile
from PIL import Image

from neutron_geomcorr import file_handler
from neutron_geomcorr.geometry_correction import GeometryCorrection


def _load_tiff_independent(path):
    """Independent TIFF reader for contract checks.

    Deliberately tifffile, NOT file_handler.load_tiff: the latter's default
    matplotlib path returns an RGBA uint8 array for the float (mode 'F')
    TIFFs this project uses (tracked for the file_handler fix PR).
    """
    return np.asarray(tifffile.imread(str(path)), dtype=np.float64)


class TestLoadContract:
    """Loaded pixel values must match an independent reader, byte for byte."""

    @pytest.mark.parametrize(
        "subdir, filename",
        [
            ("tiff", "homogeneous_image_px_intensity_4.tif"),
            ("tiff", "inhomogeneous_image_px_intensity_4.tif"),
            ("fits", "homogeneous_image_px_intensity_4.fits"),
            ("fits", "inhomogeneous_image_px_intensity_4.fits"),
        ],
    )
    def test_loaded_values_match_independent_reader(self, test_data_dir, subdir, filename):
        """assert load_files() returns the same pixel values as an independent reader"""
        path = test_data_dir / subdir / filename
        independent_loader = _load_tiff_independent if subdir == "tiff" else file_handler.load_fits

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()

        assert len(o_cgc.list_data) == 1
        loaded = np.asarray(o_cgc.list_data[0])
        assert loaded.ndim == 2

        expected = independent_loader(path)
        np.testing.assert_allclose(
            loaded.astype(np.float64),
            expected,
            rtol=1e-6,
            atol=0,
            err_msg=f"load_files() pixel values diverge from independent reader for {filename}",
        )


class TestLoadDtypeCharacterization:
    """Pin the (inconsistent) per-format dtypes NeuNorm 1.x produces.

    TIFF goes through the auto gamma filter and comes back float32; FITS hits
    an exception path inside the gamma filter and is returned unchanged as
    big-endian float64. The loader-replacement PR may deliberately normalize
    this -- these tests exist so that change is explicit, not accidental.
    """

    def test_tiff_loads_as_float32(self, tiff_data_dir):
        o_cgc = GeometryCorrection(list_files=[str(tiff_data_dir / "homogeneous_image_px_intensity_4.tif")])
        o_cgc.load_files()
        assert o_cgc.list_data[0].dtype == np.float32

    def test_fits_loads_as_big_endian_float64(self, fits_data_dir):
        o_cgc = GeometryCorrection(list_files=[str(fits_data_dir / "homogeneous_image_px_intensity_4.fits")])
        o_cgc.load_files()
        dtype = o_cgc.list_data[0].dtype
        assert dtype.kind == "f"
        assert dtype.itemsize == 8
        assert dtype.byteorder == ">"


class TestLoadHiddenSemantics:
    """Pin validation and filtering behaviors inherited from NeuNorm 1.x."""

    def test_saturated_uint16_pixel_is_replaced(self, tmp_path):
        """assert the 1.x auto gamma filter replaces saturated integer pixels"""
        image = np.full((32, 32), 100, dtype=np.uint16)
        image[10, 12] = 65535
        path = tmp_path / "saturated.tif"
        Image.fromarray(image).save(str(path))

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        loaded = np.asarray(o_cgc.list_data[0])

        # the saturated pixel is replaced by something neighborhood-like
        assert loaded[10, 12] != 65535
        assert np.isfinite(loaded[10, 12])
        # everything else is untouched
        untouched = np.ones_like(image, dtype=bool)
        untouched[10, 12] = False
        np.testing.assert_array_equal(loaded[untouched], np.full(untouched.sum(), 100, dtype=loaded.dtype))

    def test_unsaturated_integer_image_loads_unchanged(self, tmp_path):
        """assert the gamma filter is a no-op away from saturation"""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 1000, size=(32, 32), dtype=np.uint16)
        path = tmp_path / "unsaturated.tif"
        Image.fromarray(image).save(str(path))

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        np.testing.assert_array_equal(np.asarray(o_cgc.list_data[0]), image.astype(np.float32))

    def test_mismatched_shapes_raise(self, tmp_path):
        """assert loading files of different shapes raises instead of silently mixing"""
        path_a = tmp_path / "a.tif"
        path_b = tmp_path / "b.tif"
        Image.fromarray(np.full((32, 32), 5, dtype=np.uint16)).save(str(path_a))
        Image.fromarray(np.full((16, 16), 5, dtype=np.uint16)).save(str(path_b))

        o_cgc = GeometryCorrection(list_files=[str(path_a), str(path_b)])
        with pytest.raises(OSError):
            o_cgc.load_files()


class TestOrientationCanary:
    """A corner marker must land exactly where it was written.

    All shipped fixtures are row-uniform vertical cylinders, so a vertical
    flip (the classic FITS origin bug) is invisible to every other test in
    the suite. This canary catches flips and transposes in either format.
    """

    MARKER_ROW, MARKER_COL = 3, 7
    SIZE = 32
    BACKGROUND, MARKER = 100.0, 999.0

    def _assert_marker_position(self, loaded):
        loaded = np.asarray(loaded, dtype=np.float64)
        assert loaded.shape == (self.SIZE, self.SIZE)
        assert loaded[self.MARKER_ROW, self.MARKER_COL] == self.MARKER
        # transpose guard
        assert loaded[self.MARKER_COL, self.MARKER_ROW] == self.BACKGROUND
        # vertical-flip guard
        assert loaded[self.SIZE - 1 - self.MARKER_ROW, self.MARKER_COL] == self.BACKGROUND
        # horizontal-flip guard
        assert loaded[self.MARKER_ROW, self.SIZE - 1 - self.MARKER_COL] == self.BACKGROUND

    def test_tiff_orientation(self, tmp_path):
        image = np.full((self.SIZE, self.SIZE), int(self.BACKGROUND), dtype=np.uint16)
        image[self.MARKER_ROW, self.MARKER_COL] = int(self.MARKER)
        path = tmp_path / "marker.tif"
        Image.fromarray(image).save(str(path))

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        self._assert_marker_position(o_cgc.list_data[0])

    def test_fits_orientation(self, tmp_path):
        image = np.full((self.SIZE, self.SIZE), self.BACKGROUND, dtype=np.float64)
        image[self.MARKER_ROW, self.MARKER_COL] = self.MARKER
        path = tmp_path / "marker.fits"
        file_handler.make_fits(image, path)

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        self._assert_marker_position(o_cgc.list_data[0])
