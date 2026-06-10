"""Contract tests for GeometryCorrection.load_files().

Originally written against NeuNorm 1.x to pin the load contract; updated by
the loader-replacement PR (NeuNorm -> direct tifffile/astropy I/O) to encode
the new, normalized contract. Deliberate behavior changes from 1.x:

- dtype is uniformly float32 for both formats (1.x: float32 TIFF but raw
  big-endian float64 FITS)
- the auto gamma filter is gone: saturated integer pixels load unchanged
  (1.x silently neighbor-averaged them)
- ``.fit``/``.fts`` extensions are accepted in addition to ``.fits``
"""

import numpy as np
import pytest

from neutron_geomcorr import file_handler
from neutron_geomcorr.geometry_correction import GeometryCorrection

# I/O helpers for these contract tests; skip the module cleanly (instead of
# erroring at collection) in environments where they are not installed
tifffile = pytest.importorskip("tifffile", reason="tifffile is the independent TIFF reader for contract checks")
Image = pytest.importorskip("PIL.Image", reason="Pillow writes the synthetic TIFF fixtures")


def _load_tiff_independent(path):
    """Independent TIFF reader for contract checks.

    Deliberately tifffile, NOT file_handler.load_tiff: the latter's default
    matplotlib path returns an RGBA uint8 array for the float (mode 'F')
    TIFFs this project uses (tracked for the file_handler fix PR).
    """
    return np.asarray(tifffile.imread(str(path)), dtype=np.float64)


class TestLoadContract:
    """Loaded pixel values must match an independent reader.

    Exactly for FITS (both paths go through astropy); within float32
    round-off for TIFF, where load_files() yields float32 and the
    independent reader compares in float64.
    """

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


class TestLoadDtypeContract:
    """Both formats load as native float32.

    Deliberate normalization vs NeuNorm 1.x, which returned float32 for TIFF
    but raw big-endian float64 for FITS. Note for full-precision float64 FITS
    inputs this is a (documented) precision reduction.
    """

    def test_tiff_loads_as_float32(self, tiff_data_dir):
        o_cgc = GeometryCorrection(list_files=[str(tiff_data_dir / "homogeneous_image_px_intensity_4.tif")])
        o_cgc.load_files()
        assert o_cgc.list_data[0].dtype == np.float32

    def test_fits_loads_as_float32(self, fits_data_dir):
        o_cgc = GeometryCorrection(list_files=[str(fits_data_dir / "homogeneous_image_px_intensity_4.fits")])
        o_cgc.load_files()
        assert o_cgc.list_data[0].dtype == np.float32


class TestLoadHiddenSemantics:
    """Pin the loader's validation and pass-through semantics.

    Some behaviors are carried over from NeuNorm 1.x (shape-mismatch
    rejection, squeezing), others deliberately replace 1.x behavior
    (verbatim pass-through with no gamma filter, multi-frame rejection).
    """

    def test_saturated_uint16_pixel_loads_unchanged(self, tmp_path):
        """assert pixel values are passed through verbatim, even at saturation

        Deliberate change vs NeuNorm 1.x, whose auto gamma filter silently
        neighbor-averaged saturated integer pixels on load. Saturation
        handling is now the caller's responsibility.
        """
        image = np.full((32, 32), 100, dtype=np.uint16)
        image[10, 12] = 65535
        path = tmp_path / "saturated.tif"
        Image.fromarray(image).save(str(path))

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        np.testing.assert_array_equal(np.asarray(o_cgc.list_data[0]), image.astype(np.float32))

    def test_unsaturated_integer_image_loads_unchanged(self, tmp_path):
        """assert integer TIFF data loads verbatim (as float32)"""
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

    def test_unsupported_extension_raises(self, tmp_path):
        """assert non-TIFF/FITS files are rejected up front"""
        path = tmp_path / "image.png"
        path.write_bytes(b"not really a png")

        o_cgc = GeometryCorrection(list_files=[str(path)])
        with pytest.raises(OSError, match="not supported"):
            o_cgc.load_files()

    def test_single_frame_3d_fits_is_squeezed(self, tmp_path):
        """assert (1, H, W) FITS collapses to 2D, matching 1.x behavior"""
        from astropy.io import fits as astropy_fits

        data = np.arange(16 * 16, dtype=np.float64).reshape(1, 16, 16)
        path = tmp_path / "stack1.fits"
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(data)]).writeto(str(path))

        o_cgc = GeometryCorrection(list_files=[str(path)])
        o_cgc.load_files()
        loaded = np.asarray(o_cgc.list_data[0])
        assert loaded.shape == (16, 16)
        np.testing.assert_array_equal(loaded, data[0].astype(np.float32))

    def test_multi_frame_stack_is_rejected(self, tmp_path):
        """assert files that stay 3D after squeezing fail fast

        Downstream code assumes 2D images; 1.x rejected multi-frame FITS
        inside NeuNorm, and the direct loader must not silently regress to
        passing 3D arrays through.
        """
        from astropy.io import fits as astropy_fits

        data = np.arange(2 * 16 * 16, dtype=np.float64).reshape(2, 16, 16)
        path = tmp_path / "stack2.fits"
        astropy_fits.HDUList([astropy_fits.PrimaryHDU(data)]).writeto(str(path))

        o_cgc = GeometryCorrection(list_files=[str(path)])
        with pytest.raises(OSError, match="2D"):
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
