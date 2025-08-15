"""Pytest configuration and fixtures for neutron_geomcorr tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory.

    First looks for data in tests/data, then falls back to notebooks/data
    for backwards compatibility.
    """
    tests_dir = Path(__file__).parent

    # Check if data is in tests/data (preferred location)
    tests_data = tests_dir / "data"
    if tests_data.exists():
        return tests_data

    # Fall back to notebooks/data (current location)
    project_root = tests_dir.parent
    notebooks_data = project_root / "notebooks" / "data"
    if notebooks_data.exists():
        return notebooks_data

    raise RuntimeError("Test data directory not found. Expected at tests/data or notebooks/data")


@pytest.fixture(scope="session")
def tiff_data_dir(test_data_dir):
    """Return the path to the TIFF test data directory."""
    tiff_dir = test_data_dir / "tiff"
    if not tiff_dir.exists():
        raise RuntimeError(f"TIFF data directory not found at {tiff_dir}")
    return tiff_dir


@pytest.fixture(scope="session")
def fits_data_dir(test_data_dir):
    """Return the path to the FITS test data directory."""
    fits_dir = test_data_dir / "fits"
    if not fits_dir.exists():
        raise RuntimeError(f"FITS data directory not found at {fits_dir}")
    return fits_dir


@pytest.fixture
def homogeneous_tiff_files(tiff_data_dir):
    """Return list of homogeneous TIFF test files."""
    pattern = "homogeneous*.tif"
    files = sorted(tiff_data_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No homogeneous TIFF files found matching {pattern}")
    return [str(f) for f in files]


@pytest.fixture
def inhomogeneous_tiff_files(tiff_data_dir):
    """Return list of inhomogeneous TIFF test files."""
    pattern = "inhomogeneous*.tif"
    files = sorted(tiff_data_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No inhomogeneous TIFF files found matching {pattern}")
    return [str(f) for f in files]


@pytest.fixture
def homogeneous_fits_files(fits_data_dir):
    """Return list of homogeneous FITS test files."""
    pattern = "homogeneous*.fits"
    files = sorted(fits_data_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No homogeneous FITS files found matching {pattern}")
    return [str(f) for f in files]


@pytest.fixture
def inhomogeneous_fits_files(fits_data_dir):
    """Return list of inhomogeneous FITS test files."""
    pattern = "inhomogeneous*.fits"
    files = sorted(fits_data_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"No inhomogeneous FITS files found matching {pattern}")
    return [str(f) for f in files]


@pytest.fixture
def sample_fits_file(test_data_dir):
    """Return a single FITS file for basic testing."""
    # Look for any .fits file in the main data directory
    fits_files = list(test_data_dir.glob("*.fits"))
    if fits_files:
        return str(fits_files[0])

    # Fall back to files in the fits subdirectory
    fits_subdir = test_data_dir / "fits"
    if fits_subdir.exists():
        fits_files = list(fits_subdir.glob("*.fits"))
        if fits_files:
            return str(fits_files[0])

    raise RuntimeError("No FITS files found for testing")
