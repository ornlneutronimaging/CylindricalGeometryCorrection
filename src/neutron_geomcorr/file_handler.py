"""
File I/O utilities for neutron imaging data.

This module provides functions for reading and writing FITS and TIFF files
commonly used in neutron imaging experiments. It handles the conversion
between numpy arrays and standard image formats.

Functions
---------
make_fits
    Create a FITS file from numpy array data.
make_tiff
    Create a TIFF file from numpy array data.
load_fits
    Load a FITS file and return as numpy array.
load_tiff
    Load a TIFF file and return as numpy array.

Examples
--------
>>> import numpy as np
>>> from neutron_geomcorr.file_handler import make_fits, load_fits
>>> data = np.random.rand(512, 512)
>>> make_fits(data, 'test.fits')
>>> loaded_data = load_fits('test.fits')
>>> assert np.allclose(data, loaded_data)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

# Optional imports with helpful error messages
try:
    from astropy.io import fits

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    fits = None

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import matplotlib.image as mpimg

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    mpimg = None


def make_fits(data: NDArray[np.float64], filename: Union[str, Path], overwrite: bool = True) -> None:
    """
    Create a FITS file from numpy array data.

    Parameters
    ----------
    data : ndarray
        2D array of image data to save.
    filename : str or Path
        Path to the output FITS file.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is True.

    Raises
    ------
    ImportError
        If astropy is not installed.
    FileExistsError
        If file exists and overwrite is False.
    ValueError
        If data is not a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 100)
    >>> make_fits(data, 'output.fits')

    Notes
    -----
    This function requires the astropy package to be installed.
    Install it with: pip install astropy
    """
    if not HAS_ASTROPY:
        raise ImportError("astropy is required for FITS file operations. Install it with: pip install astropy")

    validate_image_data(data)

    filename = Path(filename)

    if filename.exists():
        if overwrite:
            filename.unlink()
        else:
            raise FileExistsError(f"File {filename} already exists")

    hdu = fits.PrimaryHDU(data.astype(np.float64))
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename)
    hdulist.close()


def make_tiff(data: NDArray[np.float64], filename: Union[str, Path], overwrite: bool = True) -> None:
    """
    Create a TIFF file from numpy array data.

    Parameters
    ----------
    data : ndarray
        2D array of image data to save.
    filename : str or Path
        Path to the output TIFF file.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is True.

    Raises
    ------
    ImportError
        If PIL/Pillow is not installed.
    FileExistsError
        If file exists and overwrite is False.
    ValueError
        If data is not a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    >>> make_tiff(data, 'output.tif')

    Notes
    -----
    This function requires the Pillow package to be installed.
    Install it with: pip install Pillow

    The data will be converted to appropriate dtype for TIFF format.
    For floating point data, consider scaling to 0-65535 range for
    16-bit TIFF or 0-255 for 8-bit TIFF.
    """
    if not HAS_PIL:
        raise ImportError("Pillow is required for TIFF file operations. Install it with: pip install Pillow")

    validate_image_data(data)

    filename = Path(filename)

    if filename.exists():
        if overwrite:
            filename.unlink()
        else:
            raise FileExistsError(f"File {filename} already exists")

    # Handle different data types appropriately
    if data.dtype == np.float64 or data.dtype == np.float32:
        # Convert float to uint16 for better TIFF compatibility
        # Normalize to 0-65535 range
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data_normalized = (data - data_min) / (data_max - data_min)
            data_uint16 = (data_normalized * 65535).astype(np.uint16)
        else:
            data_uint16 = np.zeros_like(data, dtype=np.uint16)
        image = Image.fromarray(data_uint16)
    else:
        # For integer types, use as-is
        image = Image.fromarray(data)

    image.save(str(filename))


def load_fits(filename: Union[str, Path]) -> NDArray[np.float64]:
    """
    Load a FITS file and return as numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the FITS file to load.

    Returns
    -------
    ndarray
        2D array of image data from the FITS file.

    Raises
    ------
    ImportError
        If astropy is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the FITS file does not contain valid image data.

    Examples
    --------
    >>> data = load_fits('image.fits')
    >>> print(data.shape)
    (512, 512)

    Notes
    -----
    This function loads the primary HDU data from the FITS file.
    For multi-HDU FITS files, only the first HDU is loaded.
    """
    if not HAS_ASTROPY:
        raise ImportError("astropy is required for FITS file operations. Install it with: pip install astropy")

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found")

    with fits.open(filename) as hdu_list:
        if len(hdu_list) == 0:
            raise ValueError(f"No HDUs found in {filename}")

        hdu = hdu_list[0]
        if hdu.data is None:
            raise ValueError(f"No data found in primary HDU of {filename}")

        image = np.asarray(hdu.data, dtype=np.float64)

    return image


def load_tiff(filename: Union[str, Path]) -> NDArray[np.float64]:
    """
    Load a TIFF file and return as numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the TIFF file to load.

    Returns
    -------
    ndarray
        2D array of image data from the TIFF file.

    Raises
    ------
    ImportError
        If neither matplotlib nor PIL/Pillow is installed.
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> data = load_tiff('image.tif')
    >>> print(data.shape)
    (512, 512)

    Notes
    -----
    This function attempts to use matplotlib.image.imread first,
    falling back to PIL if matplotlib is not available.
    The returned array is always converted to float64.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found")

    if HAS_MATPLOTLIB:
        # Use matplotlib if available
        image = mpimg.imread(str(filename))
    elif HAS_PIL:
        # Fall back to PIL
        with Image.open(filename) as img:
            image = np.array(img)
    else:
        raise ImportError(
            "Either matplotlib or Pillow is required for TIFF file operations. "
            "Install with: pip install matplotlib or pip install Pillow"
        )

    # Ensure consistent float64 output
    return np.asarray(image, dtype=np.float64)


def get_supported_formats() -> dict[str, bool]:
    """
    Check which file formats are supported based on installed packages.

    Returns
    -------
    dict
        Dictionary indicating support for each format.
        Keys are 'fits' and 'tiff', values are bool.

    Examples
    --------
    >>> formats = get_supported_formats()
    >>> if formats['fits']:
    ...     print("FITS support is available")
    """
    return {"fits": HAS_ASTROPY, "tiff": HAS_MATPLOTLIB or HAS_PIL}


def validate_image_data(data: NDArray) -> None:
    """
    Validate that data is suitable for image I/O operations.

    Parameters
    ----------
    data : ndarray
        Array to validate.

    Raises
    ------
    ValueError
        If data is not a 2D array or contains invalid values.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 100)
    >>> validate_image_data(data)  # No error
    >>> data_3d = np.random.rand(100, 100, 3)
    >>> validate_image_data(data_3d)  # Raises ValueError
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got {data.ndim}D array")

    if data.size == 0:
        raise ValueError("Data array is empty")

    if not np.isfinite(data).all():
        raise ValueError("Data contains non-finite values (inf or nan)")
