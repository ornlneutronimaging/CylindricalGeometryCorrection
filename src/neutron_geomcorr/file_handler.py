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
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    tifffile = None


def make_fits(
    data: NDArray[np.number], filename: Union[str, Path], overwrite: bool = True, allow_nan: bool = True
) -> None:
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
    allow_nan : bool, optional
        Whether to accept non-finite values (NaN/inf). Default is True:
        FITS stores NaN natively, and legitimately corrected data can
        contain NaN pixels. Set to False to reject such data.

    Raises
    ------
    ImportError
        If astropy is not installed.
    FileExistsError
        If file exists and overwrite is False.
    ValueError
        If data is not a 2D array, or contains non-finite values while
        allow_nan is False.

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

    validate_image_data(data, allow_nan=allow_nan)

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


def make_tiff(
    data: NDArray[np.number], filename: Union[str, Path], overwrite: bool = True, allow_nan: bool = True
) -> None:
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
    allow_nan : bool, optional
        Whether to accept non-finite values (NaN/inf). Default is True:
        float TIFF stores NaN natively, and legitimately corrected data
        can contain NaN pixels. Set to False to reject such data.

    Raises
    ------
    ImportError
        If tifffile is not installed.
    FileExistsError
        If file exists and overwrite is False.
    ValueError
        If data is not a 2D array, or contains non-finite values while
        allow_nan is False.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    >>> make_tiff(data, 'output.tif')

    Notes
    -----
    Float data is written as a native 32-bit float TIFF, preserving the
    quantitative scale (and staying readable by ImageJ, which has no
    64-bit float support); integer data is written with its dtype
    unchanged. Earlier versions min-max rescaled float data to uint16
    per image, silently discarding scale and offset.
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF file operations. Install it with: pip install tifffile")

    validate_image_data(data, allow_nan=allow_nan)

    filename = Path(filename)

    if filename.exists():
        if overwrite:
            filename.unlink()
        else:
            raise FileExistsError(f"File {filename} already exists")

    if np.issubdtype(data.dtype, np.floating):
        tifffile.imwrite(filename, data.astype(np.float32))
    else:
        tifffile.imwrite(filename, data)


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
        If the FITS file does not contain valid image data, or the data
        is not a single 2D image after squeezing singleton dimensions.

    Examples
    --------
    >>> data = load_fits('image.fits')
    >>> print(data.shape)
    (512, 512)

    Notes
    -----
    This function loads the primary HDU data from the FITS file.
    For multi-HDU FITS files, only the first HDU is loaded. Singleton
    dimensions (e.g. a (1, H, W) cube) are squeezed away; anything that
    is not a single 2D image afterwards is rejected, enforcing the
    documented return contract.
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

        image = np.squeeze(np.asarray(hdu.data, dtype=np.float64))

    if image.ndim != 2:
        raise ValueError(f"{filename} does not contain a single 2D image (shape after squeezing: {image.shape})")

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
        If tifffile is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is not a single 2D image after squeezing singleton
        dimensions.

    Examples
    --------
    >>> data = load_tiff('image.tif')
    >>> print(data.shape)
    (512, 512)

    Notes
    -----
    The file is read with tifffile and the returned array is always
    converted to float64. Earlier versions read through
    matplotlib.image.imread, which returns an RGBA uint8 render — not
    the pixel data — for float (mode 'F') TIFFs.
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF file operations. Install it with: pip install tifffile")

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found")

    image = np.squeeze(np.asarray(tifffile.imread(str(filename)), dtype=np.float64))

    if image.ndim != 2:
        raise ValueError(f"{filename} does not contain a single 2D image (shape after squeezing: {image.shape})")

    return image


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
    return {"fits": HAS_ASTROPY, "tiff": HAS_TIFFFILE}


def validate_image_data(data: NDArray, allow_nan: bool = False) -> None:
    """
    Validate that data is suitable for image I/O operations.

    Parameters
    ----------
    data : ndarray
        Array to validate.
    allow_nan : bool, optional
        Whether to accept non-finite values (NaN/inf). Default is False,
        preserving the historical strictness for direct callers; the
        writers pass True because FITS and float TIFF store NaN natively.

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

    if not allow_nan and not np.isfinite(data).all():
        raise ValueError("Data contains non-finite values (inf or nan)")
