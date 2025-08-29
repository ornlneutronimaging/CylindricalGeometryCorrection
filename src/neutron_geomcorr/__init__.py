"""
neutron-geomcorr: Neutron imaging geometric correction for cylindrical samples.

This package provides tools for correcting geometric distortions in neutron
transmission imaging of cylindrical samples. It handles both homogeneous (solid)
and inhomogeneous (hollow) cylindrical geometries.

Modules
-------
geometry_correction
    Core module containing the GeometryCorrection class for performing
    cylindrical geometry corrections.
file_handler
    File I/O utilities for reading and writing FITS and TIFF files.

Classes
-------
GeometryCorrection
    Main class for cylindrical geometry correction of neutron images.

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
>>> from neutron_geomcorr import GeometryCorrection
>>> gc = GeometryCorrection(list_files=['sample.tif'])
>>> gc.load_files()
>>> gc.define_parameters(pixel_center=256, outer_radius=100)
>>> gc.correct()
>>> corrected_images = gc.list_data_corrected

>>> from neutron_geomcorr import load_tiff, make_fits
>>> data = load_tiff('neutron_image.tif')
>>> # Process data...
>>> make_fits(data, 'processed.fits')
"""

from neutron_geomcorr.file_handler import (
    get_supported_formats,
    load_fits,
    load_tiff,
    make_fits,
    make_tiff,
    validate_image_data,
)
from neutron_geomcorr.geometry_correction import GeometryCorrection

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "GeometryCorrection",
    "__version__",
    "make_fits",
    "make_tiff",
    "load_fits",
    "load_tiff",
    "get_supported_formats",
    "validate_image_data",
]
