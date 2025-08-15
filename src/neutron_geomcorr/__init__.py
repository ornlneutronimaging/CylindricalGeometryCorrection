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

Classes
-------
GeometryCorrection
    Main class for cylindrical geometry correction of neutron images.

Examples
--------
>>> from neutron_geomcorr import GeometryCorrection
>>> gc = GeometryCorrection(list_files=['sample.tif'])
>>> gc.load_files()
>>> gc.define_parameters(pixel_center=256, outer_radius=100)
>>> gc.correct()
>>> corrected_images = gc.list_data_corrected
"""

from neutron_geomcorr.geometry_correction import GeometryCorrection

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["GeometryCorrection", "__version__"]
