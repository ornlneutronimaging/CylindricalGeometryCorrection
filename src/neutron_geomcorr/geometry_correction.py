"""
Cylindrical geometry correction for neutron transmission imaging.

This module provides functionality to correct for the non-uniform path length
through cylindrical samples in neutron transmission imaging. It handles both
homogeneous (solid) and inhomogeneous (hollow) cylindrical samples.

The correction algorithm compensates for the varying neutron path length through
a cylinder when viewed in transmission geometry, where the edges appear brighter
than the center due to the shorter path length.

Classes
-------
GeometryCorrection
    Main class for performing cylindrical geometry corrections on neutron
    transmission images.

Notes
-----
The cylinder must be positioned vertically (along the image height axis) for
the correction to work properly.

References
----------
.. [1] Kockelmann, W., et al. "Neutron imaging in materials science."
       Physics Reports 718 (2018): 1-34.

Examples
--------
>>> from neutron_geomcorr.geometry_correction import GeometryCorrection
>>> # Load and correct homogeneous cylinder images
>>> gc = GeometryCorrection(list_files=['image1.tif', 'image2.tif'])
>>> gc.load_files()
>>> gc.define_parameters(pixel_center=256, outer_radius=100)
>>> gc.correct()
>>> corrected_data = gc.list_data_corrected
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray


class GeometryCorrection:
    """
    Perform cylindrical geometry correction on neutron transmission images.

    This class handles the correction of neutron transmission images of
    cylindrical samples, compensating for the varying path length through
    the cylinder at different positions.

    Parameters
    ----------
    list_files : list of str, optional
        List of file paths to neutron transmission images to be corrected.
        Default is empty list.

    Attributes
    ----------
    list_files : list of str
        List of input file paths.
    list_data : list of ndarray
        List of loaded image data arrays.
    list_data_corrected : list of ndarray
        List of corrected image data arrays.
    pixel_center : int
        Horizontal pixel position of the cylinder center.
    outer_radius : int
        Radius of the cylinder (or outer radius for hollow cylinders).
    inner_radius : int or float
        Inner radius for hollow cylinders (np.nan for solid cylinders).

    Examples
    --------
    >>> gc = GeometryCorrection(list_files=['sample.tif'])
    >>> gc.load_files()
    >>> gc.define_parameters(pixel_center=256, outer_radius=100)
    >>> gc.correct()
    """

    list_data: Optional[List[NDArray[np.float64]]] = []
    list_data_corrected: List[NDArray[np.float64]] = []

    step1: bool = False  # load
    step2: bool = False  # parameters definition

    _outer_radius: float = np.nan
    _inner_radius: float = np.nan
    _pixel_center: int = 0
    _list_files: List[str] = []

    def __init__(self, list_files: Optional[List[str]] = None) -> None:
        """
        Initialize the GeometryCorrection object.

        Parameters
        ----------
        list_files : list of str, optional
            List of file paths to neutron transmission images.
            Default is empty list.
        """
        self.list_files = list_files

    def run(
        self,
        notebook: bool = False,
        pixel_center: Union[int, float] = np.nan,
        outer_radius: Union[int, float] = np.nan,
        inner_radius: Union[int, float] = np.nan,
    ) -> None:
        """
        Run the full correction process.

        This method loads files and applies the cylindrical geometry correction
        in one call.

        Parameters
        ----------
        notebook : bool, optional
            Whether to display a progress bar (for Jupyter notebooks).
            Default is False.
        pixel_center : int
            Horizontal pixel position of the cylinder center.
        outer_radius : int
            Radius of the cylinder (or outer radius for hollow cylinders).
        inner_radius : int, optional
            Inner radius for hollow cylinders. Default is np.nan (solid cylinder).

        Raises
        ------
        ValueError
            If pixel_center is not a positive integer within image bounds.
        ValueError
            If outer_radius is not a positive integer or defines cylinder
            outside image bounds.
        ValueError
            If inner_radius is not a positive integer (when specified) or
            defines cylinder outside image bounds.

        Examples
        --------
        >>> gc = GeometryCorrection(list_files=['sample.tif'])
        >>> gc.run(pixel_center=256, outer_radius=100)
        """
        self.load_files(notebook=notebook)
        self.define_parameters(pixel_center=pixel_center, outer_radius=outer_radius, inner_radius=inner_radius)

    @property
    def list_files(self) -> List[str]:
        """Get the list of input file paths."""
        return self._list_files

    @list_files.setter
    def list_files(self, list_files: List[str]) -> None:
        """
        Set and validate the list of input files.

        Parameters
        ----------
        list_files : list of str
            List of file paths to validate and set.

        Raises
        ------
        TypeError
            If list_files is not a list.
        ValueError
            If any file in the list does not exist.
        """
        import os

        # list_files should be a list
        if not isinstance(list_files, list):
            raise TypeError("List of Files should be a list")

        # make sure the file exist
        for _file in list_files:
            if not os.path.exists(_file):
                raise ValueError(f"File {_file} does not exist!")

        self._list_files = list_files

    @property
    def pixel_center(self) -> int:
        """Get the horizontal pixel position of cylinder center."""
        return self._pixel_center

    @pixel_center.setter
    def pixel_center(self, pixel_center: int) -> None:
        """
        Set and validate the cylinder center position.

        Parameters
        ----------
        pixel_center : int
            Horizontal pixel position of cylinder center.

        Raises
        ------
        AttributeError
            If data has not been loaded yet.
        ValueError
            If pixel_center is not an integer or is outside image bounds.
        """
        if self.step1 is False:
            raise AttributeError("Please define the list of files first by running the 'load_files' method!")

        if not isinstance(pixel_center, int):
            raise ValueError("Pixel center must be an integer!")

        [_, width] = np.shape(self.list_data[0])
        if (pixel_center <= 0) or (pixel_center >= width):
            raise ValueError("Pixel center must be inside the image!")

        self._pixel_center = pixel_center

    @property
    def outer_radius(self) -> float:
        """Get the outer radius of the cylinder."""
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, outer_radius: int) -> None:
        """
        Set and validate the outer radius.

        Parameters
        ----------
        outer_radius : int
            Radius of cylinder (or outer radius for hollow cylinders).

        Raises
        ------
        ValueError
            If outer_radius is not a positive integer or defines cylinder
            outside image bounds.
        """
        if not isinstance(outer_radius, int):
            raise ValueError("Radius 1 must be an integer!")

        if outer_radius <= 0:
            raise ValueError("Radius 1 must be greater than 0!")

        [_, width] = np.shape(self.list_data[0])
        if (self.pixel_center - outer_radius) < 0:
            raise ValueError("Cylinder defined by Radius 1 goes outside the image size (left side)!")

        if (self.pixel_center + outer_radius) >= width:
            raise ValueError("Cylinder defined by Radius 1 goes outside the image size (right side)!")

        if np.isnan(self._inner_radius):
            self._outer_radius = outer_radius
        else:
            if self._inner_radius > outer_radius:
                self._outer_radius, self._inner_radius = self._inner_radius, outer_radius
            else:
                self._outer_radius = outer_radius

    @property
    def inner_radius(self) -> float:
        """Get the inner radius of hollow cylinder."""
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, inner_radius: int) -> None:
        """
        Set and validate the inner radius for hollow cylinders.

        Parameters
        ----------
        inner_radius : int
            Inner radius of hollow cylinder.

        Raises
        ------
        ValueError
            If inner_radius is not a positive integer or defines cylinder
            outside image bounds.
        """
        if not isinstance(inner_radius, int):
            raise ValueError("Radius 2 must be an integer!")

        if inner_radius <= 0:
            raise ValueError("Radius 2 must be greater than 0!")

        [_, width] = np.shape(self.list_data[0])
        if (self.pixel_center - inner_radius) < 0:
            raise ValueError("Cylinder defined by Radius 2 goes outside the image size (left side)!")

        if (self.pixel_center + inner_radius) >= width:
            raise ValueError("Cylinder defined by Radius 2 goes outside the image size (right side)!")

        if self._outer_radius > inner_radius:
            self._inner_radius = inner_radius
        else:
            self._outer_radius, self._inner_radius = inner_radius, self._outer_radius

    # general method

    def load_files(self, notebook: bool = False) -> None:
        """
        Load image files into memory.

        Parameters
        ----------
        notebook : bool, optional
            Whether to display a progress bar (for Jupyter notebooks).
            Default is False.

        Raises
        ------
        OSError
            If a file has an unsupported extension (TIFF/FITS only) or its
            shape does not match the previously loaded images.

        Notes
        -----
        TIFF files are read with tifffile and FITS files with astropy.
        All images are returned as float32 and squeezed to 2D, and every
        image must have the same shape. After loading, sets step1 flag
        to True.
        """
        if notebook:
            from IPython.display import display
            from ipywidgets import widgets

            progress_ui = widgets.IntProgress(max=len(self.list_files), description="Progress:")
            display(progress_ui)

        list_data = []
        for _index, _file in enumerate(self.list_files):
            _lower = str(_file).lower()
            if _lower.endswith((".tif", ".tiff")):
                import tifffile

                _data = tifffile.imread(_file)
            elif _lower.endswith((".fits", ".fit", ".fts")):
                from astropy.io import fits

                with fits.open(_file, ignore_missing_end=True) as hdulist:
                    _data = hdulist[0].data
            else:
                raise OSError(f"File format of {_file} is not supported (TIFF/FITS only)")

            _image = np.squeeze(np.asarray(_data, dtype=np.float32))
            if list_data and _image.shape != list_data[0].shape:
                raise OSError("Shape of sample does not match previously loaded data set!")
            list_data.append(_image)

            if notebook:
                progress_ui.value = _index + 1

        if notebook:
            progress_ui.close()

        self.list_data = list_data
        self.step1 = True

    def define_parameters(
        self,
        pixel_center: Union[int, float] = np.nan,
        outer_radius: Union[int, float] = np.nan,
        inner_radius: Union[int, float] = np.nan,
    ) -> None:
        """
        Define cylinder geometry parameters.

        Parameters
        ----------
        pixel_center : int
            Horizontal pixel position of cylinder center.
        outer_radius : int
            Radius of cylinder (or outer radius for hollow cylinders).
        inner_radius : int, optional
            Inner radius for hollow cylinders. Default is np.nan (solid cylinder).

        Raises
        ------
        ValueError
            If any parameter is invalid or defines cylinder outside image bounds.

        Notes
        -----
        For hollow cylinders, the method automatically ensures outer_radius > inner_radius
        by swapping values if necessary.
        """
        self.pixel_center = pixel_center
        self.outer_radius = outer_radius
        if not np.isnan(inner_radius):
            self.inner_radius = inner_radius

    def get_sample_thickness_at_center(self) -> float:
        """
        Calculate the sample thickness at the cylinder center.

        Returns
        -------
        float
            Sample thickness at center. For solid cylinders, returns 2 * outer_radius.
            For hollow cylinders, returns 2 * (outer_radius - inner_radius).

        Examples
        --------
        >>> gc.define_parameters(pixel_center=256, outer_radius=100)
        >>> thickness = gc.get_sample_thickness_at_center()  # Returns 200
        """
        if np.isnan(self.inner_radius):
            return 2 * self.outer_radius
        else:
            return 2 * (self.outer_radius - self.inner_radius)

    def calculate_pixel_intensity(self, slice: Optional[NDArray[np.float64]] = None) -> float:  # noqa: A002
        """
        Calculate the pixel intensity for normalization.

        Parameters
        ----------
        slice : ndarray
            1D array representing a horizontal slice of the image.

        Returns
        -------
        float
            Normalized pixel intensity at the center position.

        Raises
        ------
        ValueError
            If slice is empty.
        """
        if len(slice) == 0:
            raise ValueError("Slice is empty!")

        _effective_diameter = self.get_sample_thickness_at_center()
        return slice[self.pixel_center] / _effective_diameter

    def isolate_cylinder_from_image(self, index: int = 0) -> NDArray[np.float64]:
        """
        Extract the cylinder region from an image.

        Parameters
        ----------
        index : int, optional
            Index of the image in list_data. Default is 0.

        Returns
        -------
        ndarray
            2D array containing only the cylinder region.

        Examples
        --------
        >>> cylinder_region = gc.isolate_cylinder_from_image(0)
        """
        _image = self.list_data[index]
        _pixel_center = self.pixel_center
        _outer_radius = self.outer_radius
        return _image[:, _pixel_center - _outer_radius : _pixel_center + _outer_radius + 1]

    def _correct_file_index(self, index: int = 0) -> NDArray[np.float64]:
        """
        Apply correction to a single image.

        Parameters
        ----------
        index : int, optional
            Index of the image to correct. Default is 0.

        Returns
        -------
        ndarray
            Corrected 2D image array.
        """
        _image = self.isolate_cylinder_from_image(index=index)

        [height, width] = np.shape(_image)
        corrected_image = np.zeros((height, width))
        for _slice_index in np.arange(height):
            _slice = _image[_slice_index, :]
            # _pixel_intensity = self.calculate_pixel_intensity(slice=_slice)
            for _index_pixel, _pixel in enumerate(np.arange(-self.outer_radius, self.outer_radius + 1)):
                _intensity_of_pixel = _slice[_index_pixel]
                _coeff = self.general_correction(x=_pixel)
                _corrected_value = (_intensity_of_pixel * _coeff) / 2.0
                corrected_image[_slice_index, _index_pixel] = _corrected_value

        # remove first pixel
        corrected_image = corrected_image[:, 1:-1]
        absolute_radius = self.get_sample_thickness_at_center() / 2

        return corrected_image / absolute_radius

    def correct(self, notebook: bool = False) -> None:
        """
        Apply cylindrical geometry correction to all loaded images.

        Parameters
        ----------
        notebook : bool, optional
            Whether to display a progress bar (for Jupyter notebooks).
            Default is False.

        Notes
        -----
        Results are stored in the list_data_corrected attribute.
        """
        if notebook:
            from IPython.display import display
            from ipywidgets import widgets

            progress_ui = widgets.IntProgress(max=len(self.list_files), description="Progress:")
            display(progress_ui)

        self.list_data_corrected = []
        for _index, _file in enumerate(self.list_data):
            self.list_data_corrected.append(self._correct_file_index(_index))

            if notebook:
                progress_ui.value = _index + 1

        if notebook:
            progress_ui.close()

    def general_correction(self, x: float = 0) -> float:
        """
        Calculate the correction factor for a given position.

        Parameters
        ----------
        x : float, optional
            Horizontal position relative to cylinder center. Default is 0.

        Returns
        -------
        float
            Correction factor to apply at position x.
        """
        if np.isnan(self._inner_radius):
            return GeometryCorrection.homogeneous_correction(x=x, radius=self._outer_radius)
        else:
            return GeometryCorrection.inhomogeneous_correction(
                x=x, inner_radius=self._inner_radius, outer_radius=self._outer_radius
            )

    @staticmethod
    def homogeneous_correction(x: float = 0, radius: float = np.nan) -> float:
        """
        Calculate correction factor for solid cylinders.

        Parameters
        ----------
        x : float, optional
            Horizontal position relative to cylinder center. Default is 0.
        radius : float
            Cylinder radius.

        Returns
        -------
        float
            Correction factor for solid cylinder at position x.
        """
        if np.abs(x) > radius:
            return 0
        rp = 2 * radius * np.sin(np.arccos(x / radius))
        if x == 0:
            return 1
        if rp == 0:
            return np.nan
        return (2 * radius) / rp

    @staticmethod
    def inhomogeneous_correction(x: float = 0, inner_radius: float = np.nan, outer_radius: float = np.nan) -> float:
        """
        Calculate correction factor for hollow cylinders.

        Parameters
        ----------
        x : float, optional
            Horizontal position relative to cylinder center. Default is 0.
        inner_radius : float
            Inner radius of hollow cylinder.
        outer_radius : float
            Outer radius of hollow cylinder.

        Returns
        -------
        float
            Correction factor for hollow cylinder at position x.
        """

        def factor_inho(x: float = 0, inner_radius: float = np.nan, outer_radius: float = np.nan) -> float:
            r = np.abs(x)
            if r >= outer_radius:
                return 0
            elif (r >= inner_radius) and (r <= outer_radius):
                return 2 * outer_radius * np.sin(np.arccos(x / outer_radius))
            else:
                rp1 = 2 * inner_radius * np.sin(np.arccos(x / inner_radius))
                rp2 = 2 * outer_radius * np.sin(np.arccos(x / outer_radius))
                rp = rp2 - rp1
                return rp

        _value = factor_inho(x=x, inner_radius=inner_radius, outer_radius=outer_radius)
        if x == 0:
            return 1
        if _value == 0:
            _value = np.nan

        return 2 * (outer_radius - inner_radius) / _value
