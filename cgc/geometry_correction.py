import os
import numpy as np

from NeuNorm.normalization import Normalization


class GeometryCorrection():

    list_files = []
    list_data = []
    list_data_corrected = []

    step1 = False # load
    step2 = False # parameters definition

    _outer_radius = np.NaN
    _inner_radius = np.NaN

    def __init__(self, list_files=[]):
        self.list_files = list_files

    def run(self, notebook=False, pixel_center=np.NaN, outer_radius=np.NaN, inner_radius=np.NaN):
        '''run the full process without having to call the steps one by one

        Parameters:
            notebook: boolean - to display or not a progress bar when loading the file via notebook (default True)
            pixel_center: int - center of cylinder (default np.NaN)
            outer_radius: int - radius of cylinder (or outer cylinder for inhomogeneous sample) (default np.NaN)
            inner_radius: int - radius of outer cylinder (default is np.NaN)

        Raises:
            ValueError if pixel_center is not >0 int
            ValueError if outer_radius is not >0 int
            ValueError if inner_radius is not >0 int (unless np.NaN)
            ValueError if pixel_center outside image size
            ValueError if outer_radius define cylinder outside image
            ValueError if inner_radius define cylinder outside image

        '''
        self.load_files(notebook=notebook)
        self.define_parameters(pixel_center=pixel_center, outer_radius=outer_radius, inner_radius=inner_radius)

    @property
    def list_files(self):
        return self._list_files

    @list_files.setter
    def list_files(self, list_files):
        # list_files should be a list
        if not isinstance(list_files, list):
            raise TypeError('List of Files should be a list')

        # make sure the file exist
        for _file in list_files:
            if not os.path.exists(_file):
                raise ValueError("File {} does not exist!".format(_file))

        self._list_files = list_files

    @property
    def pixel_center(self):
        return self._pixel_center

    @pixel_center.setter
    def pixel_center(self, pixel_center):
        if self.step1 is False:
            raise AttributeError("Please define the list of files first by running the 'load_files' method!")

        if not isinstance(pixel_center, int):
            raise ValueError("Pixel center must be an integer!")

        [_, width] = np.shape(self.list_data[0])
        if (pixel_center <= 0) or (pixel_center >= width):
            raise ValueError("Pixel center must be inside the image!")

        self._pixel_center = pixel_center

    @property
    def outer_radius(self):
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, outer_radius):
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
    def inner_radius(self):
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, inner_radius):
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

    def load_files(self, notebook=False):
        """loading the tiff or fits images

        step1

        Parameters:
            notebook: boolean - to display or not a progress bar when loading the file via notebook (default True)
        """
        o_norm = Normalization()
        o_norm.load(file=self.list_files, notebook=notebook)
        self.list_data = o_norm.data['sample']['data']
        self.step1 = True
        del o_norm

    def define_parameters(self, pixel_center=np.NaN, outer_radius=np.NaN, inner_radius=np.NaN):
        '''define the center of the cylinder and the radius 1 and optionally 2 if working with inhomogeneous sample

        step2

        Parameters:
            pixel_center: int - center of cylinder in the horizontal direction. Algorithm consider that the
                vertical axis is symmetrical (default np.NaN)
            outer_radius: int - radius of cylinder (or outer cylinder for inhomogeneous sample) (default np.NaN)
            inner_radius: int - radius of outer cylinder (default is np.NaN)

        Raises:
            ValueError if pixel_center is not >0 int
            ValueError if outer_radius is not >0 int
            ValueError if inner_radius is not >0 int (unless np.NaN)
            ValueError if pixel_center outside image size
            ValueError if outer_radius define cylinder outside image
            ValueError if inner_radius define cylinder outside image
        '''
        self.pixel_center = pixel_center
        self.outer_radius = outer_radius
        if not np.isnan(inner_radius):
            self.inner_radius = inner_radius

    def get_sample_thickness_at_center(self):
        """Depending on if we are working with homogeneous or inhomogeneous samples, this algorithm calculates the
        sample thickness. For an homogeneous sample, it's simply 2 times the outer_radius. For an inhomogenous sample,
        the sample thickness is 2 times (radius_outer_cylinder - radius_inner_cylinder)"""
        if np.isnan(self.inner_radius):
            return 2 * self.outer_radius
        else:
            return 2 * (self.outer_radius - self.inner_radius)

    def calculate_pixel_intensity(self, slice=[]):
        """return the intensity of each pixel by using the radius and pixel_center info

        pixel_intensity is simply the value of the slice at the center position divided by the diameter

        Parameters:
            slice: array - slice of the image

        Returns:
            the pixel intensity value
        """
        if slice == []:
            raise ValueError("Slice is empty!")

        _effective_diameter = self.get_sample_thickness_at_center()
        return slice[self.pixel_center] / _effective_diameter

    def isolate_cylinder_from_image(self, index=0):
        """Isolate the cylinder from the rest of the image

        Parameters:
              index: int - file index (default 0, first file)

        Returns:
              image cropped using pixel center and outer radius information
        """
        _image = self.list_data[index]
        _pixel_center = self.pixel_center
        _outer_radius = self.outer_radius
        return _image[: , _pixel_center - _outer_radius : _pixel_center + _outer_radius + 1]

    def _correct_file_index(self, index=0):
        """main algorithm that correct geometry of file index specified

        Parameters:
            index: int - file index (default 0, first file)

        Returns:
            2D of image corrected

        """
        # _image = self.list_data[index]
        _image = self.isolate_cylinder_from_image(index=index)

        [height, width] = np.shape(_image)
        corrected_image = np.zeros((height, width))
        for _slice_index in np.arange(height):
            _slice = _image[_slice_index, :]
            # _pixel_intensity = self.calculate_pixel_intensity(slice=_slice)
            for _index_pixel, _pixel in enumerate(np.arange(-self.outer_radius, self.outer_radius+1)):
                _intensity_of_pixel = _slice[_index_pixel]
                _coeff = self.general_correction(x=_pixel)
                _corrected_value = (_intensity_of_pixel * _coeff ) / 2.0
                corrected_image[_slice_index, _index_pixel] = _corrected_value

        # remove first pixel
        corrected_image = corrected_image[:, 1:-1]
        absolute_radius = self.get_sample_thickness_at_center()/2

        return corrected_image/absolute_radius

    def correct(self, notebook=False):
        """main algorithm that is going to correct the cylindrical geometry

        Parameters:
            notebook: boolean - display or not progress bar showing progress of correction (default False)

        """
        if notebook:
            from ipywidgets import widgets
            from IPython.core.display import display

            progress_ui = widgets.IntProgress(max=len(self.list_files),
                                              description = 'Progress:')
            display(progress_ui)

        self.list_data_corrected = []
        for _index, _file in enumerate(self.list_data):
            self.list_data_corrected.append(self._correct_file_index(_index))

            if notebook:
                progress_ui.value = _index + 1

        if notebook:
            progress_ui.close()

    def general_correction(self, x=0):
        if np.isnan(self._inner_radius):
            return GeometryCorrection.homogeneous_correction(x=x, radius=self._outer_radius)
        else:
            return GeometryCorrection.inhomogeneous_correction(x=x, inner_radius=self._inner_radius,
                                                               outer_radius=self._outer_radius)

    @staticmethod
    def homogeneous_correction(x=0, radius=np.NaN):
        if np.abs(x) > radius:
            return 0
        rp = 2 * radius * np.sin(np.arccos(x / radius))
        if x == 0:
            return 1
        if rp == 0:
            return np.NaN
        return (2 * radius) / rp

    @staticmethod
    def inhomogeneous_correction(x=0, inner_radius=np.NaN, outer_radius=np.NaN):
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