import os

from NeuNorm.normalization import Normalization


class GeometryCorrection():

    list_files = []
    list_data = []

    def __init__(self, list_files=[]):
        self.list_files = list_files

    def run(self, notebook=False):
        '''run the full process without having to call the steps one by one

        Parameters:
            notebook: boolean - to display or not a progress bar when loading the file via notebook (default True)
        '''
        self.load_files(notebook=notebook)

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

    def load_files(self, notebook=False):
        """loading the tiff or fits images

        Parameters:
            notebook: boolean - to display or not a progress bar when loading the file via notebook (default True)
        """
        o_norm = Normalization()
        o_norm.load(file=self.list_files, notebook=notebook)
        self.list_data = o_norm.data['sample']['data']

