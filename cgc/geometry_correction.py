import os


class GeometryCorrection():

    def __init__(self, list_files=[]):
        self.list_files = list_files

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
