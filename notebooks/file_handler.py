from astropy.io import fits
from PIL import Image
import os
import numpy as np
from PIL import Image


def make_fits(data=[], filename=''):
    '''create fits file'''
    fits.writeto(filename, data, overwrite=True)


def make_tiff(data=[], filename=''):
    if os.path.exists(filename):
        os.remove(filename)
    _image = Image.fromarray(data)
    _image.save(filename)


def load_fits(filename=""):
    '''load fits image

    Parameters
    ----------
       full file name of fits image
    '''
    try:
        tmp = fits.open(filename, ignore_missing_end=True)[0].data
        if len(tmp.shape) == 3:
            tmp = tmp.reshape(tmp.shape[1:])
        return tmp
    except OSError:
        raise OSError("Unable to read the FITS file provided!")


def load_tiff(filename=''):
    try:
        _image = Image.open(filename)
        # metadata = dict(_image.tag_v2)
        data = np.asarray(_image)
        _image.close()
        return data
    except OSError:
        raise OSError("Unable to read the TIFF file provided!")
