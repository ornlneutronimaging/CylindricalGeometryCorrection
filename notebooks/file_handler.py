import pyfits
from PIL import Image
import os
import numpy as np
import matplotlib.image as mpimg


def make_fits(data=[], filename=''):
    if os.path.exists(filename):
        os.remove(filename)
    hdu = pyfits.PrimaryHDU(data)
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(filename)
    hdulist.close()
    
def make_tiff(data=[], filename=''):
    if os.path.exists(filename):
        os.remove(filename)
    _image = Image.fromarray(data)
    _image.save(filename)
    
def load_fits(filename=''):
    hdu_list = pyfits.open(filename)
    hdu = hdu_list[0]
    _image = hdu.data
    _image = np.asarray(_image)
    hdu_list.close()
    return _image

def load_tiff(filename=''):
    _image = mpimg.imread(filename)
    return _image