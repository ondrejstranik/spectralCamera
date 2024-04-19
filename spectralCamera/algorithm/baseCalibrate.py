'''
class to calibrate RGB Images
'''

import numpy as np

class BaseCalibrate():
    ''' base class to calibrate BW image into spectral images'''

    DEFAULT = {'wavelength': np.array([550])}

    def __init__(self,wavelength=None, **kwarg):
        ''' initialise the class 
        wavelength ... wavelength'''
        
        if wavelength is None:
            self.wavelength = BaseCalibrate.DEFAULT['wavelength']
        else:
            self.wavelength = wavelength

    def getSpectralImage(self,rawImage,**kwargs):
        ''' extend only into spectral dimension'''
        WYXImage = rawImage[None,...]
        return  WYXImage

    def getWavelength(self):
        ''' get the RGB wavelengths '''
        return self.wavelength

if __name__ == "__main__":
    pass































