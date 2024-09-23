'''
class to calibrate filter based Images
'''

import numpy as np
from spectralCamera.algorithm.baseCalibrate import BaseCalibrate
from spectralCamera.instrument.camera.pfCamera.photonFocus import Photonfocus

class CalibratePFImage(BaseCalibrate):
    ''' main class to calibrate photon focus multispectral based images '''

    DEFAULT = {'darkValue':10} # average dark value of the chip

    def __init__(self, **kwargs):
        ''' initialise the class '''
        
        super().__init__(**kwargs)

        darkValue = kwargs['darkValue'] if 'darkValue' in kwargs else CalibratePFImage.DEFAULT['darkValue']
        self.pf = Photonfocus()
        self.pf.GetCalibrationData()
        self.pf.darkImage = darkValue

        self.wavelength = self.pf.pixelChar['wv']

        print(f'number of wavelength {len(self.wavelength)}')
        print(f'wavelength {self.wavelength}')



    def getSpectralImage(self,rawImage,spectralCorrection=True,**kwargs):
        ''' get the spectral image from raw image'''
        WYXImage = self.pf.imageDataToSpectralCube(rawImage,spectralCorrection=spectralCorrection)
        return  WYXImage


if __name__ == "__main__":

    pass
































