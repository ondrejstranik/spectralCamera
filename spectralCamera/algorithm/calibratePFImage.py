'''
class to calibrate filter based Images
'''

import numpy as np
from spectralCamera.algorithm.baseCalibrate import BaseCalibrate
from spectralCamera.instrument.camera.pfCamera.photonFocus import Photonfocus
from scipy.ndimage import gaussian_filter


class CalibratePFImage(BaseCalibrate):
    ''' main class to calibrate photon focus multispectral based images '''

    DEFAULT = {'darkValue':10, # average dark value of the chip
               'excludeWavelength': []} # [nm] bands to leave out of the spectral image

    def __init__(self, **kwargs):
        ''' initialise the class
        excludeWavelength ... list of wavelengths [nm]; the band closest to each
            of them is removed from the spectral image and the wavelength list
            (e.g. [700] - the 700 nm filter of the chip is strongly non-homogeneous).
            The raw filter pixel with the closest transmission peak is removed
            from the spectral correction too, so it does not leak into the other
            bands (changes their spectral response by < 2 %) '''

        super().__init__(**kwargs)

        self.darkValue = kwargs['darkValue'] if 'darkValue' in kwargs else CalibratePFImage.DEFAULT['darkValue']
        self.pf = Photonfocus()
        #self.pf.GetPlainCalibrationData()
        self.pf.GetCalibrationData()
        self.pf.darkImage = self.darkValue

        # bands to keep
        wv = self.pf.pixelChar['wv']
        self.bandMask = np.ones(len(wv), dtype=bool)
        excludeWavelength = kwargs.get('excludeWavelength', CalibratePFImage.DEFAULT['excludeWavelength'])
        for w in excludeWavelength:
            self.bandMask[np.argmin(np.abs(wv - w))] = False

        # remove the raw filter pixels from the spectral correction
        filterchar = self.pf.pixelChar.get('filterchar')
        if len(excludeWavelength) > 0 and 'coef_pixels' in self.pf.pixelChar and np.ndim(filterchar) == 2:
            filterPeak = self.pf.pixelChar['w'][np.argmax(filterchar, axis=0)]
            coef = self.pf.pixelChar['coef_pixels'].copy()
            for w in excludeWavelength:
                coef[np.argmin(np.abs(filterPeak - w))] = 0
            self.pf.pixelChar['coef_pixels'] = coef

        self.wavelength = wv[self.bandMask]

        #print(f'number of wavelength {len(self.wavelength)}')
        #print(f'wavelength {self.wavelength}')



    def getSpectralImage(self,rawImage,
                         spectralCorrection=True,
                         darkValue=None,
                           **kwargs):
        ''' get the spectral image from raw image'''
        
        if darkValue is not None:
            self.darkValue = darkValue
            self.pf.darkImage = self.darkValue

        WYXImage = self.pf.imageDataToSpectralCube(rawImage,
                                                   spectralCorrection=spectralCorrection)

        # TODO: temporary!! smoothing
        # move to sCamera
        #if spectraSigma >0:
        #    smoothWYXImage = gaussian_filter(WYXImage, sigma=spectraSigma, axes=0)
        #else:
        #    smoothWYXImage = WYXImage

        #return smoothWYXImage

        # remove the excluded bands (getattr - for pickled classes without bandMask)
        bandMask = getattr(self, 'bandMask', None)
        if bandMask is not None and not bandMask.all():
            WYXImage = WYXImage[bandMask]

        return  WYXImage


if __name__ == "__main__":

    pass
































