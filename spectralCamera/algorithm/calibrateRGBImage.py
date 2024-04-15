'''
class to calibrate RGB Images
'''

import numpy as np


import napari


class CalibrateRGBImage():
    ''' main class to calibrate rgb images '''

    DEFAULT = {'wavelength': [400,550,610],
               'rgbOrder' : 'RGB' }

    def __init__(self,rgbOrder=None, wavelength=None):
        ''' initialise the class 
        rgbOrder ... string defining the order of RGB channels
        wavelength ... list with wavelength'''
        
        if rgbOrder is None:
            self.rgbOrder = self.DEFAULT['rgbOrder']
        else:
            self.rgbOrder = rgbOrder     

        if wavelength is None:
            self.wavelength = self.DEFAULT['wavelength']
        else:
            self.wavelength = wavelength

        #if self.rgbOrder == 'W':
        #    self.wavelength = [(np.mean(np.array(self.wavelength)))]


    def getSpectralImage(self,rawImage,*args):
        ''' get the spectral image from raw image'''

        if self.rgbOrder== 'RGB': # three images next to each other
            myShape = np.shape(rawImage)
            if rawImage.shape[1]%3 == 0:
                WYXImage = np.reshape(rawImage,(myShape[0],3,-1))
            else:
                WYXImage = np.reshape(rawImage[:,0:-(rawImage.shape[1]%3)],(myShape[0],3,-1))

            WYXImage = np.moveaxis(WYXImage, 1, 0)

        if self.rgbOrder== 'RGGB': # square color pixel RG|GB  
            myShape = [np.shape(rawImage)[0]//2,np.shape(rawImage)[1]//2]
            WYXImage = np.zeros((3,*myShape))
            WYXImage[0,:,:] = rawImage[1::2,1::2] # blue
            WYXImage[1,:,:] = (rawImage[0::2,1::2]  + rawImage[1::2,0::2]) / 2 # green
            WYXImage[2,:,:] = rawImage[0::2,0::2]  # red

        #if self.rgbOrder== 'W': # black/white image
        #    WYXImage = rawImage[None,...]

        return  WYXImage

    def getWavelength(self):
        ''' get the RGB wavelengths '''
        return self.wavelength

if __name__ == "__main__":
    pass































