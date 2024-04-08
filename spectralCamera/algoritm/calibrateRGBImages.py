'''
class to calibrate RGB Images
'''

import numpy as np


import napari


class CalibrateRGBImage():
    ''' main class to calibrate rgb images '''

    #TODO: get proper value of the wavelength in the case of RGB filters
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

        if self.rgbOrder == 'W':
            self.wavelength = [(np.mean(np.array(self.wavelength)))]


    def getSpectralImage(self,rawImage,*args):
        ''' get the spectral image from raw image'''

        if self.rgbOrder== 'RGB': # three images next to each other
            myShape = np.shape(rawImage)
            WYXImage = np.reshape(rawImage,(myShape[0],3,-1))
            WYXImage = np.moveaxis(WYXImage, 1, 0)

        if self.rgbOrder== 'RGGB': # square color pixel RG|GB  
            myShape = [np.shape(rawImage)[0]//2,np.shape(rawImage)[1]//2]
            WYXImage = np.zeros((3,*myShape))
            WYXImage[0,:,:] = rawImage[1::2,1::2] # blue
            WYXImage[1,:,:] = (rawImage[0::2,1::2]  + rawImage[1::2,0::2]) / 2 # green
            WYXImage[2,:,:] = rawImage[0::2,0::2]  # red

        if self.rgbOrder== 'W': # black/white image
            WYXImage = rawImage[None,...]

        return  WYXImage

    def getWavelength(self):
        ''' get the RGB wavelengths '''
        return self.wavelength

if __name__ == "__main__":


    # get the image from webcam    
    from HSIplasmon.camera.webCamera import webCamera
    cam = webCamera()
    cam.prepareCamera()
    cam.setParameter('n_frames', 1)
    cam.startAcquisition()
    rawImage = cam.getLastImage()
    cam.closeCamera()
    
    # initiate the calibration class
    myCal = CalibrateRGBImage('W')
    spectralImage = myCal.getSpectralImage(rawImage)

    # show the calibrated spectral image in spectral viewer
    from HSIplasmon.SpectraViewerModel2 import SpectraViewerModel2
    wavelength = myCal.getWavelength()
    sViewer = SpectraViewerModel2(spectralImage, wavelength)
    sViewer.run()






























