"""
class SCamera

@author: ostranik
"""
#%%

import os
import time
import numpy as np
from viscope.instrument.base.baseProcessor import BaseProcessor
from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
from spectralCamera.algorithm.calibrateLoader import CalibrateLoader
from spectralCamera.algorithm.fileSIVideo import FileSIVideo

class SCamera(BaseProcessor):
    ''' class to control spectral camera
    process raw data from camera to create spectral image,
    is able to save all processed spectral images
    '''
    DEFAULT = {'name': 'sCamera',
               'aberrationCorrection': False,
               'spectralCorrection':True,
               'spectraSigma': 0, # sigma for gaussian filter in spectral axis
               'darkValue': 0} # average value of dark

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= SCamera.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # bw camera
        self.camera = None
        self.spectraCalibration = None
        
        # spectralCamera parameters
        self.sImage = None
        self.wavelength = None
        self.aberrationCorrection = SCamera.DEFAULT['aberrationCorrection']
        self.spectralCorrection = SCamera.DEFAULT['spectralCorrection']
        self.spectraSigma = SCamera.DEFAULT['spectraSigma']
        self.darkValue = SCamera.DEFAULT['darkValue']
        
        self.dTime = 0 # acquisition/processing time between two spectral images
        self.t0 = time.time() # time of acquisition of the last spectral image

        # parameters for saving of images
        self.flagSaving = False
        self.fileSIVideo = None

        # get default calibration data
        self.setCalibrationData()


    def setCalibrationData(self, spectraCalibration=None):
        ''' set the calibration class '''

        if spectraCalibration is None:
            self.spectraCalibration = CalibrateRGBImage()
        else:
            if spectraCalibration is str:
                self.spectraCalibration = CalibrateLoader.load(spectraCalibration)
            else:
                self.spectraCalibration = spectraCalibration
        
        self.wavelength = self.spectraCalibration.getWavelength()

    def getWavelength(self):
        return self.wavelength        

    def imageDataToSpectralCube(self,imageData):
        ''' convert image to hyper spectral cube'''
        
        return self.spectraCalibration.getSpectralImage(imageData,
                aberrationCorrection = self.aberrationCorrection,
                spectralCorrection= self.spectralCorrection,
                spectraSigma= self.spectraSigma,
                darkValue= self.darkValue)

    def getLastSpectralImage(self):
        ''' direct call of the camera image and spectral processing of it '''
        self.sImage = self.imageDataToSpectralCube(self.camera.getLastImage())
        return self.sImage

    def connect(self,camera=None):
        ''' connect data processor with the camera '''
        if camera is not None:
            super().connect(camera.flagLoop)
            self.setParameter('camera',camera)
        else:
            super().connect()

    def setParameter(self,name, value):
        ''' set parameter of the spectral camera'''
        super().setParameter(name,value)

        if name== 'calibrationData':
            self.setCalibrationData(value)

        if name== 'camera':
            self.camera = value
            self.flagToProcess = self.camera.flagLoop

    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name=='wavelength':
            return self.getWavelength()

        if name== 'camera':
            return self.camera

        if name== 'calibrationData':
            return self.spectraCalibration

    def isRecording(self):
        ''' check if saving video sequence of images is activated'''
        return self.flagSaving

    def startRecording(self,folder):
        ''' start recording all acquired spectral images'''
        self.fileSIVideo = FileSIVideo(folder=folder)
        self.fileSIVideo.saveWavelength(self.wavelength) 
        self.flagSaving = True

    def stopRecording(self,folder):
        ''' stop recording the acquired spectral images'''
        self.flagSaving = False

    def processData(self):
        ''' process newly arrived data '''
        #print(f"processing data from {self.DEFAULT['name']}")
        self.sImage = self.imageDataToSpectralCube(self.camera.rawImage)
        self.dTime = time.time() -self.t0
        self.t0 = self.t0 + self.dTime
        if self.flagSaving:
            self.fileSIVideo.saveImage(self.sImage)

        return self.sImage


#%%

if __name__ == '__main__':
    pass
