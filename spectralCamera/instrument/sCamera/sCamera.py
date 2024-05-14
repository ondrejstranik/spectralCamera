"""
spectral camera - Data Processor 

Created on Mon Nov 15 12:08:51 2021

@author: ostranik
"""
#%%

import os
import time
import numpy as np
from viscope.instrument.base.baseProcessor import BaseProcessor
from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
from spectralCamera.algorithm.calibrateLoader import CalibrateLoader


class SCamera(BaseProcessor):
    ''' class to control spectral camera'''
    DEFAULT = {'name': 'sCamera'}

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
        
        return self.spectraCalibration.getSpectralImage(imageData)

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

    def processData(self):
        ''' process newly arrived data '''
        print(f"processing data from {self.DEFAULT['name']}")
        self.sImage = self.imageDataToSpectralCube(self.camera.rawImage)
        return self.sImage


#%%

if __name__ == '__main__':
    pass
