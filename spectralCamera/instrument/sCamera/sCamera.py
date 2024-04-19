"""
Camera DeviceModel

Created on Mon Nov 15 12:08:51 2021

@author: ostranik
"""
#%%

import os
import time
import numpy as np
from viscope.instrument.base.baseInstrument import BaseInstrument
from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
from spectralCamera.algorithm.calibrateLoader import CalibrateLoader


class SCamera(BaseInstrument):
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
            self.spectraCalibration = CalibrateRGBImage(rgbOrder='W')
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
        self.sImage = self.imageDataToSpectralCube(self.camera.getLastImage())
        return self.sImage

    def setParameter(self,name, value):
        ''' set parameter of the spectral camera'''
        super().setParameter(name,value)

        if name== 'calibrationData':
            self.setCalibrationData(value)

        if name== 'camera':
            self.camera = value

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


    def loop(self):
        ''' infinite loop of the spectral camera thread '''
        while True:
            if self.camera.flagLoop.is_set():
                self.sImage = self.imageDataToSpectralCube(self.camera.rawImage)
                self.flagLoop.set()
                self.camera.flagLoop.clear()
                yield
            time.sleep(0.03)


#%%

if __name__ == '__main__':
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    import numpy as np
    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    sCamera = SCamera(name='RGBWebCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('threadingNow',True)

    for ii in range(5):
        sCamera.flagLoop.wait()
        print(f'worker loop reported: {ii+1} of 5')
        print(f' spectral Image sum: {np.sum(sCamera.sImage)}')
        sCamera.camera.flagLoop.clear()


    camera.disconnect()
    sCamera.disconnect()
