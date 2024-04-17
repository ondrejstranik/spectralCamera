"""
virtual basic microscope

components: camera

@author: ostranik
"""
#%%

import time

from viscope.virtualSystem.base.baseSystem import BaseSystem
from viscope.virtualSystem.component.component import Component
from spectralCamera.virtualSystem.component.sample2 import Sample2
from spectralCamera.virtualSystem.component.component2 import Component2
import numpy as np


class MultiSpectralMicroscope(BaseSystem):
    ''' class to emulate microscope '''
    DEFAULT = {}
               
    
    def __init__(self,*args, **kwargs):
        ''' initialisation '''
        super().__init__(*args, **kwargs)

        # set default spectral sample
        self.sample = Sample2()
        self.sample.setSpectralDisk()

    def setVirtualDevice(self,sCamera=None, camera2=None):
        ''' set instruments of the microscope '''
        self.device['sCamera'] = sCamera
        self.device['camera'] = sCamera.camera
        self.device['camera2'] = camera2


    def calculateVirtualFrameCamera(self):
        ''' update the virtual Frame of the spectral camera '''
        
        # image sample onto dispersive element
        iFrame=self.sample.get()

        # horizontal dispersion (RGB)
        # it will disperse the channel into single wavelength images aligned horizontally 
        if self.device['sCamera'].spectraCalibration.__class__.__name__ =='CalibrateRGBImage' and (
            self.device['sCamera'].spectraCalibration.rgbOrder == 'RGB'):
            
            # adjust wavelength
            iFrame = Component2.spectraRangeAdjustment(iFrame,self.sample.getWavelength(),self.device['sCamera'].getWavelength())
            oFrame = np.zeros((iFrame.shape[0],self.device['camera'].getParameter('height'),
                        self.device['camera'].getParameter('width')//iFrame.shape[0]))
            Component.ideal4fImaging(iFrame=iFrame,oFrame=oFrame,iFramePosition = np.array([0,0]),
                            magnification=1,iPixelSize=self.sample.pixelSize,oPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'])
            oFrame = Component2.disperseHorizontal(oFrame)

        # RGGB dispersion onto super-pixels
        if self.device['sCamera'].spectraCalibration.__class__.__name__ =='CalibrateRGBImage' and (
            self.device['sCamera'].spectraCalibration.rgbOrder == 'RGGB'):
            # adjust wavelength
            iFrame = Component2.spectraRangeAdjustment(iFrame,self.sample.getWavelength(),self.device['sCamera'].getWavelength())

            oFrame = np.zeros((iFrame.shape[0],self.device['camera'].getParameter('height')//2,
                        self.device['camera'].getParameter('width')//2))
            Component.ideal4fImaging(iFrame=iFrame,oFrame=oFrame,iFramePosition = np.array([0,0]),
                            magnification=1,iPixelSize=self.sample.pixelSize,oPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'])
            oFrame = Component2.disperseIntoRGGBBlock(oFrame)
        # dispersion on square super-pixel
        if self.device['sCamera'].spectraCalibration.__class__.__name__== 'CalibrateFilterImage':
            # adjust wavelength
            iFrame = Component2.spectraRangeAdjustment(iFrame,self.sample.getWavelength(),self.device['sCamera'].getWavelength())
            _order = self.device['sCamera'].spectraCalibration.order 
            oFrame = np.zeros((iFrame.shape[0],self.device['camera'].getParameter('height')//_order,
                        self.device['camera'].getParameter('width')//_order))
            Component.ideal4fImaging(iFrame=iFrame,oFrame=oFrame,iFramePosition = np.array([0,0]),
                            magnification=1,iPixelSize=self.sample.pixelSize,oPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'])
            oFrame = Component2.disperseIntoBlock(oFrame, blockShape=np.array([_order,_order]))

        # image it onto camera-chip
        # convinient way to crop not full superpixel
        oFrame = Component.ideal4fImagingOnCamera(camera=self.device['camera'],iFrame=oFrame,
                                iFramePosition=np.array([0,0]),iPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'],
                                magnification=1)


        print('virtual Frame updated')

        return oFrame

    def calculateVirtualFrameCamera2(self):
        ''' update the virtual Frame of the nofilter camera '''
        
        # image sample onto dispersive element
        iFrame=self.sample.get()
        
        # sum wavelength
        iFrame = np.sum(iFrame, axis= 0)

        # image it onto camera-chip
        oFrame = Component.ideal4fImagingOnCamera(camera=self.device['camera2'],iFrame=iFrame,
                                iFramePosition=np.array([0,0]),iPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'],
                                magnification=1)


        print('virtual Frame updated')

        return oFrame



    def loop(self):
        ''' infinite loop to carry out the microscope state update
        it is a state machine, which should be run in separate thread '''
        while True:
            yield 
            if self.device['camera'].flagSetParameter.is_set():
                print(f'calculate virtual frame - camera ')
                self.device['camera'].virtualFrame = self.calculateVirtualFrameCamera()
                self.device['camera'].flagSetParameter.clear()
            if self.device['camera2'].flagSetParameter.is_set():
                print(f'calculate virtual frame - camera2')
                self.device['camera2'].virtualFrame = self.calculateVirtualFrameCamera2()
                self.device['camera2'].flagSetParameter.clear()

            time.sleep(0.03)

        

#%%

if __name__ == '__main__':

pass