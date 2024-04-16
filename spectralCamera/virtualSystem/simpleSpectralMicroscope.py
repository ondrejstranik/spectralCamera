"""
virtual basic microscope

components: camera

@author: ostranik
"""
#%%

import time

from viscope.virtualSystem.base.baseSystem import BaseSystem
from viscope.virtualSystem.component.component import Component
from spectralCamera.virtualSystem.component.component2 import Component2
from spectralCamera.virtualSystem.component.sample2 import Sample2

import numpy as np


class SimpleSpectralMicroscope(BaseSystem):
    ''' class to emulate microscope '''
    DEFAULT = {}
               
    
    def __init__(self,*args, **kwargs):
        ''' initialisation '''
        super().__init__(*args, **kwargs)

        # set default spectral sample
        self.sample = Sample2()
        self.sample.setSpectralAstronaut()

    def setVirtualDevice(self,camera=None):
        ''' set instruments of the microscope '''
        self.device['camera'] = camera

    def calculateVirtualFrame(self):
        ''' update the virtual Frame of the camera '''

        # image sample onto dispersive element
        # which is only a part of the camera chip size
        iFrame=self.sample.get()
        oFrame = np.zeros((iFrame.shape[0],self.device['camera'].getParameter('height'),
                    self.device['camera'].getParameter('width')//iFrame.shape[0]))
        Component.ideal4fImaging(iFrame=iFrame,oFrame=oFrame,iFramePosition = np.array([0,0]),
                        magnification=1,iPixelSize=self.sample.pixelSize,oPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'])
        # disperse the image horizontally 
        # it will disperse the channel into single wavelength images aligned horizontally 
        oFrame = Component2.disperseHorizontal(oFrame)

        # image it onto camera-chip (1:1)
        # but it automatically adjust the potential different size of the dispersed image and the camera chip
        oFrame = Component.ideal4fImagingOnCamera(camera=self.device['camera'],iFrame=oFrame,
                                iFramePosition=np.array([0,0]),iPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'],
                                magnification=1)

        print('virtual Frame updated')

        return oFrame


    def loop(self):
        ''' infinite loop to carry out the microscope state update
        it is a state machine, which should be run in separate thread '''
        while True:
            yield 
            if self.deviceParameterIsChanged():
                print(f'calculate virtual frame')
                self.device['camera'].virtualFrame = self.calculateVirtualFrame()
                self.deviceParameterFlagClear()

            time.sleep(0.03)

        

#%%

if __name__ == '__main__':

    pass