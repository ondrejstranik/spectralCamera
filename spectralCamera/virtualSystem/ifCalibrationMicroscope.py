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


class IFCalibrationMicroscope(BaseSystem):
    ''' class to emulate microscope '''
    DEFAULT = {}
               
    
    def __init__(self,*args, **kwargs):
        ''' initialisation '''
        super().__init__(*args, **kwargs)

        self.microArraySize = np.array([20,20])

        # set default spectral sample
        self.sample = Sample2()
        self.sample.setCalibrationImage(sampleSize=self.microArraySize)

    def setVirtualDevice(self,camera=None):
        ''' set instruments of the microscope '''
        self.device['camera'] = camera

    def calculateVirtualFrame(self):
        ''' update the virtual Frame of the camera '''

        # calibration sample onto micro-array
        iFrame=self.sample.get()
        (oFrame,lens00Position) = Component2.disperseIntoLines(iFrame)

        # image it onto camera-chip (1:1)
        # but it automatically adjust the potential different size of the dispersed image and the camera chip
        oFrame = Component.ideal4fImagingOnCamera(camera=self.device['camera'],iFrame=oFrame,
                                iFramePosition=-lens00Position,iPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'],
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

    from viscope.instrument.virtual.virtualCamera import VirtualCamera
    from viscope.main import Viscope
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    #from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope

    camera1 = VirtualCamera()
    camera1.connect()
    camera1.setParameter('threadingNow',True)

    vM = IFCalibrationMicroscope()
    vM.setVirtualDevice(camera1)
    vM.connect()

    viscope = Viscope()
    viewer  = AllDeviceGUI(viscope)
    viewer.setDevice([camera1])
    
    viscope.run()

    camera1.disconnect()
    vM.disconnect()