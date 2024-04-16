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

    def setVirtualDevice(self,sCamera=None):
        ''' set instruments of the microscope '''
        self.device['sCamera'] = sCamera
        self.device['camera'] = sCamera.camera

    def calculateVirtualFrame(self):
        ''' update the virtual Frame of the camera '''
        # TODO: set the spectral channel size
        # TODO: set the dispersionIntoBlock
        
        # image sample onto dispersive element
        iFrame=self.sample.get()

        # horizontal dispersion (RGB)
        # it will disperse the channel into single wavelength images aligned horizontally 
        if hasattr(self.device['sCamera'].spectraCalibration,'rgbOrder') and (
            self.device['sCamera'].spectraCalibration.rgbOrder == 'RGB'):
            
            oFrame = np.zeros((iFrame.shape[0],self.device['camera'].getParameter('height'),
                        self.device['camera'].getParameter('width')//iFrame.shape[0]))
            Component.ideal4fImaging(iFrame=iFrame,oFrame=oFrame,iFramePosition = np.array([0,0]),
                            magnification=1,iPixelSize=self.sample.pixelSize,oPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'])
            oFrame = Component2.disperseHorizontal(oFrame)

        # RGGB dispersion onto super-pixels
        if hasattr(self.device['sCamera'].spectraCalibration,'rgbOrder') and (
            self.device['sCamera'].spectraCalibration.rgbOrder == 'RGGB'):
            oFrame = np.zeros((iFrame.shape[0],self.device['camera'].getParameter('height')//2,
                        self.device['camera'].getParameter('width')//2))
            Component.ideal4fImaging(iFrame=iFrame,oFrame=oFrame,iFramePosition = np.array([0,0]),
                            magnification=1,iPixelSize=self.sample.pixelSize,oPixelSize=self.device['camera'].DEFAULT['cameraPixelSize'])
            oFrame = Component2.disperseIntoRGGBBlock(oFrame)






        # image it onto camera-chip
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

    from viscope.instrument.virtual.virtualCamera import VirtualCamera
    from viscope.main import Viscope
    from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui

    #camera
    camera = VirtualCamera()
    camera.connect()
    camera.setParameter('threadingNow',True)

    #spectral camera
    sCal = CalibrateRGBImage(rgbOrder='RGB')
    sCamera = SCamera(name='sCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('calibrationData',sCal)
    sCamera.setParameter('threadingNow',True)


    # virtual microscope
    vM = SimpleSpectralMicroscope()
    vM.setVirtualDevice(camera)
    vM.connect()

    # main event loop
    viscope = Viscope()
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)
    viscope.run()

    camera.disconnect()
    sCamera.disconnect()
    vM.disconnect()


