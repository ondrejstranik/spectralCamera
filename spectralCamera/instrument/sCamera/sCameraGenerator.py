"""
Camera DeviceModel

Created on Mon Nov 15 12:08:51 2021

@author: ostranik
"""
#%%

from viscope.instrument.virtual.virtualCamera import VirtualCamera
from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera
from spectralCamera.instrument.sCamera.sCamera import SCamera
from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
from spectralCamera.algorithm.calibrateFilterImage import CalibrateFilterImage
from spectralCamera.algorithm.calibrateIFImage import CalibrateIFImage

#TODO: adjust the size of virtual cameras according the super pixel numbers
# TODO: add virtual integral field camera

class RGBWebCamera():
    ''' class to generate instruments for RGB web camera'''

    def __init__(self, cameraName= None, sCameraName= None,rgbOrder='RGGB',**kwargs):
        ''' initialisation '''

        #camera
        self.camera = WebCamera(name=cameraName, rgbOrder=rgbOrder)
        self.camera.connect()
        self.camera.setParameter('threadingNow',True)

        #spectral camera
        sCal = CalibrateRGBImage(rgbOrder=rgbOrder)
        self.sCamera = SCamera(name=sCameraName)
        self.sCamera.connect()
        self.sCamera.setParameter('camera',self.camera)
        self.sCamera.setParameter('calibrationData',sCal)
        self.sCamera.setParameter('threadingNow',True)

class VirtualRGBCamera():
    ''' class to generate instruments for virtual RGB camera'''

    def __init__(self, cameraName= None, sCameraName= None,rgbOrder='RGGB',**kwargs):
        ''' initialisation '''

        #camera
        self.camera = VirtualCamera(name=cameraName)
        self.camera.connect()
        self.camera.setParameter('threadingNow',True)

        #spectral camera
        sCal = CalibrateRGBImage(rgbOrder=rgbOrder)
        self.sCamera = SCamera(name=sCameraName)
        self.sCamera.connect()
        self.sCamera.setParameter('camera',self.camera)
        self.sCamera.setParameter('calibrationData',sCal)
        self.sCamera.setParameter('threadingNow',True)

class VirtualFilterCamera():
    ''' class to generate instruments for virtual RGB camera'''

    def __init__(self, cameraName= None, sCameraName= None, order=None,**kwargs):
        ''' initialisation '''

        #camera
        self.camera = VirtualCamera(name=cameraName)
        self.camera.connect()
        self.camera.setParameter('threadingNow',True)

        #spectral camera
        sCal = CalibrateFilterImage(order= order)
        self.sCamera = SCamera(name=sCameraName)
        self.sCamera.connect()
        self.sCamera.setParameter('camera',self.camera)
        self.sCamera.setParameter('calibrationData',sCal)
        self.sCamera.setParameter('threadingNow',True)


class VirtualIFCamera():
    ''' class to generate  instruments for virtual integral field camera '''

    def __init__(self, cameraName= None, sCameraName= None, order=None,**kwargs):
        ''' initialisation '''

        #camera
        self.camera = VirtualCamera(name=cameraName)
        self.camera.connect()
        self.camera.setParameter('threadingNow',True)

        #spectral camera
        sCal = CalibrateIFImage(order= order)
        self.sCamera = SCamera(name=sCameraName)
        self.sCamera.connect()
        self.sCamera.setParameter('camera',self.camera)
        self.sCamera.setParameter('calibrationData',sCal)
        self.sCamera.setParameter('threadingNow',True)





#%%

if __name__ == '__main__':
    from viscope.instrument.virtual.virtualCamera import VirtualCamera
    #from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualFilterCamera
    from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualIFCamera

    from viscope.main import Viscope
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui
    from viscope.gui.allDeviceGUI import AllDeviceGUI

    from spectralCamera.virtualSystem.multiSpectralMicroscope import MultiSpectralMicroscope
    
    #camera
    camera2 = VirtualCamera(name='BWCamera')
    camera2.connect()
    camera2.setParameter('threadingNow',True)

    #spectral camera system
    #scs = VirtualFilterCamera()
    scs = VirtualIFCamera()

    camera = scs.camera
    sCamera = scs.sCamera

    # virtual microscope
    vM = MultiSpectralMicroscope()
    vM.setVirtualDevice(sCamera=sCamera, camera2=camera2)
    vM.connect()

    # main event loop
    viscope = Viscope()
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)

    viewer  = AllDeviceGUI(viscope)
    viewer.setDevice([camera,camera2])

    viscope.run()

    sCamera.disconnect()
    camera.disconnect()
    camera2.disconnect()
    vM.disconnect()