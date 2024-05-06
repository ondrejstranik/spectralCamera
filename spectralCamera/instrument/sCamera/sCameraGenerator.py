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
        sCal = CalibrateIFImage(camera=self.camera)
        self.sCamera = SCamera(name=sCameraName)
        self.sCamera.connect()
        self.sCamera.setParameter('camera',self.camera)
        self.sCamera.setParameter('calibrationData',sCal)
        self.sCamera.setParameter('threadingNow',True)





#%%

if __name__ == '__main__':
    pass