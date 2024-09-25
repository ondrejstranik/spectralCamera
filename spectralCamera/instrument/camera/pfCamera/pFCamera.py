"""
Camera photonFocus

wrapper for the photonFocus class

@author: ostranik
"""
#%%

import time
import numpy as np
from viscope.instrument.base.baseCamera import BaseCamera
from spectralCamera.instrument.camera.pfCamera.photonFocus import Photonfocus

class PFCamera(BaseCamera):
    ''' class to control photon Focus Camera. wrapper for PhotonFocus class '''
    DEFAULT = {'name': 'pfCamera',
                'exposureTime': 100,
                'nFrame': 1,
                'cameraIdx': 0}

    def __init__(self, name=None,*args,**kwargs):
        ''' initialisation '''

        if name is None: name=PFCamera.DEFAULT['name'] 
        super().__init__(name=name,**kwargs)
        
        # camera parameters
        self.cameraIdx = kwargs['cameraIdx'] if 'cameraIdx' in kwargs else PFCamera.DEFAULT['cameraIdx']
        self.exposureTime = PFCamera.DEFAULT['exposureTime']
        self.nFrame = PFCamera.DEFAULT['nFrame']

        self.cam = None

    def connect(self):
        super().connect()

        self.cam = Photonfocus()
        self.cam.PrepareCamera(cameraIdx=self.cameraIdx)
        self.cam.SetParameter("PixelFormat", "Mono12")

        self.setParameter('exposureTime',self.exposureTime)

        # get the image size
        self.height = self.cam.height
        self.width = self.cam.width

        self.startAcquisition()

    def disconnect(self):
        super().disconnect()

        self.stopAcquisition()
        self.cam.DisconnectCamera()

    def startAcquisition(self):
        self.cam.StartAcquisition()

    def stopAcquisition(self):
        self.cam.StopAcquisition()

    def getLastImage(self):
        myframe = None
        for ii in range(self.nFrame):
            if ii==0: _, myframe= self.cam.getLastImage()
            else:
                _, temporary_frame = self.cam.getLastImage()
                myframe = myframe + temporary_frame*1.0

        self.rawImage = myframe/self.nFrame
        return self.rawImage

    def _getExposureTime(self):
        exposureTime_us = self.cam.GetParameter("ExposureTime")
        self.exposureTime = exposureTime_us/1000
        return self.exposureTime

    def _setExposureTime(self,value): # ms
        self.exposureTime = value
        self.cam.SetParameter("ExposureTime",value*1000)



#%%

if __name__ == '__main__':
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera

    cam = WebCamera(name='WebCamera',filterType='RGGB')
    cam.connect()
    cam.setParameter('exposureTime',300)
    cam.setParameter('nFrames', 5)

    cam._displayStreamOfImages()
    cam.disconnect()


