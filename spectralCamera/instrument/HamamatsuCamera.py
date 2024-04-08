# -*- coding: utf-8 -*-
"""
wrapper for Hamamatsu camera
"""
#%%

import numpy as np
import ctypes
from datetime import date
from os import path, mkdir
from timeit import default_timer as timer
import time 

from HSIplasmon.camera.hamamatsu import HamamatsuCameraMR
from HSIplasmon.algorithm.calibrateRamanImages import CalibrateRamanImages
import HSIplasmon as hsi


class hamamatsuCamera:
    ''' wrapper class fro HCamData, so that it is compatible with other cameras '''

    def __init__(self, *args, **kwags):
    
        
        # camera parameters
        self._camera = HamamatsuCameraMR(camera_id=0)

        self.SaveBuffer = []
        self.spectraCalibration = None
        self.wavelength = None
        self.height = None
        self.width = None

        #### Measurement parameters
        self.n_frames = 1
        self.ExposureTime = 1000

    def prepareCamera(self):
        ''' prepare camera and make initial setting '''

        self.height = self._camera.getPropertyValue('image_height')[0]
        self.width = self._camera.getPropertyValue('image_width')[0]


    def GetCalibrationData(self,calibrationFile=None,spectraCalibration=None):
        ''' get the spectral calibration data '''
        
        
        if calibrationFile is not None:
            calibrationFile = calibrationFile

        if spectraCalibration is not None:
            self.spectraCalibration = spectraCalibration
            
        else:
        
            self.spectraCalibration = CalibrateRamanImages()
            self.spectraCalibration = self.spectraCalibration.loadClass(calibrationFile)
        
        self.wavelength = self.spectraCalibration.getWavelength()

         

    def set_exposure_time(self, ExposureTime = 1000):
        ''' ExposureTime in miliseconds '''

        self._camera.setPropertyValue('exposure_time', ExposureTime/1000)

    def startAcquisition(self):
        self._camera.startAcquisition()

    def stopAcquisition(self):
        self._camera.stopAcquisition()


    def getLastImage(self,n_frames=None,*args):
        ''' get image from the buffer 
        return numpy array '''
        
        if n_frames is not None:
            self.n_frames = n_frames

        for ii in range (self.n_frames):
            while self._camera.newFrames()== []:
                time.sleep(0.03) 
            if ii==0:
                myIm = self._camera.getLast()
                myIm = myIm.astype(float)
            else:
                myIm = myIm + self._camera.getLast()
            
        myIm = myIm/self.n_frames
        return myIm

    def getWavelength(self):
        return self.spectraCalibration.getWavelength()        
        #return self.wavelength
    
    def setParameter(self,name, value):
        ''' set parameter of the camera'''

        if name== 'ExposureTime':
            self.set_exposure_time(value)

        if name== 'n_frames':
            self.n_frames = int(value)

    def getParameter(self,name):
        ''' get parameter of the camera '''

        if name=='ExposureTime':
            return self.ExposureTime
        if name=='Wavelength':
            return self.getWavelength()
        if name=='n_frames':
            return self.n_frames
        if name=='width':
            return self.width
        if name=='height':
            return self.height


    def displayStreamOfImages(self):
        ''' display the live images on the screen '''

        import napari
        from napari.qt.threading import thread_worker
        import time        

        @thread_worker
        def yieldHSImage():
            while True:
                yield  self.getLastImage()
                time.sleep(0.03)

        def update_layer(new_image):
            rawlayer.data = new_image


        # start napari        
        viewer = napari.Viewer()

        im = np.zeros((self.height,self.width))
        # raw image
        rawlayer = viewer.add_image(im, rgb=False, colormap="gray", 
                                            name='Raw',  blending='additive')

        # prepare threads
        worker = yieldHSImage()
        worker.yielded.connect(update_layer)

        worker.start()
        napari.run()


    def imageDataToSpectralCube(self,imageData):
        ''' convert image to hyper spectral cube'''
        
        return self.spectraCalibration.getSpectralImage(imageData)


    def closeCamera(self):
        ''' close the camera connection'''

        self._camera.shutdown()



if __name__ == "__main__":
    import napari
    import time
    cam = hamamatsuCamera()
    cam.prepareCamera()
    cam.set_exposure_time(500)
    cam.setParameter('n_frames', 5)

    cam.startAcquisition()
    cam.displayStreamOfImages()
    cam.stopAcquisition()
    cam.closeCamera()




# %%
