# -*- coding: utf-8 -*-
"""
wrapper for Basler camera
"""
#%%

import numpy as np

from datetime import date
from os import path, mkdir
from timeit import default_timer as timer
import time 

import HSIplasmon as hsi

from pypylon import pylon


class baslerCamera:
    '''compatibility class for pypylon (baslerCamera)'''

    DEFAULT = {'maxNumBuffer':5,
                'blackLevel':100,
                'cameraMode': 'Mono12',
                'gain': 0,
                'exposureTime': 1000, # in milisecond
                'n_frames': 1}


    def __init__(self, *args, **kwags):
    
        # camera parameters
        self._camera = None

        self.height = None
        self.width = None

        # camera parameter for compatibility only
        self.spectraCalibration = None
        self.wavelength = None

        #### Measurement parameters
        self.n_frames = self.DEFAULT['n_frames']
        self.exposureTime = self.DEFAULT['exposureTime']

    def prepareCamera(self):
        ''' prepare camera and make initial setting '''

        # create instance of the camera
        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self._camera.Open()

        # set camera to mono12 mode
        self.setParameter('cameraMode',self.DEFAULT['cameraMode'])
        # set gain
        self.setParameter('gain',self.DEFAULT['gain'])
        # set blacklevel
        self.setParameter('blackLevel',self.DEFAULT['blackLevel'])
        # set exposure time
        self.setParameter('exposureTime',self.exposureTime)        

        # get the image size
        self.height = self._camera.HeightMax.GetValue()
        self.width = self._camera.WidthMax.GetValue()


    def GetCalibrationData(self,calibrationFile=None,spectraCalibration=None):
        ''' for compatibility only. this camera is not to be multispectral '''
        if spectraCalibration is not None:
            self.spectraCalibration = spectraCalibration
            self.wavelength = self.spectraCalibration.getWavelength()
        else:
            self.spectraCalibration = None
            self.wavelength = np.array([1])

    def startAcquisition(self):
        self._camera.StartGrabbing()

    def stopAcquisition(self):
        self._camera.StopGrabbing()


    def getLastImage(self,n_frames=None,*args):
        ''' get image from the buffer 
        return numpy array '''
        
        grabberTimeout = 5000 # [ms]

        if n_frames is not None:
            self.n_frames = n_frames

        for ii in range(self.n_frames):

            # it is waiting till image is recieved
            grabResult = self._camera.RetrieveResult(grabberTimeout, pylon.TimeoutHandling_ThrowException)
            print(f'image number {ii}')

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                if ii==0:
                    myIm = grabResult.Array
                    myIm = myIm.astype(float)
                else:
                    myIm = myIm + grabResult.Array

            else:
                myIm = np.zeros((self.height, self.width))
            grabResult.Release()

        myIm = myIm/self.n_frames
        return myIm

    def getWavelength(self):
        return self.wavelength
    
    def setParameter(self,name, value):
        ''' set parameter of the camera'''

        if name== 'ExposureTime':
            self._camera.ExposureTime.SetValue(value*1000)
            self.exposureTime = value
        if name=='cameraMode':
            self._camera.PixelFormat.SetValue(value)
        if name=='blackLevel':
            self._camera.BlackLevel.SetValue(value)
        if name== 'n_frames':
            self.n_frames = int(value)

    def getParameter(self,name):
        ''' get parameter of the camera '''

        if name=='ExposureTime':
            return self.exposureTime
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
        ''' convert image to hyper spectral cube
        only for compatibility. This camera is not to be hyperspectral'''

        return imageData[None,...]

    def closeCamera(self):
        ''' close the camera connection'''
        self._camera.Close()



if __name__ == "__main__":
    cam = baslerCamera()
    cam.prepareCamera()
    cam.setParameter('exposureTime',300)
    cam.setParameter('n_frames', 5)

    cam.startAcquisition()
    cam.displayStreamOfImages()
    cam.stopAcquisition()
    cam.closeCamera()




# %%
