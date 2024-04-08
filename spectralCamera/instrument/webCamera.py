"""
Camera DeviceModel

Created on Mon Nov 15 12:08:51 2021

@author: ostranik
"""
#%%

import os
import time
import numpy as np
import cv2


class webCamera():
    ''' class to control usb/integrated camera. wrapper for cv2 '''
    DEFAULTS = {'cameraIdx': 0,
                'exposureTime': 1/32,
                'n_frames': 10}

    def __init__(self, cameraIdx=None):
        ''' initialisation '''
        
        # camera parameters
        if cameraIdx is None:
            self.cameraIdx = self.DEFAULTS['cameraIdx']
        else:
            self.cameraIdx = cameraIdx
        self.exposureTime = self.DEFAULTS['exposureTime']
        self.n_frames = self.DEFAULTS['n_frames']
        self.cap = None
        self.frame = None

        self.height = None
        self.width = None


        self.spectraCalibration = None
        self.wavelength = None


    def prepareCamera(self):
        self.cap = cv2.VideoCapture(self.cameraIdx)
        # switch off auto-exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) 
        self.set_exposureTime()



        # get the image size
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cameraIdx)

    def getLastImage(self):
        myframe = None
        for _ in range(self.n_frames):
            temporary_frame = None
            while temporary_frame is None:
                ret, temporary_frame = self.cap.read()
                time.sleep(0.03)

            if myframe is None:
                myshape = np.shape(temporary_frame.T)
                myframe = np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2]))
                myframe = myframe.astype('int64').T
            else:
                myframe = myframe + np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2])).T
        self.frame = myframe/self.n_frames
        return self.frame

    def GetCalibrationData(self,spectraCalibration=None,*args):
        ''' set the default calibration class '''
        
        if spectraCalibration is None:
            from HSIplasmon.algorithm.calibrateRGBImages import CalibrateRGBImage
            self.spectraCalibration = CalibrateRGBImage()
        else:
            self.spectraCalibration = spectraCalibration
        
        self.wavelength = self.spectraCalibration.getWavelength()

    def getWavelength(self):
        return self.wavelength


    def setParameter(self,name,value):
        ''' set parameter of the camera '''

        if name== 'ExposureTime':
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value/1000)
            self.exposureTime = value
        if name== 'n_frames':
            self.n_frames = int(value)

    def getParameter(self,name):
        ''' get parameter of the camera '''

        if name=='ExposureTime':
            return self.exposureTime
        if name=='Wavelength':
            return self.wavelength
        if name=='n_frames':
            return self.n_frames
        if name=='width':
            return self.width
        if name=='height':
            return self.height

    def set_exposureTime(self, value=None):
        if value is not None:
            self.exposureTime = value
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposureTime)

    def get_exposureTime(self):
        _exposureTime = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        if _exposureTime >0:
            self.exposureTime = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        return self.exposureTime

    def set_parameter(self, parameter=None, value=None):
        ''' open cv parameters
        https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_VideoCaptureProperties.html
        '''
        if (parameter <40) and (value is not None):
            self.cap.set(parameter, value)

    def startAcquisition(self):
        ''' for compatibility only '''
        pass

    def stopAcquisition(self):
        ''' for compatibility only '''
        pass

    def get_parameter(self, parameter):
        ''' open cv parameters
        https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_VideoCaptureProperties.html
        '''

        if (parameter <40):
            return self.cap.get(parameter)
        else:
            return None

    def imageDataToSpectralCube(self,imageData):
        ''' convert image to hyper spectral cube'''

        return self.spectraCalibration.getSpectralImage(imageData)

    def displayStreamOfImages(self):
        ''' display the live images on the screen '''

        import napari
        from napari.qt.threading import thread_worker, create_worker
        import time        

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
        worker = create_worker(yieldHSImage)
        worker.yielded.connect(update_layer)

        worker.start()
        napari.run()

    def closeCamera(self):
        self.cap.release()

#%%

if __name__ == '__main__':
    cam = webCamera()
    cam.prepareCamera()
    cam.setParameter('exposureTime',300)
    cam.setParameter('n_frames', 5)

    cam.startAcquisition()
    cam.displayStreamOfImages()
    cam.stopAcquisition()
    cam.closeCamera()


# %%
