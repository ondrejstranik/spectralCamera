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
from viscope.instrument.base.baseCamera import BaseCamera

class WebCamera(BaseCamera):
    ''' class to control usb/integrated camera. wrapper for cv2 '''
    DEFAULT = {'name': 'webCamera',
                #'exposureTime': 10, # ms initially automatically set the exposure time
                'nFrame': 1,
                'cameraIdx': 0,
                'filterType': 'RGGB'} # type of the filter 'RGB', 'RGGB','BW'

    def __init__(self, name=DEFAULT['name'],*args,**kwargs):
        ''' initialisation '''

        super().__init__(name=name,**kwargs)
        
        # camera parameters
        self.cameraIdx = kwargs['cameraIdx'] if 'cameraIdx' in kwargs else WebCamera.DEFAULT['cameraIdx']
        self.filterType = kwargs['filterType'] if 'filterType' in kwargs else WebCamera.DEFAULT['filterType']

        #self.exposureTime = WebCamera.DEFAULT['exposureTime']
        self.exposureTime = None
        self.nFrame = WebCamera.DEFAULT['nFrame']

        self.cap = None


    def connect(self):
        super().connect()
        self.cap = cv2.VideoCapture(self.cameraIdx)
        # switch off auto-exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) 

        # get the image size
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.filterType == 'RGB':
            self.width *= 3
        if self.filterType == 'RGGB':
            self.width *= 2
            self.height *= 2

        # get the camera optimal exposure time 
        self.exposureTime = self.getParameter('exposureTime')

    def disconnect(self):
        super().disconnect()
        self.cap.release()

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cameraIdx)

    def getLastImage(self):
        myframe = None
        for _ in range(self.nFrame):
            temporary_frame = None
            while temporary_frame is None:
                ret, temporary_frame = self.cap.read()
                time.sleep(0.03)

            if self.filterType == 'RGB':
                if myframe is None:
                    myshape = np.shape(temporary_frame.T)
                    myframe = np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2]))
                    myframe = myframe.astype('int64').T
                else:
                    myframe = myframe + np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2])).T

            if self.filterType == 'RGGB':
                _myframe = np.empty((temporary_frame.shape[0]*2,temporary_frame.shape[1]*2))
                _myframe[0::2,0::2] = temporary_frame[:,:,0] #R
                _myframe[0::2,1::2] = temporary_frame[:,:,1] //2 #R
                _myframe[1::2,0::2] = temporary_frame[:,:,1] //2 #R
                _myframe[1::2,1::2] = temporary_frame[:,:,2] //2 #B
                if myframe is None:
                    myframe = _myframe.astype('int64')
                else:
                    myframe = myframe + _myframe

            if self.filterType == 'BW':
                _myFrame = np.sum(temporary_frame,axis=0)
                if myframe is None:
                    myframe = _myframe.astype('int64')
                else:
                    myframe = myframe + _myframe                

        self.rawImage = myframe/self.nFrame
        return self.rawImage

    def _setExposureTime(self,value): # ms
        # the expression for the value of in cap.set is following:
        # 2**(cap.set-value-) = value [s]) 

        print(f'set Exposure Time {value}')
        print(f'set cap.set-value- {np.log2(value/1000)}')        
        self.cap.set(cv2.CAP_PROP_EXPOSURE, np.log2(value/1000))

        self.exposureTime = value

    def _getExposureTime(self):
        _exposureTime = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f'cap.set-value- {_exposureTime}')
        if _exposureTime >0:
            self.exposureTime = _exposureTime
        else:
            self.exposureTime = 2**(_exposureTime)*1000
        print(f'self.exposureTime {self.exposureTime}')

        return self.exposureTime

    def _setParameterOpenCV(self, parameter=None, value=None):
        ''' open cv parameters
        https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_VideoCaptureProperties.html
        '''
        if (parameter <40) and (value is not None):
            self.cap.set(parameter, value)

    def _getParameterOpenCV(self, parameter):
        ''' open cv parameters
        https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_VideoCaptureProperties.html
        '''
        if (parameter <40):
            return self.cap.get(parameter)
        else:
            return None


#%%

if __name__ == '__main__':
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera

    cam = WebCamera(name='WebCamera',filterType='RGGB')
    cam.connect()
    cam.setParameter('exposureTime',300)
    cam.setParameter('nFrames', 5)

    cam._displayStreamOfImages()
    cam.disconnect()


