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
                'exposureTime': 1/32,
                'nFrames': 1,
                'cameraIdx': 0}

    def __init__(self, name=DEFAULT['name'],*args, **kwargs):
        ''' initialisation '''

        super().__init__(name=name,*args, **kwargs)
        
        # camera parameters
        self.cameraIdx = kwargs['cameraIdx'] if 'cameraIdx' in kwargs else WebCamera.DEFAULT['cameraIdx']
        self.exposureTime = BaseCamera.DEFAULT['exposureTime']
        self.nFrame = BaseCamera.DEFAULT['nFrame']

        self.cap = None


    def connect(self):
        super().connect()
        self.cap = cv2.VideoCapture(self.cameraIdx)
        # switch off auto-exposure
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) 
        self._setExposureTime(self.exposureTime)

        # get the image size
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

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

            if myframe is None:
                myshape = np.shape(temporary_frame.T)
                myframe = np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2]))
                myframe = myframe.astype('int64').T
            else:
                myframe = myframe + np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2])).T
        self.rawImage = myframe/self.nFrame
        return self.rawImage

    def _setExposureTime(self,value):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value/1000)
        self.exposureTime = value

    def _getExposureTime(self):
        _exposureTime = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        if _exposureTime >0:
            self.exposureTime = self.cap.get(cv2.CAP_PROP_EXPOSURE)
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
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from viscope.main import Viscope

    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    viscope = Viscope()
    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice([camera])
    viscope.run()

    camera.disconnect()
