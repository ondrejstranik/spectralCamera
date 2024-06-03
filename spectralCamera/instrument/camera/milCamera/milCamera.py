# -*- coding: utf-8 -*-
"""
Camera with the mil grabber card

Created on Fri Aug 25 08:44:55 2023

@author: ungersebastian / ondrej Stranik
"""
#%%

import mil as MIL
import numpy as np
import ctypes
from datetime import date
from os import path, mkdir
from timeit import default_timer as timer


from viscope.instrument.base.baseCamera import BaseCamera

from HSIplasmon.algorithm.calibrateFrom3Images import CalibrateFrom3Images
import HSIplasmon as hsi
import pickle

class MilCamera(BaseCamera):
    ''' class to control camera over the mil frame grabber'''
    DEFAULT = {'name': 'webCamera',
               'exposureTime': 500, # ms initially automatically set the exposure time
               'nFrame': 1,
               'n_buffer_save': 2**3, # number of buffered images on the grabber card
    }


    def __init__(self, name=None, **kwargs):

        if name is None: name=MilCamera.DEFAULT['name'] 
        super().__init__(name=name,**kwargs)

        # Mil parameters
        self.MilApplication = None
        self.MilSystem = None
        self.MilDisplay = None
        self.MilDigitizer = None 

        #self.GrabBuffer = []
        self.SaveBuffer = []

        ## Standard, don't change
        self.ExposureTime_0 = 999850;
        self.AcquisitionFrameRate_0 = 1;
        self.AcquisitionFramePeriod_0 = 1000000;

        #### Measurement parameters
        self.n_buffer_save = MilCamera.DEFAULT['n_buffer_save']
        # camera parameters
        self.exposureTime = MilCamera.DEFAULT['exposureTime']
        self.nFrame = MilCamera.DEFAULT['nFrame']



    def connect(self):
        super().connect()
        self.prepareCamera()

    def prepareCamera(self):
        ''' prepare camera and make initial setting '''

        # Allocate defaults
        self.MilApplication, self.MilSystem, self.MilDisplay, self.MilDigitizer = MIL.MappAllocDefault(MIL.M_DEFAULT, ImageBufIdPtr=MIL.M_NULL)

        # image parameters
        self.height = MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_Y)
        self.width = MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_X)

        # Allocate the save buffers
        for n in range(0, self.n_buffer_save):
            self.SaveBuffer.append(
                MIL.MbufAlloc2d(self.MilSystem,
                MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_X),
                MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_Y),
                16 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC))
            MIL.MbufClear(self.SaveBuffer[n], MIL.M_COLOR_BLACK);

    def disconnect(self):
        self.free_alloc()
        super().disconnect()


    def _setExposureTime(self, value):
        ''' ExposureTime in miliseconds '''
       
        ## just change exposure time
        self.exposureTime = value #5000;#999850;
        #self.AcquisitionFrameRate = MIL.MIL_INT(int(self.n_buffer_save*np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /self.ExposureTime) ));
        #self.AcquisitionFramePeriod = MIL.MIL_INT(int((1/self.n_buffer_save)* np.ceil(self.AcquisitionFramePeriod_0 * self.ExposureTime / self.ExposureTime_0) ));
        
        self.AcquisitionFrameRate = MIL.MIL_INT(int(np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /self.exposureTime) ));
        self.AcquisitionFramePeriod = MIL.MIL_INT(int( np.ceil(self.AcquisitionFramePeriod_0 * self.exposureTime / self.ExposureTime_0) ));
        
        _ExposureTime = MIL.MIL_INT(self.exposureTime);
        
        # Put the digitizer in asynchronous mode to be able to process while grabbing.
        MIL.MdigControl(self.MilDigitizer, MIL.M_GRAB_MODE, MIL.M_ASYNCHRONOUS)
        
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("AcquisitionFrameRate"), MIL.M_TYPE_MIL_INT, ctypes.byref(self.AcquisitionFrameRate))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("AcquisitionFramePeriod"), MIL.M_TYPE_MIL_INT, ctypes.byref(self.AcquisitionFramePeriod))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("ExposureTime"), MIL.M_TYPE_MIL_INT, ctypes.byref(_ExposureTime))
        
    def free_alloc(self):
        ''' free the buffer on the mil '''
        for n in range(0, self.n_buffer_save):
            MIL.MbufFree(self.SaveBuffer[n])
    
            MIL.MappFreeDefault(self.MilApplication, self.MilSystem, self.MilDisplay, self.MilDigitizer, MIL.M_NULL)
        
   
    def getLastImageFromGrabber(self, n_buffer_save):
        ''' get - n_buffer_save - images from the buffer 
        return numpy array (average of them)'''

        # save images to the grabber card ... maximum 8
        for i_frame in range(n_buffer_save):
            #print(i_frame, " ", self.n_buffer_save)
            MIL.MdigGrab(self.MilDigitizer, self.SaveBuffer[i_frame])
            MIL.MdigGrabWait(self.MilDigitizer, MIL.M_GRAB_END);

        # make average image of the buffered images
        #print("Measurement finished. Accumulation buffers.")
        MIL.MimArith(self.SaveBuffer[0], self.n_buffer_save, self.SaveBuffer[0], MIL.M_DIV_CONST);
        for i_buffer in range(1,n_buffer_save):
            #print("buffer ", i_buffer, " / ", self.n_buffer_save)
            MIL.MimArith(self.SaveBuffer[i_buffer], self.n_buffer_save, self.SaveBuffer[i_buffer], MIL.M_DIV_CONST);
            MIL.MimArith(self.SaveBuffer[i_buffer], self.SaveBuffer[0], self.SaveBuffer[0], MIL.M_ADD);

        return MIL.MbufGet(self.SaveBuffer[0])

    def getLastImage(self):
        # TODO: finish this class method
        nBuffer = self.nFrame//self.n_buffer_save

        myframe = None
        for _ in range(nBuffer):
            temporary_frame = None
            _myframe = self.getLastImageFromGrabber(self.n_buffer_save)
            if myframe is None:
                myframe = _myframe
            else:
                myframe += _myframe

        nLast = self.nFrame % self.n_buffer_save
        if nLast !=0:
            _myframe = self.getLastImageFromGrabber(self.n_buffer_save)
        if myframe is None:
            myframe = _myframe



        self.rawImage = myframe/self.nFrame
        return self.rawImage






   
    def setParameter(self,name, value):
        ''' set parameter of the camera'''

        if name== 'ExposureTime':
            self.set_exposure_time(value)

    def getParameter(self,name):
        ''' get parameter of the camera '''

        if name=='ExposureTime':
            return self.ExposureTime





if __name__ == "__main__":
    cam = milCamera()
    cam.prepareCamera()
    cam.set_exposure_time(5000)
    
    cam.GetCalibrationData()

    cam.displayStreamOfImages()

    cam.free_alloc()

