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
#from os import path, mkdir
#from timeit import default_timer as timer


from viscope.instrument.base.baseCamera import BaseCamera


class MilCamera(BaseCamera):
    ''' class to control camera over the mil frame grabber'''
    DEFAULT = {'name': 'milCamera',
               'exposureTime': 10, # ms initially automatically set the exposure time
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
        self.ExposureTime_0 = 999850
        self.AcquisitionFrameRate_0 = 1
        self.AcquisitionFramePeriod_0 = 100

        #### Measurement parameters
        self.n_buffer_save = MilCamera.DEFAULT['n_buffer_save']
        # camera parameters
        self.exposureTime = MilCamera.DEFAULT['exposureTime']
        self.nFrame = MilCamera.DEFAULT['nFrame']



    def connect(self):
        super().connect()
        self.prepareCamera()

        self.setParameter('exposureTime',self.exposureTime)

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
        exposureTime_um = 1000* self.exposureTime
        #self.AcquisitionFrameRate = MIL.MIL_INT(int(self.n_buffer_save*np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /self.ExposureTime) ));
        #self.AcquisitionFramePeriod = MIL.MIL_INT(int((1/self.n_buffer_save)* np.ceil(self.AcquisitionFramePeriod_0 * self.ExposureTime / self.ExposureTime_0) ));
        
        self.AcquisitionFrameRate = MIL.MIL_INT(int(np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /exposureTime_um) ))
        self.AcquisitionFramePeriod = MIL.MIL_INT(int( np.ceil(self.AcquisitionFramePeriod_0 * exposureTime_um / self.ExposureTime_0) ))
        
        _ExposureTime = MIL.MIL_INT(exposureTime_um)
        
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
        MIL.MimArith(self.SaveBuffer[0], n_buffer_save, self.SaveBuffer[0], MIL.M_DIV_CONST);
        for i_buffer in range(1,n_buffer_save):
            #print("buffer ", i_buffer, " / ", self.n_buffer_save)
            MIL.MimArith(self.SaveBuffer[i_buffer], n_buffer_save, self.SaveBuffer[i_buffer], MIL.M_DIV_CONST);
            MIL.MimArith(self.SaveBuffer[i_buffer], self.SaveBuffer[0], self.SaveBuffer[0], MIL.M_ADD);

        return MIL.MbufGet(self.SaveBuffer[0])

    def getLastImage(self):
        nBuffer = self.nFrame//self.n_buffer_save
        print(f'number of full buffers: {nBuffer}')
        myframe = None
        for ii in range(nBuffer):
            print(f'full buffer number: {ii}')
            _myframe = self.getLastImageFromGrabber(self.n_buffer_save)
            if myframe is None:
                myframe = _myframe.astype(float)
            else:
                myframe = myframe + _myframe

        nLast = self.nFrame % self.n_buffer_save
        myframeLast = 0
        if nLast !=0:
            print(f'last not full buffer with number of images: {nLast}')
            myframeLast = self.getLastImageFromGrabber(nLast)
        if myframe is None:
            self.rawImage = myframeLast.astype(float)
        else:
            self.rawImage = myframe*self.n_buffer_save/self.nFrame + myframeLast*nLast/self.nFrame

        return self.rawImage



if __name__ == "__main__":
    cam = MilCamera()
    cam.connect()
    cam._displayStreamOfImages()

    cam.disconnect()




# %%
