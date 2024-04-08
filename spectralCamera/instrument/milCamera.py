# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:44:55 2023

@author: ungersebastian
"""
#%%

import mil as MIL
import numpy as np
import ctypes
from datetime import date
from os import path, mkdir
from timeit import default_timer as timer

from HSIplasmon.algorithm.calibrateFrom3Images import CalibrateFrom3Images
import HSIplasmon as hsi
import pickle

class milCamera:

    def __init__(self, *args, **kwags):
    
        
        # Mil parameters
        self.MilApplication = None
        self.MilSystem = None
        self.MilDisplay = None
        self.MilDigitizer = None 

        #self.GrabBuffer = []
        self.SaveBuffer = []
        self.spectraCalibration = None
        self.wavelength = None

        ## Standard, don't change
        self.ExposureTime_0 = 999850;
        self.AcquisitionFrameRate_0 = 1;
        self.AcquisitionFramePeriod_0 = 1000000;

        #### Measurement parameters
        self.n_frames = 2**0 # == 16
        self.n_buffer_save = 2**3 # == 8
        self.ExposureTime = 5000
        self.path_batch = None #folder to save images


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

    def startAcquisition(self):
        ''' start grabbing the images. 
        This camera does is automatically, so this function is for compatibility only'''
        pass

    def stopAcquisition(self):
        ''' stop grabbing the images. 
        This camera does is automatically, so this function is for compatibility only'''
        pass
       
    def GetCalibrationData(self,calibrationFile=None):
        ''' get the spectral calibration data '''
        
        if calibrationFile is not None:
            calibrationFile = calibrationFile

        self.spectraCalibration = CalibrateFrom3Images()
        self.spectraCalibration = self.spectraCalibration.loadClass(calibrationFile)
        
        self.wavelength = self.spectraCalibration.getWavelength()

    def set_exposure_time(self, ExposureTime = 999850):
        ''' ExposureTime in miliseconds '''
       
        ## just change exposure time
        self.ExposureTime = ExposureTime #5000;#999850;
        #self.AcquisitionFrameRate = MIL.MIL_INT(int(self.n_buffer_save*np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /self.ExposureTime) ));
        #self.AcquisitionFramePeriod = MIL.MIL_INT(int((1/self.n_buffer_save)* np.ceil(self.AcquisitionFramePeriod_0 * self.ExposureTime / self.ExposureTime_0) ));
        
        self.AcquisitionFrameRate = MIL.MIL_INT(int(np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /self.ExposureTime) ));
        self.AcquisitionFramePeriod = MIL.MIL_INT(int( np.ceil(self.AcquisitionFramePeriod_0 * self.ExposureTime / self.ExposureTime_0) ));
        
        self._ExposureTime = MIL.MIL_INT(self.ExposureTime);
        
        # Put the digitizer in asynchronous mode to be able to process while grabbing.
        MIL.MdigControl(self.MilDigitizer, MIL.M_GRAB_MODE, MIL.M_ASYNCHRONOUS)
        
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("AcquisitionFrameRate"), MIL.M_TYPE_MIL_INT, ctypes.byref(self.AcquisitionFrameRate))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("AcquisitionFramePeriod"), MIL.M_TYPE_MIL_INT, ctypes.byref(self.AcquisitionFramePeriod))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("ExposureTime"), MIL.M_TYPE_MIL_INT, ctypes.byref(self._ExposureTime))
        
    def free_alloc(self):
        ''' free the buffer on the mil '''
        for n in range(0, self.n_buffer_save):
            MIL.MbufFree(self.SaveBuffer[n])
    
            MIL.MappFreeDefault(self.MilApplication, self.MilSystem, self.MilDisplay, self.MilDigitizer, MIL.M_NULL)
        
    def set_export(self, root, batch):
        ''' set parameters for the saving the images into files '''
        
        path_root = root
        today = str(date.today())
        
        path_date = path.join(path_root, today)
        try: 
            mkdir(path_date)
        except:
            pass
                
        self.path_batch = path.join(path_date, batch)
        if not path. exists(self.path_batch):
            mkdir(self.path_batch)
        else:
            pass        

       
    def grabAndSave(self, name, n_ges = 1):
        ''' grabbing the images and saving them to disk '''
        
        for i_mess in range(n_ges):
            
            print('Start grabbing ', i_mess, ' / ', n_ges)
            
            start = timer()
            
            print('frame', " ", 'buffer save', " ", 'buffer_read')
            for i_frame in range(self.n_frames*self.n_buffer_save):
                i_buff_save = i_frame % self.n_buffer_save;
                print(i_frame, " ", i_buff_save)
            
                MIL.MdigGrab(self.MilDigitizer, self.SaveBuffer[i_buff_save])
                MIL.MdigGrabWait(self.MilDigitizer, MIL.M_GRAB_END);
            
            print("Measurement finished. Accumulating buffers.")
            print("buffer 0 / ", self.n_buffer_save)
            MIL.MimArith(self.SaveBuffer[0], self.n_buffer_save, self.SaveBuffer[0], MIL.M_DIV_CONST);
            for i_buffer in range(1,self.n_buffer_save):
                print("buffer ", i_buffer, " / ", self.n_buffer_save)
                MIL.MimArith(self.SaveBuffer[i_buffer], self.n_buffer_save, self.SaveBuffer[i_buffer], MIL.M_DIV_CONST);
                MIL.MimArith(self.SaveBuffer[i_buffer], self.SaveBuffer[0], self.SaveBuffer[0], MIL.M_ADD);
            
            end = timer()
            
            time = end - start
            
            print('total time: ', time, 's')
            
            print('... Saving ...')
            file = path.join(self.path_batch, ''.join([name, "_",str(i_mess+1),'.tif']))
            
            MIL.MbufSave(file,self.SaveBuffer[0])
            
            print('... finished ...')
        
    def getLastImage(self,*args):
        ''' get image from the buffer 
        return numpy array '''

        # save images to the grabber card ... maximum 8
        for i_frame in range(self.n_buffer_save):
            #print(i_frame, " ", self.n_buffer_save)
            MIL.MdigGrab(self.MilDigitizer, self.SaveBuffer[i_frame])
            MIL.MdigGrabWait(self.MilDigitizer, MIL.M_GRAB_END);

        # make average image of the buffered images
        #print("Measurement finished. Accumilationg buffers.")
        MIL.MimArith(self.SaveBuffer[0], self.n_buffer_save, self.SaveBuffer[0], MIL.M_DIV_CONST);
        for i_buffer in range(1,self.n_buffer_save):
            #print("buffer ", i_buffer, " / ", self.n_buffer_save)
            MIL.MimArith(self.SaveBuffer[i_buffer], self.n_buffer_save, self.SaveBuffer[i_buffer], MIL.M_DIV_CONST);
            MIL.MimArith(self.SaveBuffer[i_buffer], self.SaveBuffer[0], self.SaveBuffer[0], MIL.M_ADD);

        return MIL.MbufGet(self.SaveBuffer[0])

    def getWavelength(self):
        return self.wavelength
    
    def setParameter(self,name, value):
        ''' set parameter of the camera'''

        if name== 'ExposureTime':
            self.set_exposure_time(value)

    def getParameter(self,name):
        ''' get parameter of the camera '''

        if name=='ExposureTime':
            return self.ExposureTime
        if name=='Wavelength':
            return self.getWavelength()

   
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


    def imageDataToSpectralCube(self,imageData,aberrationCorrection=False):
        ''' convert image to hyper spectral cube '''

        #return np.swapaxes(self.spectraCalibration.getSpectralImage(imageData),-1,0)

        return self.spectraCalibration.getSpectralImage(imageData,aberrationCorrection=aberrationCorrection)

    def closeCamera(self):
        ''' close the camera connection '''
        self.free_alloc()




if __name__ == "__main__":
    cam = milCamera()
    cam.prepareCamera()
    cam.set_exposure_time(5000)
    
    cam.GetCalibrationData()

    cam.displayStreamOfImages()

    cam.free_alloc()

