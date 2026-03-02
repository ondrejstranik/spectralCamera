"""
class emulating sCamera for reading spectral images from File

Created on Mon Nov 15 12:08:51 2021

@author: ostranik
"""
#%%

import os
import time
import numpy as np
from viscope.instrument.base.baseSequencer import BaseSequencer
from spectralCamera.algorithm.fileSIVideo import FileSIVideo
import traceback

class SCameraFromFile(BaseSequencer):
    ''' class to emulating sCamera for delivering saved spectral images
    '''
    DEFAULT = {'name': 'SCameraFromFile'
                }

    def __init__(self, name=None, **kwargs):
        ''' initialisation '''

        if name== None: name= SCameraFromFile.DEFAULT['name']
        super().__init__(name=name, **kwargs)
        
        # spectralCamera parameters
        self.sImage = None
        self.wavelength = None
        self.t0 = 0 # time of acquisition of the last spectral image
        self.image = np.zeros((2,2)) #  it is defined only for compatibility with SCamera processor
        
        # image file parameters
        self.fileSIVideo = FileSIVideo()
        self.fileName = None # list of files name
        self.fileTime = None # list of files time
        self.nFile = 0 # number of file
        self.idx = [] # indexes of files to process
        self.currentIdx = None # current Idx, which is loaded

        # processor, which has to be free in order to send next image
        self.processor = None
        self.isReading = False # indicates if it is reading the images
        self.flagToProcess = None

    def getWavelength(self):
        return self.wavelength        

    def getLastSpectralImage(self):
        ''' direct call of the camera image and spectral processing of it '''
        return self.sImage

    def setFolder(self,folder):
        ''' set folder with the images and get info about the files'''
        self.fileSIVideo.setFolder(folder)
        self.wavelength = self.fileSIVideo.loadWavelength()

        self.fileName, self.fileTime = self.fileSIVideo.getImageInfo()
        self.nFile = len(self.fileName)
        #self.fileTime = self.fileTime/1e9 # convert to seconds
        print(f'fileName {self.fileName}')

    def getFolder(self):
        ''' get current folder'''
        return self.fileSIVideo.folder

    def startReadingImages(self,idx=None):
        ''' initiate sending the spectral images'''
        if idx is None:
            self.idx = list(range(self.nFile))
        else:
            self.idx = idx
        if idx is None:
            return

        print('starting reading images')
        print(f' idx {self.idx}')
        self.isReading = True

    def stopReadingImages(self):
        ''' stop the thread of reading new images '''
        self.isReading = False

    def isReading(self):
        ''' check if it is reading images'''
        return self.isReading


    def connect(self,processor=None):
        ''' connect data processor with the camera '''
        super().connect()
        if processor is not None:
            self.setParameter('processor',processor)

    def setParameter(self,name, value):
        ''' set parameter of the spectral camera'''
        super().setParameter(name,value)

        if name== 'processor':
            self.processor = value
            try:
                self.flagToProcess = self.processor.flagLoop
            except:
                print(f'this processor does not have flagToProcess')

    def getParameter(self,name):
        ''' get parameter of the camera '''
        _value = super().getParameter(name)
        if _value is not None: return _value        

        if name=='wavelength':
            return self.getWavelength()

        if name== 'processor':
            return self.processor

    def loop(self):
        ''' process new data file'''
        print('running processData loop in sCamera from File')
        
        while True:
            #wait till new sequence will come                        
            while not self.isReading:
                yield False
                time.sleep(0.1)
            try:
                for ii in self.idx:
                    print(f'processing file # {ii}')
                    self.sImage = self.fileSIVideo.loadImage(self.fileName[ii])
                    self.t0 = self.fileTime[ii]/1e9
                    self.currentIdx = ii
                    yield True
                    self.flagLoop.set()

                    # wait till the images are processed
                    #while ((self.flagToProcess is not None) and 
                    #(not self.flagToProcess.is_set())):
                    #    time.sleep(0.003)
                    while not self.flagToProcess.is_set():
                        time.sleep(0.003)



                    # stop reading the images
                    if not self.isReading:
                        break
            except:
                print(f"An exception occurred in thread of {self.name}:\n")
                print(f'self.fileName {self.fileName}')
                traceback.print_exc()
 
            self.isReading = False
            yield False




#%%

if __name__ == '__main__':
    pass

