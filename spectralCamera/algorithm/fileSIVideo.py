"""
class FileSIVideo

@author: ostranik
"""
#%%

import time
import numpy as np
from pathlib import Path
import re

class FileSIVideo:
    ''' class to save load time series of spectral Images into a folder '''
    
    DEFAULT = {'nameSet'  : {
                            'wavelength': 'wavelength.npy',
                            'image': 'time_{}'}
                            }

    def __init__(self,folder=None, **kwargs):
        ''' initialisation '''

        # data container
        self.folder = '' if folder is None else folder

    def setFolder(self,folder):
        self.folder = str(folder)

    def saveWavelength(self,wavelength):
        ''' save wavelength'''
        np.save(str(self.folder) +'/' +  self.DEFAULT['nameSet']['wavelength'],wavelength)

    def loadWavelength(self,folder=None):
        ''' loading wavelength'''

        if folder is not None: self.folder = folder

        _file = Path(self.folder + '/' + self.DEFAULT['nameSet']['wavelength'])        
        if not _file.is_file():
            return
        wavelength = np.load(_file)
        return wavelength

    def saveImage(self,sImage,timeTag=None):
        ''' save image in the folder with time tag
        the default timeTag is nanoseconds from the beginning of epoch time'''
        timeTag = time.time_ns() if timeTag is None else timeTag
        np.save(self.folder + '/' + self.DEFAULT['nameSet']['image'].format(timeTag),sImage)

    def loadImage(self,fileName, folder=None):
        ''' loading the spectral image'''
        if folder is not None: self.folder= folder
        image = np.load(self.folder + '/'+ fileName)
        return image

    def getImageInfo(self,folder=None):
        ''' getting list of spectral images and the corresponding timeTags in the folder
        use the default time tag (nanoseconds from the beginning of epoch time)
        return ( fileName:list, fileTime:np.array)'''
        if folder is not None: self.folder= folder

        vfolder = Path(self.folder)
        fileList = list(vfolder.glob(self.DEFAULT['nameSet']['image'].format('*')))
        _fileName = [x.parts[-1] for x in fileList]
        _fileTime = [int(re.search('\d+',x).group(0)) for x in _fileName]
        # sorted order of the file according their time
        sortedIdx = np.argsort(_fileTime)
        fileName = [_fileName[ii] for ii in sortedIdx]
        fileTime = np.array(_fileTime)[sortedIdx]

        return (fileName, fileTime)
        
    def loadAllImage(self,folder=None):
        ''' load all image sequence
         return (allImage:np.array .... image indexing = first index,
                wavelength: np.array 
                time: np:array .. image time)
        '''
        if folder is not None: self.folder= folder

        wavelength = self.loadWavelength()
        (fileName, fileTime) = self.getImageInfo()

        for ii,fName in enumerate(fileName):
            _image = self.loadImage(fName)
            if ii == 0:
                allImage = np.zeros((len(fileName),*_image.shape))
            allImage[ii,...] = _image
        
        return (allImage,wavelength,fileTime)


#%%
if __name__ == '__main__':
    pass
