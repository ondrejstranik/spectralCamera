'''
class to calibrate RGB Images
'''
from pathlib import Path
import numpy as np
import pickle
import spectralCamera

class BaseCalibrate():
    ''' base class to calibrate BW image into spectral images'''

    DEFAULT = {'wavelength': np.array([550])}

    def __init__(self,wavelength=None, **kwarg):
        ''' initialise the class 
        wavelength ... wavelength'''
        
        if wavelength is None:
            self.wavelength = BaseCalibrate.DEFAULT['wavelength']
        else:
            self.wavelength = wavelength

    def getSpectralImage(self,rawImage,**kwargs):
        ''' extend only into spectral dimension'''
        WYXImage = rawImage[None,...]
        return  WYXImage

    def getWavelength(self):
        ''' get the RGB wavelengths '''
        return self.wavelength

    def saveClass(self, classFileName= None, classFolder=None):
        ''' save the class to a file '''
        if classFileName is None: classFileName = self.__class__.__name__ + '.obj'

        if classFolder is None: 
            classFolder = spectralCamera.dataFolder
            
        fullFile = classFolder + '/' + classFileName

        file = open(fullFile, 'wb') 
        pickle.dump(self, file)
        file.close()    

    def loadClass(self,classFile=None):
        ''' load the class itself from file '''

        # it is necessary in order to unpickle not only from global variable space
        myVars = globals()
        myVars.__setitem__(self.__class__.__name__,self.__class__)

        if classFile is None:
            fullFile = spectralCamera.dataFolder + '/' + self.__class__.__name__ + '.obj'
        else:
            fullFile = str(classFile)

        self = pickle.load(open(fullFile, 'rb'))

        return self


if __name__ == "__main__":
    pass

#%%






























