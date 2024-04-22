'''
class to calibrate RGB Images
'''

import numpy as np
from spectralCamera.algorithm.baseCalibrate import BaseCalibrate

class CalibrateIFImage(BaseCalibrate):
    ''' main class to calibrate virtual integral field images
    it take lines from the raw image '''

    DEFAULT = {'wavelength': np.arange(400,800,10), # spectral range
               'position00' : np.array([250,0]), # position of the line of superpixel 00
                'gridVector': np.array([7,10]), # 1st grid vector of the super pixel (2nd is perpendicular)
                'nYX': np.array([50,50])} # size of the super pixel array

    def __init__(self, wavelength=None, camera=None, **kwargs):
        ''' initialise the class '''
        if wavelength is None: wavelength = CalibrateIFImage.DEFAULT['wavelength']
        super().__init__(wavelength=wavelength,**kwargs)

        self.position00 = kwargs['position00'] if 'position00' in kwargs else self.DEFAULT['position00']
        self.gridVector = kwargs['gridVector'] if 'gridVector' in kwargs else self.DEFAULT['gridVector']
        self.nYX = kwargs['nYX'] if 'nYX' in kwargs else self.DEFAULT['nYX']
        self.height = camera.height
        self.width = camera.width

        self._calculatePosition()

    def _calculatePosition(self):
        ''' calculate position in the image '''
    
        nY = self.nYX[0]
        nX = self.nYX[1]
        gridVector2 = np.array([-self.gridVector[1],self.gridVector[0]])

        xIdx, yIdx = np.meshgrid(np.arange(nX), np.arange(nY))
        self.positionIdx = np.array([yIdx.ravel(),xIdx.ravel()]).T

        self.positionYX = self.positionIdx@np.vstack((gridVector2,self.gridVector))
        self.positionYX = self.positionYX + self.position00
    
        # restrict on only full dispersed super-pixel on the chip
        insideChip = ((self.positionYX[:,0]>=0) & (self.positionYX[:,0]<self.height) &
        (self.positionYX[:,1]>=0) & (self.positionYX[:,1] <self.width - np.size(self.wavelength)))
        self.positionIdx = self.positionIdx[insideChip]
        self.positionYX = self.positionYX[insideChip]

    def getSpectralImage(self,rawImage,**kwargs):
        ''' get the spectral image from raw image'''

        nW = np.size(self.wavelength)
        WYXImage = np.zeros((nW,*self.nYX))
        
        for ii in range(nW):
            WYXImage[ii,self.positionIdx[:,0],self.positionIdx[:,1]] = rawImage[self.positionYX[:,0],self.positionYX[:,1]+ii]

        return  WYXImage


if __name__ == "__main__":

    pass


































