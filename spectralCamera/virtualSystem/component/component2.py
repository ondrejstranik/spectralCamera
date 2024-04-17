"""
class to generate virtual sample

@author: ostranik
"""
#%%

import numpy as np
from skimage import data
from skimage.transform import resize, rescale

class Component2():
    ''' class to calculate light propagation through different optical components'''
    DEFAULT = {}

    @classmethod    
    def disperseIntoBlock(cls,iFrame:np.ndarray, blockShape:np.ndarray=None):
        ''' reshape the input iFrame (ndim=3) (wavelength, y, x) into frame with (ndim=2)
        the wavelength are dispersed into blocks of superPixels'''

        if blockShape is None: blockShape = np.array([1,iFrame.shape[0]]) 
        n = blockShape[1]
        m = iFrame.shape[0]//n # remove excess of spectra channels

        _iFrame = iFrame[0:n*m,:,:]

        oFrame = np.swapaxes(_iFrame,0,1)
        oFrame = np.swapaxes(oFrame,1,2)
        oFrame = np.reshape(oFrame,(iFrame.shape[1],iFrame.shape[2],m,n))
        oFrame = np.swapaxes(oFrame,1,2)
        oFrame = np.reshape(oFrame,(iFrame.shape[1]*m,iFrame.shape[2]*n))

        return oFrame        
    
    @classmethod
    def disperseHorizontal(cls, iFrame: np.ndarray):
        ''' reshape the input iFrame (ndim=3) (wavelength, y, x) into frame with (ndim=2)
        the wavelength are dispersed into blocks of superPixels'''
        oFrame = np.moveaxis(iFrame,0,1)
        oFrame = np.reshape(oFrame,(oFrame.shape[0],-1))
        return oFrame
    @classmethod
    def disperseIntoRGGBBlock(cls, iFrame:np.ndarray):
        ''' reshape into RGGB super pixel '''
        oFrame = np.empty((iFrame.shape[1]*2,iFrame.shape[2]*2))        
        oFrame[0::2,0::2] = iFrame[0,:,:] #R
        oFrame[0::2,1::2] = iFrame[1,:,:] //2 #R
        oFrame[1::2,0::2] = iFrame[1,:,:] //2 #R
        oFrame[1::2,1::2] = iFrame[2,:,:] #B
        return oFrame

    @classmethod
    def disperseIntoLines(cls,iFrame:np.ndarray, gridVector = np.array([2,5])):
        ''' disperse it into a lines, imitate Integral field camera
        set of micro-lenses with slanted grating dispersion element
        lines are dispersed horizontally '''
        #TODO: correct value assignment

        gridVector = np.array(gridVector)

        nW = iFrame.shape[0]
        nY = iFrame.shape[1]
        nX = iFrame.shape[2]

        xIdx, yIdx = np.meshgrid(np.arange(nX), np.arange(nY))
    
        positionIdx = np.array([yIdx.ravel(),xIdx.ravel()])

        gridVector2 = np.array([-gridVector[1],gridVector[0]])
        positionYX = positionIdx[0][:,None]*gridVector2 + positionIdx[1][:,None]*gridVector
        positionYX = positionYX - np.min(positionYX, axis=0)

        oFrame = np.zeros((np.max(positionYX[:,0])+1,np.max(positionYX[:,1])+nW+1))

        for ii in range(nW):
            oFrame[positionYX[:,0],positionYX[:,1]+ii] = iFrame[ii,positionIdx[0],positionIdx[1]]
        
        return oFrame

    @classmethod
    def _SpectraToSpectraIdx(cls,source,destination):
        ''' return index of the source spectra, 
        which will be mapped to the destination spectra.
        It finds the closed values. (no inter/extrapolation) 
        source, distanation ... 1D array'''

        spDistance = np.abs(source[:,None] - destination[None,:])
        spIndex = np.argmin(spDistance,axis=0)
        
        return spIndex

    @classmethod
    def spectraRangeAdjustment(cls,spImage,rangeIn,rangeOut):
        ''' adjust the spectral range of the image
        it is simple indexing of the original image '''

        spIndex = cls._SpectraToSpectraIdx(rangeIn,rangeOut)
        return spImage[spIndex,:,:]


#%%
if __name__ == '__main__':

    import napari
    spImage = np.random.rand(30,5,10) +1
    oFrame =Component2.disperseIntoLines(spImage, gridVector=[4,10])

    viewer = napari.view_image(oFrame)

    napari.run()









