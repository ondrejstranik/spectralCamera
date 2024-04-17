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
    pass  

