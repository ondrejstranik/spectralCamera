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
        the wavelength are dispersed into blocks'''

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




#%%
