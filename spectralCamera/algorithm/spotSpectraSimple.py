'''
class for calculating spot spectra from 3D spectral cube
'''
#%%


import numpy as np
from skimage.transform import rotate
import traceback


class SpotSpectraSimple:
    ''' class for calculating spot spectra '''
    
    DEFAULT = {'pxAve': 3, # radius of the spot
                }


    def __init__(self,wxyImage=None,spotPosition= [],**kwarg):
        ''' initialization of the parameters '''

        if wxyImage is not None: self.wxyImage = wxyImage  # spectral Image
        if spotPosition is not []: self.spotPosition = spotPosition

        # parameters of the mask
        self.pxAve= int(kwarg['pxAve']) if 'pxAve' in kwarg else  self.DEFAULT['pxAve']

        self.maskSize = None # total size of the mask
        self.maskSpot = None # weights for calculation of spots spectra
        self.maskSpotIdx = None # indexes of of the mask

        self.outliers = None # bool vector with spotPosition, which are outside the image

        self.spectraSpot = []

        self.setMask()
        

    def setMask(self,pxAve=None):
        ''' set the geometry of spots mask  and calculate spectra'''

        if pxAve is not None:
            self.pxAve = int(pxAve)

        # mask is a square
        self.maskSize = 2*self.pxAve +1
        self.maskSpot = np.ones((self.maskSize,self.maskSize))

        # return if there is no image
        if not hasattr(self,'wxyImage'):
            return

        # identify the spots, whose mask is not whole in image
        # convert self.spotPosition to numpy array. it is better to operate on
        _spotPosition = np.array(self.spotPosition, dtype=int)

        # return in there are no spots
        if _spotPosition.size ==0:
            return

        # define the bool vector with outliers
        self.outliers = np.zeros(_spotPosition.shape[0], dtype=bool)

        try:
            # get the indexes of the masks
            _maskSpotIdx = np.where(self.maskSpot)

            self.maskSpotIdx = (
            _maskSpotIdx[0]+ _spotPosition[:,0][:,None]-self.maskSize//2,
            _maskSpotIdx[1]+ _spotPosition[:,1][:,None]-self.maskSize//2
            )

            _olm = np.any((
                self.maskSpotIdx[0]<0, 
                self.maskSpotIdx[0]>self.wxyImage.shape[1]-1, 
                self.maskSpotIdx[1]<0,
                self.maskSpotIdx[1]>self.wxyImage.shape[2]-1,
                ),axis=0)

            self.outliers = np.any(_olm,axis=1)

        except:
            print('error in setting self.maskImage')
            traceback.print_exc()

        # calculate the spectra with the new mask
        self.calculateSpectra()

    def setSpot(self, spotPosition):
        ''' set position of the spots  and calculate spectra'''
        self.spotPosition = np.array(spotPosition)

        self.setMask()
        
    def setImage(self, wxyImage):
        ''' set the spectra image and calculate image and calculate spectra'''
        self.wxyImage = wxyImage

        self.calculateSpectra()

    def calculateSpectra(self):
        ''' calculate the spectra '''

        if self.spotPosition is None or len(self.spotPosition)==0:
            return
        else:
            nSpot = len(self.spotPosition)
        
        if self.maskSpotIdx is None:
            print('no self.maskSpotIdx or self.maskBcgIdx')
            return
        
        if not hasattr(self,'wxyImage') or self.wxyImage is None:
            return

        _spectraSpot = np.ones((nSpot,self.wxyImage.shape[0]))

        try:
            _spectraSpot[~self.outliers,:] = np.sum(
                self.wxyImage[:,
                            self.maskSpotIdx[0][~self.outliers,:],
                            self.maskSpotIdx[1][~self.outliers,:]
                            ],axis=2).T
        except:
            print('error in calculateSpectra')
            traceback.print_exc()

        self.spectraSpot = _spectraSpot.tolist()


    def getMask(self):
        ''' return the image of the mask of spots and background '''
        return self.maskImage
    
    def getSpectra(self):
        ''' return spectra of the spots'''
        return self.spectraSpot


#%%

if __name__ == "__main__":
    pass
















# %%
