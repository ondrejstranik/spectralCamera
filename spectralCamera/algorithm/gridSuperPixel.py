'''
class to warp the hyperspectral image
'''
# %% this comment line is code running in Jupiter notebook
import numpy as np
#from skimage import data, filters, measure, morphology
#from skimage.filters import threshold_otsu, rank
#from skimage.transform import warp
#from scipy.interpolate import griddata

from spectralCamera.algorithm.basisVectors import lattice_basis_vectors

class GridSuperPixel():
    ''' main class to classify the grid of super-pixels in the image'''

    DEFAULT = {'devMax': 0.2} # maximal divation of the vector grid from average. it is used index the grid

    def __init__(self):
        ''' initialise the class '''
        self.position = None # position of the spots - input array N x 2 (float)
        self.imIdx = None # 2d index -(int,int) - of  the spots  
        self.idx00 = None # position (int) of the index (0,0)
        self.xy00 = None # position of the spots with index (0,0)
        self.xVec = None # first vector of the grid
        self.yVec = None # second vector of the grid

        # bool value for the boxes inside the image
        self.inside = None

        # devMax ...maximal deviation of the pixel position from 
        # the latices vector in order to identify it 
        self.devMax = self.DEFAULT['devMax']

    def setGridPosition(self, position):
        ''' set position of each spectral pixel
        '''
        self.position = position
        self.inside = np.ones_like(self.position[:,0], dtype=bool)

    def getGridInfo(self):
        ''' get the smallest lattice vectors and origin of the lattice'''

        # get lens position lattice vectors
        #com = np.vstack((self.position[1,:],self.position[0,:])).T
        v1 = lattice_basis_vectors(self.position,1000)

        # identity the two basis vectors close to x and y axis
        xIdx= np.argmax(np.dot(v1,np.array([1,0])))
        yIdx= np.argmax(np.dot(v1,np.array([0,1])))
        self.xVec = v1[xIdx,:]
        self.yVec = v1[yIdx,:]

        # getting the staring point
        _x0 = np.mean(self.position[:,0])
        _y0 = np.mean(self.position[:,1])
        self.idx00 = np.argmin((self.position[:,0]-_x0)**2 + (self.position[:,1]-_y0)**2)
        self.xy00 = self.position[self.idx00,:]

    def shiftIdx00(self,new00Idx):
        ''' shift the position of origin of the lattice '''
        self.imIdx = self.imIdx - new00Idx
        # recalculate zero index
        #self.idx00 = np.arange(self.imIdx.shape[0])[(self.imIdx[:,0]==0) & (self.imIdx[:,1]==0)][0]
        self.idx00 = np.argmax((self.imIdx[:,0]==0) & (self.imIdx[:,1]==0))

        self.xy00 = self.position[self.idx00,:]

    def getPixelIndex(self):
        ''' 
        get index of each spectral pixel 
        '''
      
        # define the index matrix
        idxValue = np.zeros_like(self.position).astype('int')
        idxCheck = np.zeros(idxValue.shape[0]).astype('bool')

        idxValue[self.idx00,:] = np.array([0,0])
        idxCheck[self.idx00] = True

        idxToCheck = [self.idx00]
        ii = 0
        while idxToCheck != []:
            if ii%10 == 0:
                print(f'Pixel indexing step {ii}')
            
            idxToCheckNew = []

            # check all position from the list idxToCheck
            for mi in idxToCheck:
                # identify the four closest neighbor
                vecuXuY = np.array([[1.0,0.0],[-1.0,0.0],[0.0,1.0],[0.0,-1.0]])
                for myVec in vecuXuY:
                    vecXY = self.position[mi,:] + myVec[0]*self.xVec + myVec[1]*self.yVec
                    idxXY = np.argmin(np.linalg.norm(self.position - vecXY,axis=1))

                    if ((idxCheck[idxXY]==False) and 
                        (np.linalg.norm(self.position[mi,:] - self.position[idxXY,:]
                        + myVec[0]*self.xVec + myVec[1]*self.yVec) < 
                        self.devMax*np.linalg.norm( myVec[0]*self.xVec + myVec[1]*self.yVec))):
                        
                        idxValue[idxXY,:] = idxValue[mi] + myVec
                        idxCheck[idxXY] = True
                        idxToCheckNew.append(idxXY)

            idxToCheck = idxToCheckNew
            ii +=1

        # remove the points which are not identified
        foundPointIdx = (idxCheck==True)
        idxValue = idxValue[foundPointIdx]
        self.imIdx = idxValue.astype(int)
        self.position = self.position[foundPointIdx]
        self.inside = np.ones_like(self.position[:,0]).astype(bool)

        # recalculate zero index
        self.shiftIdx00(self,[0,0])        

    def getPositionInt(self):
        ''' get the grid position rounded on integer (full pixel) '''

        return np.round(self.position).astype(int)

    def getPositionOutsideImage(self,image,bheight=2, bwidth=30):
        ''' set flag self.inside for the blocks ,which are not whole on the image '''

        myPosition = self.getPositionInt()

        outsideY = (myPosition[:,0]< bheight ) | (myPosition[:,0]> image.shape[0] - bheight )
        outsideX = (myPosition[:,1]< bwidth ) | (myPosition[:,1]> image.shape[1] - bwidth )

        self.inside = ~(outsideX | outsideY)

        return self.inside

    def getSpectraBlock(self, image, bheight=2, bwidth=30):
        ''' cut spectral pixel block out of the image
        height of the block = 1 + 2*bheight 
        width of the bloc = 1 + 2*bwidth 
        '''

        mySpec = np.zeros((self.position.shape[0], 2*bheight+1,2*bwidth+1))

        myPosition = self.getPositionInt()

        # cut out blocks out of the raw image
        for ii in range(2*bwidth+1):
            for jj in range(2*bheight+1):
                mySpec[self.inside,jj,ii] = image[myPosition[self.inside,0]+jj-bheight,
                myPosition[self.inside,1]+ii-bwidth]

        return mySpec

    def getSpectralBlockImage(self,image,bheight=2, bwidth=30):
        ''' indicite the spectral blocks in the image '''

        blockImage = np.zeros_like(image,dtype=bool)
        myPosition = self.getPositionInt()
        for ii in range(2*bwidth+1):
            for jj in range(2*bheight+1):
                blockImage[myPosition[self.inside,0]+jj-bheight,
                myPosition[self.inside,1]+ii-bwidth] = True

        return blockImage

    def getAlignedImage(self,mySpec):
        ''' get the overview image of the spectral block
        each block on a cartesian grid '''

        btheigth = mySpec.shape[1]
        btwidth = mySpec.shape[2]
        
        # position of the blocks
        yIdx = (self.imIdx[:,0]-self.imIdx[:,0].min())*btheigth
        xIdx = (self.imIdx[:,1]-self.imIdx[:,1].min())*btwidth

        oIm = np.zeros((yIdx.max()+btheigth,xIdx.max()+btwidth))

        for ii in range(btwidth):
            for jj in range(btheigth):
                oIm[jj+yIdx,ii+xIdx] = mySpec[:,jj,ii]

        return oIm

    def getAlignedImageCut(self,mySpec):
        ''' align the  mySpec along the width '''

        btheigth = mySpec.shape[1]
        btwidth = mySpec.shape[2]

        yIdx = (self.imIdx[:,0]-self.imIdx[:,0].min())*btheigth
        xIdx = (self.imIdx[:,1]-self.imIdx[:,1].min())

        ooIm = np.zeros((btwidth,yIdx.max()+btheigth,xIdx.max()+1))

        for ii in range(btwidth):
            for jj in range(btheigth):
                ooIm[ii,jj+yIdx,xIdx] = mySpec[:,jj,ii]

        return ooIm

    def getWYXImage(self,mySpec):
        ''' use plain averaging along the y axis to get spectral image '''

        btheigth = mySpec.shape[1]
        btwidth = mySpec.shape[2]

        # shift the indexing to positive values
        idxValueShift = np.array([[self.imIdx[:,0].min(),self.imIdx[:,1].min()]])
        imIdx = self.imIdx - idxValueShift

        # define the size of the 3D spectral image
        wyxImageShape = (btwidth,imIdx[:,0].max()+1,imIdx[:,1].max()+1)        

        # averaging according the y-axis
        mySpecAve = np.mean(mySpec,axis=1)

        # generate the (lambda y x) image
        wyxImage = np.zeros(wyxImageShape)
        for ii in range(btwidth):
            wyxImage[ii,imIdx[:,0],imIdx[:,1]] = mySpecAve[:,ii]

        return wyxImage



if __name__ == "__main__":

    import skimage as ski
    import napari

    #example image
    myImage = ski.data.coins()


    #generate grid (not ideal)
    xx,yy = np.meshgrid(np.arange(20),np.arange(10))
    xx = xx + 0.1*np.random.rand(*xx.shape)
    yy = yy + 0.1*np.random.rand(*yy.shape)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    
    position = xx[:,None]*np.array([7.1,20]) + yy[:,None]*np.array([20.1,7.1])
    position = position[np.random.permutation(position.shape[0]),:]

    # characterize the grid
    gridSP = GridSuperPixel()
    gridSP.setGridPosition(position)
    gridSP.getGridInfo()
    gridSP.getPixelIndex()

    # transform the image
    gridSP.getPositionOutsideImage(myImage)
    spBlock = gridSP.getSpectraBlock(myImage)
    alignedIm = gridSP.getAlignedImage(spBlock)

    # prepare a sub selected  points
    pointsSelect = (gridSP.imIdx[:,0]%1 == 0 ) & (gridSP.imIdx[:,1]%1 == 0 ) & gridSP.inside
    points = gridSP.position[pointsSelect,:]

    features = {'pointIndex0': gridSP.imIdx[pointsSelect,0],
                'pointIndex1': gridSP.imIdx[pointsSelect,1]
                }
    text = {'string': '[{pointIndex0},{pointIndex1}]',
            'translation': np.array([-30, 0])
            }
    # prepare pointImage 
    pointImage = np.zeros_like(myImage).astype(bool)
    pInt = gridSP.getPositionInt()
    pointImage[pInt[gridSP.inside,0],pInt[gridSP.inside,1]] = True

    # display the images
    viewer = napari.Viewer()
    viewer.add_image(myImage)
    viewer.add_image(pointImage, opacity=0.5)
    viewer.add_points(points,features=features,text=text, size= 50, opacity=0.5)

    # display cutted image
    viewer2 = napari.Viewer()
    viewer2.add_image(alignedIm)
    napari.run()


# %%
