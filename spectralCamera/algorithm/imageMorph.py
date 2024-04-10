'''
class to warp the hyperspectral image
'''
# %% this comment line is code running in Jupiter notebook
import numpy as np
from skimage import data, filters, measure, morphology
from skimage.filters import threshold_otsu, rank
from skimage.transform import warp
from scipy.interpolate import griddata

from HSIplasmon.algorithm.basisVectors import lattice_basis_vectors
import HSIplasmon as hsi


from timeit import default_timer as timer

import napari


class ImageMorph():
    ''' main class to morph the image '''

    def __init__(self, image):
        ''' initialise the class '''
        self.image = image
        self.position = None
        #self.positionInt = None
        self.imIdx = None
        self.idx00 = None
        self.xy00 = None
        self.xVec = None
        self.yVec = None
        self.idealPosition = None

        # devMax ...maximal deviation of the pixel position from 
        # the latices vector in order to identify it 
        self.devMax = 0.1 

        self.warpMatrix = None

    def getPixelPosition(self):
        ''' get position of each spectral pixel
        it identify the peaks in the image and get their position (subpixel)
        '''

        # renaming
        myim = self.image 
        
        # remove bad pixel and cosmic ray
        # TODO: improve the value of the threshing 
        myim[(myim> 10000)] = 0

        # threshold
        threshold = filters.threshold_otsu(myim)
        mask = myim > threshold

        # mask
        mask = morphology.remove_small_objects(mask, 10)
        mask = morphology.remove_small_holes(mask, 50)
        labels = measure.label(mask)

        # calculate properties of objects
        props = measure.regionprops_table(labels, myim,
                properties=['label', 'centroid'])

        self.position = np.vstack((props['centroid-0'],props['centroid-1'])).T
        #self.positionInt = np.round(self.position).astype('int')

    def getLatticeInfo(self):
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

    def shiftIdx00(self,new00idx):
        ''' shift the position of origin of the lattice '''
        self.idx00 = new00idx
        self.imIdx = self.imIdx - new00idx


    def getPixelIndexSimple(self):
        '''
        do not use. only for back compatibility.
        simple quick method working only for not distorted images 
        decompose spot position into the basis vector
        '''

        mat = np.hstack((self.xVec[:,None],self.yVec[:,None]))
        b = (self.position - self.xy00).T
        _imIdx = np.dot(np.linalg.inv(mat),b)

        # move to zero position
        #self.imIdx = np.round((_imIdx - np.array([[_imIdx[0,:].min()],[_imIdx[1,:].min()]]))).astype('int').T
        
        # round it to whole integer
        self.imIdx = np.round(_imIdx).T.astype('int')

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
                #print(idxToCheck)
                print(f'Pixel indexing step {ii}')
            
            idxToCheckNew = []

            # check all position from the list idxToCheck
            for mi in idxToCheck:
                # identify the flour closest neighbor
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
        #self.positionInt = self.positionInt[foundPointIdx]

        # recalculate zero index
        self.idx00 = np.arange(self.imIdx.shape[0])[(self.imIdx[:,0]==0) & (self.imIdx[:,1]==0)]

    def getIdealLattice(self, onPixel=False):
        ''' generate the ideal lattice 
        if onPixel ... idealPosition is not regular grid, but values are integer
                    ... it also change the position of xy00 (but not self.xy00)
        '''
        # pixel position on regular grid
        xy00 = self.xy00
        xVec = self.xVec
        yVec = self.yVec
        self.idealPosition = np.dot(self.imIdx,np.vstack((xVec,yVec))) + xy00

        if onPixel:
            # pixel position on gegular grid with integer values
            self.idealPosition = np.round(self.idealPosition).astype(int)


    def getWarpedImage(self,image):
        ''' warp image to ideal position '''

        warpedImage = warp(image, self.warpMatrix, mode='edge')        
        return warpedImage

    def _getOnPixelImage(self,image):
        ''' affine transform - scale and shift the image so that grid is on full pixels
        TODO: not applicable
        '''
        
        myScale = self.xVec/np.round(self.xVec).astype(int)
        myTranslate = self.xy00*myScale - np.round(self.xy00).astype(int)

        tform = ski.transform.AffineTransform(scale=myScale, translation= myTranslate)

        warped = ski.transform.warp(image, inverse_map=tform.inverse)

        
        return warped     

    def setWarpMatrix(self,pointGridStep=1):
        ''' calculate the distortion matrix '''
        # get the grid
        xx, yy = np.meshgrid(np.arange(self.image.shape[1]),np.arange(self.image.shape[0]))

        # prepare the points
        pointsSelect = (self.imIdx[:,0]%pointGridStep == 0 ) & (self.imIdx[:,1]%pointGridStep == 0 )

        points = self.position[pointsSelect,:]
        pointsIdeal = self.idealPosition[pointsSelect,:]

        vector = pointsIdeal -  points

        # interpolate the shift on all pixel in the image
        vy = griddata(points, vector[:,0], (yy, xx), method='linear', fill_value= 0)
        vx = griddata(points, vector[:,1], (yy, xx), method='linear', fill_value= 0)

        self.warpMatrix = np.array([yy - vy, xx - vx])


    def getSpectraBlock(self, image, bheigth=2, bwidth=30, idealPosition=True):
        ''' cut spectral pixel block out of the image
        height of the block = 1 + 2*bheight 
        width of the bloc = 1 + 2*bwidth 
        if idealPosition==True ... blocks from warped position (idealPosition)
        else                   ... blocks from original position (position)
        '''

        mySpec = np.zeros((self.idealPosition.shape[0], 2*bheigth+1,2*bwidth+1))

        
        if idealPosition:
            # get integer value of the ideal position        
            myPosition = np.round(self.idealPosition).astype(int)
        else:
            # get integer value of the position        
            myPosition = np.round(self.position).astype(int)

        # cut out blocks out of the raw image
        for ii in range(2*bwidth+1):
            for jj in range(2*bheigth+1):
                mySpec[:,jj,ii] = image[myPosition[:,0]+jj-bheigth,
                myPosition[:,1]+ii-bwidth]

        return mySpec

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
        
    myim = np.load(hsi.dataFolder + '\\' + 'filter_583nm.npy')
        
    imMo = ImageMorph(myim)
    imMo.getPixelPosition()
    imMo.getLatticeInfo()
    #imMo.getPixelIndexSimple()
    #indexSimple = imMo.imIdx
    imMo.getPixelIndex()
    indexAdvanced = imMo.imIdx

    imMo.getIdealLattice(onPixel=True)

    imMo.setWarpMatrix(pointGridStep=1)

    start = timer()
    idealImage = imMo.getWarpedImage(myim)
    end = timer()
    print('warping time: ', end - start, 's')

    # prepare the points
    pointsSelect = (imMo.imIdx[:,0]%10 == 0 ) & (imMo.imIdx[:,1]%10 == 0 )

    points = imMo.position[pointsSelect,:]
    pointsIdeal = imMo.idealPosition[pointsSelect,:]
    
    #featuresSimple = {'pointIndex0': indexSimple[pointsSelect,0],
    #            'pointIndex1': indexSimple[pointsSelect,1]
    #            }

    featuresAdvanced = {'pointIndex0': indexAdvanced[pointsSelect,0],
                'pointIndex1': indexAdvanced[pointsSelect,1]
                }
    text = {'string': '[{pointIndex0},{pointIndex1}]',
            'translation': np.array([-30, 0])
            }
    text2 = {'string':'{mytext}',
             'translation': np.array([-30, 0])}
    features2 = {'mytext': ['origin']}

    # prepare vectors
    vectors = np.zeros((points.shape[0],2,2))
    # delta vectors
    vectors[:,1,:] = pointsIdeal - points
    # origin of vectors
    vectors[:,0,:] = points

    # display the images
    viewer = napari.Viewer()
    viewer.add_image(myim)
    #viewer.add_points(points,features=featuresSimple,text=text, size= 50)
    viewer.add_points(pointsIdeal,features=featuresAdvanced,text=text, size= 50, opacity=0.5)
    viewer.add_points(points,features=featuresAdvanced,text=text, size= 50, face_color= 'green', opacity=0.5)
    #viewer.add_points(imMo.xy00[None,:], features=features2, size=50, text=text2, face_color= 'red', opacity= 0.5)
    #viewer.add_vectors(vectors)
    viewer.add_image(idealImage, opacity = 0.5)

    # show the spectra line distortion
    mySample = np.load(hsi.dataFolder + '\\' + 'sample.npy')
    idealSample = imMo.getWarpedImage(mySample)

    viewer2 = napari.Viewer()
    viewer2.add_image(mySample, opacity= 0.5)
    viewer2.add_image(idealSample, opacity= 0.5)
    viewer2.add_points(pointsIdeal,features=featuresAdvanced,text=text, size= 50, opacity=0.5)


    # overview aligned image
    mySpec = imMo.getSpectraBlock(idealImage,bheigth=2, bwidth=30)
    alignedIm = imMo.getAlignedImage(mySpec)

    viewer3 = napari.Viewer()
    viewer3.add_image(alignedIm, opacity= 0.5)

    # partially aligned image
    alignedCutIm = imMo.getAlignedImageCut(mySpec)

    viewer4 = napari.Viewer()
    viewer4.add_image(alignedCutIm, opacity= 0.5)

    # spectral image
    wyxImage = imMo.getWYXImage(mySpec)

    viewer5 = napari.Viewer()
    viewer5.add_image(wyxImage, opacity= 0.5)

    # comparison of spectral images (raw image, warped image off pixels, warped image on pixels)

    # raw
    mySpec = imMo.getSpectraBlock(mySample,bheigth=2, bwidth=30, idealPosition=False)
    #wyxImageRaw = imMo.getWYXImage(mySpec)
    wyxImageRaw = imMo.getAlignedImage(mySpec)


    # warped off pixel
    imMo.getIdealLattice(onPixel=False)
    imMo.setWarpMatrix()
    idealImageOffPixel = imMo.getWarpedImage(mySample)
    mySpec = imMo.getSpectraBlock(idealImageOffPixel,bheigth=2, bwidth=30, idealPosition=True)
    #wyxImageOffPixel = imMo.getWYXImage(mySpec)
    wyxImageOffPixel = imMo.getAlignedImage(mySpec)


    # warped on pixel
    imMo.getIdealLattice(onPixel=True)
    imMo.setWarpMatrix()
    idealImageOnPixel = imMo.getWarpedImage(mySample)
    mySpec = imMo.getSpectraBlock(idealImageOnPixel,bheigth=2, bwidth=30, idealPosition=True)
    #wyxImageOnPixel = imMo.getWYXImage(mySpec)
    wyxImageOnPixel = imMo.getAlignedImage(mySpec)

    viewer6 = napari.Viewer()
    viewer6.add_image(wyxImageRaw, opacity= 0.5)
    viewer6.add_image(wyxImageOffPixel, opacity= 0.5)
    viewer6.add_image(wyxImageOnPixel, opacity= 0.5)


    napari.run()

