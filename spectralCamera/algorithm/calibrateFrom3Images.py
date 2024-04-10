'''
class to calibrate HIS image from three narrow band images
'''

import numpy as np
from skimage.transform import warp
from scipy.interpolate import griddata

from HSIplasmon.algorithm.imageMorph import ImageMorph
import HSIplasmon as hsi
import pickle

import napari


class CalibrateFrom3Images():
    ''' main class to calibrate optical and spectral aberration from three images '''

    DEFAULT = {'imageNameStack':['filter_583nm', 'filter_500nm','filter_700nm'],
               'wavelengthStack': [583,500,700],
               'classFile': 'CalibrateFrom3Images',
               'bwidth': 30,
               'bheight': 2}


    def __init__(self,imageNameStack=None, wavelengthStack=None):
        ''' initialise the class 
        imageNameStack ... list of .npy files with the calibration images (names without .npy extension)
                        TODO: add folder options (now files in folder hsi.dataFolder)
        wavelengthStack ... list with '''
        
        self.imageNameStack = None
        self.wavelengthStack = None

        self.imMoStack = []

        self.positionMatrixStack = None
        self.boolMatrixStack = None

        self.dSpectralWarpMatrix = None
        self.dSubpixelWarpMatrix = None
        self.warpMatrix = None
        self.xx = None
        self.yy = None

        self.pixelPositionWavelength = None

        self.bwidth = self.DEFAULT['bwidth']
        self.bheight = self.DEFAULT['bheight']
        
        if imageNameStack is None:
            self.imageNameStack = self.DEFAULT['imageNameStack']
        else:
            self.imageNameStack = imageNameStack

        if wavelengthStack is None:
            self.wavelengthStack = np.array(self.DEFAULT['wavelengthStack'])
        else:
            self.wavelengthStack = np.array(wavelengthStack)

    def processImageStack(self, save=True):
        ''' process the single images and get their lattices
        save == True ... save the process images in .obj files'''

        for imageName in self.imageNameStack:
            myIm = np.load(hsi.dataFolder + '\\' + imageName + '.npy')
            imMo = ImageMorph(myIm)
            imMo.getPixelPosition()
            imMo.getLatticeInfo()
            imMo.getPixelIndex()
            #imMo.getIdealLattice(onPixel=False)
            #imMo.setWarpMatrix()
            file = open(hsi.dataFolder + '\\' + imageName + '.obj', 'wb') 
            pickle.dump(imMo, file)
            file.close()

            self.imMoStack.append[imMo]

    def loadGrids(self):
        ''' load grids obtained from the imageStacks '''

        self.imMoStack = []

        for imMoName in self.imageNameStack:
            self.imMoStack.append(pickle.load(open(hsi.dataFolder + '\\' + imMoName + '.obj', 'rb')))

    def saveClass(self, classFile=None):
        ''' save the class to a file '''
        if classFile is None:
            classFile = self.DEFAULT['classFile']

        file = open(hsi.dataFolder + '\\' + classFile + '.obj', 'wb') 
        pickle.dump(self, file)
        file.close()

    def loadClass(self,classFile=None):
        ''' load the class itself from file '''
        # it is necessary in order to unpickle not only from __main__
        # alternatively use a package dill
        import __main__
        __main__.CalibrateFrom3Images = CalibrateFrom3Images

        if classFile is None:
            classFile = self.DEFAULT['classFile']


        return pickle.load(open(hsi.dataFolder + '\\' + classFile + '.obj', 'rb'))


    def _setGlobalGridZero(self):
        ''' set [0,0] position in all grid in the same area. Grid Zeros centered around  first imageNameStack Grid
        algorithm:
            find first rangeIdxMax**2 grid points around [0,0]
            the new [0,0] is the one which is aligned either right [shorter Wavelength] or left [longer Wavelength]
            from the given [0,0] (first imageNameStack) and the maximal x-distance is distanceMax
          
        '''

        rangeIdxMax = 5
        distanceMax = 50
        
        imMo0 = self.imMoStack[0]

        for ii,imMo in enumerate(self.imMoStack[1:]):

            # select only range around [0,0]
            pointsSelect = ((imMo.imIdx[:,0] < rangeIdxMax ) & (imMo.imIdx[:,0] > -rangeIdxMax ) & 
                            (imMo.imIdx[:,1]<rangeIdxMax) & (imMo.imIdx[:,1] >- rangeIdxMax ))

            # calculate the differences in the position
            xdiff = imMo.position[pointsSelect,1]-imMo0.xy00[1]
            ydiff = imMo.position[pointsSelect,0]-imMo0.xy00[0]

            # if longer wavelength then original use the grid left from the original
            if self.wavelengthStack[ii+1]>self.wavelengthStack[0]:
                xdiff = -xdiff

            # select points only for given x distance
            pointSelect2 = ((xdiff > 0 ) & (xdiff < distanceMax ))                

            # find the new zero, which is mostly aligned on the horizontal axis with the reference zero
            _new00idx = np.argmin(np.abs(ydiff[pointSelect2]))
            new00idx = imMo.imIdx[pointsSelect][pointSelect2][_new00idx]

            # shift the zero
            imMo.shiftIdx00(new00idx)

    def _setPositionMatrix(self):
        ''' define the position matrix in order to operate on the grids data from different file
        it transform the list of position to matrix of position.
        '''

        # define the size indexing matrix 
        imIdx = np.vstack((self.imMoStack[0].imIdx,self.imMoStack[1].imIdx,self.imMoStack[2].imIdx))
        idxMatrixOffset = imIdx.min(axis=0)
        idxMatrixSize = imIdx.max(axis=0) - idxMatrixOffset +1

        # convert the grid position list to grid position matrix
        # this allow to operate on the grid with the same index
        positionMatrixStack = []
        boolMatrixStack = []

        for imMo in self.imMoStack:
            pointMatrix = np.zeros((2,*idxMatrixSize))
            pointMatrix[:,imMo.imIdx[:,0] -idxMatrixOffset[0],
                imMo.imIdx[:,1]-idxMatrixOffset[1]] = imMo.position.T
            boolMatrix = np.zeros(idxMatrixSize).astype('bool')
            boolMatrix[imMo.imIdx[:,0] -idxMatrixOffset[0],
                imMo.imIdx[:,1]-idxMatrixOffset[1]] = True
            
            positionMatrixStack.append(pointMatrix)
            boolMatrixStack.append(boolMatrix)

        # define them in the class
        self.positionMatrixStack = positionMatrixStack
        self.boolMatrixStack = boolMatrixStack

    def prepareGrids(self):
        ''' prepare the grids from different calibration, so that warping can be calculated'''
        self._setGlobalGridZero()
        self._setPositionMatrix()

    def _setSpectralWarpMatrix(self):
        ''' calculate the chromatic distortion matrix from vectors shifts
        and vector for pixel to spectral calibration
        '''

        #get the relative shift in the grid positions for different wavelength

        # calculate relative shift
        vectorMatrixShift10 = self.positionMatrixStack[1] - self.positionMatrixStack[0]
        vectorMatrixShift20 = self.positionMatrixStack[2] - self.positionMatrixStack[0]
        # set shift to zero, where the values are not defined
        vectorMatrixShift10[:,~(self.boolMatrixStack[1] & self.boolMatrixStack[0])] = 0
        vectorMatrixShift20[:,~(self.boolMatrixStack[2] & self.boolMatrixStack[0])] = 0

        # get the mean spectral (x) shift
        meanXShift10= np.mean(vectorMatrixShift10[1,(self.boolMatrixStack[1] & self.boolMatrixStack[0])])
        meanXShift20= np.mean(vectorMatrixShift20[1,(self.boolMatrixStack[2] & self.boolMatrixStack[0])])
        # subtract the mean spectral (x) shifts
        vectorMatrixShift10[1,(self.boolMatrixStack[1] & self.boolMatrixStack[0])] -= meanXShift10
        vectorMatrixShift20[1,(self.boolMatrixStack[2] & self.boolMatrixStack[0])] -= meanXShift20

        # prepare the all points and the relative shift vectors
        # the original grid points should be zero
        points00 = self.imMoStack[0].position
        vector00 = np.zeros_like(points00)

        points10 = self.positionMatrixStack[1][:,(self.boolMatrixStack[1] & self.boolMatrixStack[0])].T
        vector10 = vectorMatrixShift10[:,(self.boolMatrixStack[1] & self.boolMatrixStack[0])].T

        points20 = self.positionMatrixStack[2][:,(self.boolMatrixStack[2] & self.boolMatrixStack[0])].T
        vector20 = vectorMatrixShift20[:,(self.boolMatrixStack[2] & self.boolMatrixStack[0])].T

        # put them all together
        points = np.vstack((points00,points10,points20))
        vector = np.vstack((vector00,vector10,vector20))

        # define the grid points
        self.xx, self.yy = np.meshgrid(np.arange(self.imMoStack[0].image.shape[1]),np.arange(self.imMoStack[0].image.shape[0]))

        # interpolate the shift on all pixel in the image
        vy = griddata(points, vector[:,0], (self.yy, self.xx), method='cubic', fill_value= 0)
        vx = griddata(points, vector[:,1], (self.yy, self.xx), method='cubic', fill_value= 0)

        self.dSpectralWarpMatrix = np.array([vy,vx])

        # vector for spectral to pixel calibration
        self.pixelPositionWavelength = np.array([0, meanXShift10, meanXShift20 ])

    def _setSubpixelShiftMatrix(self):
        ''' calculate matrix to shift original spots on integer grids '''

        # calculate relative shift
        vectorMatrixSubpixelShift = self.positionMatrixStack[0] - np.round(self.positionMatrixStack[0])

        # define it only where all there wavelengths are defined
        boolMatrixAll = (self.boolMatrixStack[2] & self.boolMatrixStack[1] & self.boolMatrixStack[0])

        points0 = self.positionMatrixStack[0][:,boolMatrixAll].T
        points1 = self.positionMatrixStack[1][:,boolMatrixAll].T
        points2 = self.positionMatrixStack[2][:,boolMatrixAll].T
        vector_ = vectorMatrixSubpixelShift[:,boolMatrixAll].T

        # put them all together
        points = np.vstack((points0,points1,points2))
        vector = np.vstack((vector_,vector_,vector_))

        # define the grid points
        self.xx, self.yy = np.meshgrid(np.arange(self.imMoStack[0].image.shape[1]),np.arange(self.imMoStack[0].image.shape[0]))

        # interpolate the shift on all pixel in the image
        vy = griddata(points, vector[:,0], (self.yy, self.xx), method='cubic', fill_value= 0)
        vx = griddata(points, vector[:,1], (self.yy, self.xx), method='cubic', fill_value= 0)

        self.dSubpixelShiftMatrix = np.array([vy,vx])

    def setWarpMatrix(self,spectral=True, subpixel=True):
        ''' set the final warping matrix
        spectral == True ... correct for the bending of hte spectral lines
        subpixel == True ... shift the initial calibration wavelength spots on full pixels
        '''



        # necessary to calculate in order to get the wavelength calibration
        self._setSpectralWarpMatrix()

        if spectral:
            self.warpMatrix = self.warpMatrix + self.dSpectralWarpMatrix

        if subpixel:
            self._setSubpixelShiftMatrix()
            self.warpMatrix = self.warpMatrix + self.dSubpixelShiftMatrix

    def getWarpedImage(self, image):
        ''' make the warping on the image'''
        warpedImage = warp(image, self.warpMatrix, mode='edge')        
        return warpedImage

    def getSpectraBlock(self,image,bheight=None, bwidth=None):
        ''' get the spectral blocks out of the image
        wrapper for a method from the imageMorph class 
        '''
        if bheight is not None:
            self.bheight = bheight

        if bwidth is not None:
            self.bwidth = bwidth

        return self.imMoStack[0].getSpectraBlock(image,bheigth=self.bheight, bwidth=self.bwidth, idealPosition=False)

    def getAlignedImage(self,mySpec):
        ''' get the aligned spectral blocks as an image
        wrapper for a method from the imageMorph class
        '''
        return self.imMoStack[0].getAlignedImage(mySpec)

    def getWYXImage(self,mySpec):
        ''' get the spectral image
        wrapper for a method from the imageMorph class
        '''
        return self.imMoStack[0].getWYXImage(mySpec)
    
    def getWavelength(self):
        ''' get the wavelength vector defined on the range of 2*bwidth+1
        use parabolic fit on the 3 calibration data'''
        wavelengthFit = np.poly1d(np.polyfit(self.pixelPositionWavelength,self.wavelengthStack, 2))
        return wavelengthFit(np.arange(2*self.bwidth+1)-self.bwidth)

    def getSpectralImage(self,rawImage,aberrationCorrection=False):
        ''' get the spectral image from raw image
        aberrationCorrection == False ... no image aberration applied
        it is just wrapper function
        '''

        if aberrationCorrection:
            warpedImage = self.getWarpedImage(rawImage)
        else:
            warpedImage = rawImage

        mySpec = self.getSpectraBlock(warpedImage)
        return self.getWYXImage(mySpec)

if __name__ == "__main__":

    # load reference image
    rawImage = np.load(hsi.dataFolder + '\\' + 'white_light.npy')

    # initiate the calibration class
    myCal = CalibrateFrom3Images()
    
    
    if False:
        # calculate the calibration matrix
        myCal.loadGrids()
        myCal.prepareGrids()
        myCal.setWarpMatrix(spectral=True, subpixel=True)
        myCal.saveClass()
    else:
        # load already calculated calibration matrix
        myCal = myCal.loadClass()

    # spec subpixel warped image
    warpedImageSpecSub = myCal.getWarpedImage(rawImage)
    # spec warped image
    myCal.warpMatrix = np.array([myCal.yy,myCal.xx]) + myCal.dSpectralWarpMatrix
    warpedImageSpec = myCal.getWarpedImage(rawImage)
    # subpixel warped image
    myCal.warpMatrix = np.array([myCal.yy,myCal.xx]) + myCal.dSubpixelShiftMatrix
    warpedImageSub = myCal.getWarpedImage(rawImage)

    # show the warping matrices
    viewer = napari.Viewer()
    viewer.add_image(myCal.dSpectralWarpMatrix[1,...], opacity= 0.5, name= 'dSpectralWarpMatrix_x')
    viewer.add_image(myCal.dSpectralWarpMatrix[0,...], opacity= 0.5, name= 'dSpectralWarpMatrix_y')
    viewer.add_image(myCal.dSubpixelShiftMatrix[1,...], opacity= 0.5, name= 'dSubPixelShiftMatrix_x')
    viewer.add_image(myCal.dSubpixelShiftMatrix[0,...], opacity= 0.5, name= 'dSubPixelShiftMatrix_y')

    # show the warped images
    viewer = napari.Viewer()
    viewer.add_image(rawImage, opacity=0.5, name= 'raw')
    viewer.add_image(warpedImageSpecSub, opacity= 0.5, name= 'both')
    viewer.add_image(warpedImageSpec, opacity= 0.5, name='spec')
    viewer.add_image(warpedImageSub, opacity= 0.5, name= 'subPixel')

    # calculate block-aligned and 3D images
    mySpec = myCal.getSpectraBlock(rawImage)
    rawAlignedImage = myCal.getAlignedImage(mySpec)
    raw3DImage = myCal.getWYXImage(mySpec)

    mySpec = myCal.getSpectraBlock(warpedImageSpecSub)
    warpedAlignedImageSpecSub = myCal.getAlignedImage(mySpec)
    warped3DImageSpecSub = myCal.getWYXImage(mySpec)

    mySpec = myCal.getSpectraBlock(warpedImageSpec)
    warpedAlignedImageSpec = myCal.getAlignedImage(mySpec)
    warped3DImageSpec = myCal.getWYXImage(mySpec)

    mySpec = myCal.getSpectraBlock(warpedImageSub)
    warpedAlignedImageSub = myCal.getAlignedImage(mySpec)
    warped3DImageSub = myCal.getWYXImage(mySpec)

    # show block-aligned and 3D images

    viewer2 = napari.Viewer()
    viewer2.add_image(rawAlignedImage, opacity= 0.5)
    viewer2.add_image(warpedAlignedImageSpecSub, opacity= 0.5)
    viewer2.add_image(warpedAlignedImageSpec, opacity= 0.5)
    viewer2.add_image(warpedAlignedImageSub, opacity= 0.5)

    viewer3 = napari.Viewer()
    viewer3.add_image(raw3DImage, opacity= 0.5)
    viewer3.add_image(warped3DImageSpecSub, opacity= 0.5)
    viewer3.add_image(warped3DImageSpec, opacity= 0.5)
    viewer3.add_image(warped3DImageSub, opacity= 0.5)

    # show the calibrated spectral image in spectral viewer
    from HSIplasmon.SpectraViewerModel2 import SpectraViewerModel2
    wavelength = myCal.getWavelength()
    sViewer = SpectraViewerModel2(warped3DImageSpecSub, wavelength)


    napari.run()































