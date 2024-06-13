'''
class to calibrate HIS image from three narrow band images
'''

import numpy as np
from copy import deepcopy

from skimage.transform import warp
from scipy.interpolate import griddata

from skimage.filters import threshold_otsu, median, gaussian
from skimage.morphology import disk
from skimage import measure

import spectralCamera
import pickle

from spectralCamera.algorithm.baseCalibrate import BaseCalibrate
from spectralCamera.algorithm.gridSuperPixel import GridSuperPixel

class CalibrateFrom3Images(BaseCalibrate):
    ''' main class to calibrate IF camera from three single wavelength images '''

    DEFAULT = {'imageNameStack':['filter_600_1', 'filter_525_0','filter_700_0'],
               'wavelengthStack': [600,525,700],
               'bwidth': 30,
               'bheight': 2,
               'spectralRange': [450,720], # spectral range of the camera - determine the length of the spectral box
                'cosmicRayThreshold': 1e6,
               }

    def __init__(self,imageNameStack=None, wavelengthStack=None,**kwargs):
        ''' initialise the class 
        imageNameStack ... list of .npy files with the calibration images (names without .npy extension)
        wavelengthStack ... list with wavelength '''
        super().__init__(**kwargs)


        self.imageNameStack = None
        self.wavelengthStack = None
        self.imageStack = None
        self.imMoStack = []
        self.gridLine = None

        self.positionMatrixStack = None
        self.boolMatrixStack = None
        self.positionMatrix = None
        self.boolMatrix = None

        self.dSpectralWarpMatrix = None
        self.dSubpixelWarpMatrix = None
        self.warpMatrix = None
        self.xx = None
        self.yy = None

        self.pixelPositionWavelength = None

        #self.bwidth = self.DEFAULT['bwidth']
        #self.bheight = self.DEFAULT['bheight']

        self.bwidth = None
        self.bheight = None

        if imageNameStack is None:
            self.imageNameStack = self.DEFAULT['imageNameStack']
        else:
            self.imageNameStack = imageNameStack

        if wavelengthStack is None:
            self.wavelengthStack = np.array(self.DEFAULT['wavelengthStack'])
        else:
            self.wavelengthStack = np.array(wavelengthStack)

    def _getPeakPosition(self,image):
        ''' get position of the spectral spots from the image
            return the position of peaks and the average peak width
        
        '''
       # smoothing
        image = gaussian(image)

        # remove cosmic ray/over-saturated images
        medianIm = median(image, disk(1))
        cosmicRayMask = image> self.DEFAULT['cosmicRayThreshold']
        image[cosmicRayMask] = medianIm[cosmicRayMask]

       # smoothing
        image = gaussian(image)

        # identify the spectral spots
        threshold = threshold_otsu(image)
        mask = image > threshold

        # calculate properties of objects and get the median size of the spots
        labels = measure.label(mask)
        component_sizes = np.bincount(labels.ravel())
        myArea = np.median(component_sizes)

        # remove too small and too large objects
        thresholdFactor = 2
        mySelection = np.logical_or((component_sizes < myArea/(1 +  thresholdFactor)),
                                    (component_sizes > myArea*(1 + thresholdFactor)))
        myMask = mySelection[labels]
        mask[myMask] = 0

        # calculate position of the spots
        labels = measure.label(mask)
        props = measure.regionprops_table(labels, image,
                properties=['centroid','axis_minor_length'])

        peakPosition = np.vstack((props['centroid-0'],props['centroid-1'])).T

        # this is the width / spread of the peak along the y-axis 
        # (it is independent on the width of the spectral filter)
        peakWidth = np.mean(props['axis_minor_length'])

        return (peakPosition, peakWidth)

    def setImageStack(self,imageStack=None,wavelengthStack=None):
        ''' set image stack with the corresponding wavelength
        if not provided, the images are loaded from default files'''

        if imageStack is not None:
            self.imageStack = imageStack
            self.wavelengthStack = wavelengthStack
            return
        else:
            self.imageStack = []
            for imageName in self.imageNameStack:
                myIm = np.load(spectralCamera.dataFolder + '\\' + imageName + '.npy')
                self.imageStack.append(myIm)

 
    def processImageStack(self):
        ''' process the single images and get their lattices '''

        for myIm in self.imageStack:

            (peakPosition, peakWidth) = self._getPeakPosition(myIm)

            # get indexing of the peaks
            imMo = GridSuperPixel()
            imMo.setGridPosition(peakPosition)
            imMo.getGridInfo()
            imMo.getPixelIndex()
           
            self.imMoStack.append(imMo)
       
        self.bwidth = int(peakWidth//2)

    def _saveGridStack(self):
        ''' save grid stacks
        this is only for debugging purposes. it speed up the process '''

        for ii,imMo in enumerate(self.imMoStack):
            file = open(spectralCamera.dataFolder + '\\' + self.imageNameStack[ii] + '.obj', 'wb') 
            pickle.dump(imMo, file)
            file.close()            

    def _loadGridStack(self):
        ''' load grids obtained from the imageNameStack
         this is only for debugging purposes. it speed up the process '''

        self.imMoStack = []

        for imMoName in self.imageNameStack:
            self.imMoStack.append(pickle.load(open(spectralCamera.dataFolder + '\\' + imMoName + '.obj', 'rb')))

    def _setGlobalGridZero(self):
        ''' set [0,0] position in all grid in the same area. Grid Zeros centered around  first imageNameStack Grid
        algorithm:
            find first rangeIdxMax**2 grid points around [0,0]
            the new [0,0] is the one which is aligned either right [shorter Wavelength] or left [longer Wavelength]
            from the given [0,0] (first imMoStack) and the maximal x-distance is distanceMax
          
        '''

        rangeIdxMax = 10
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

        # old style - use the variables positionMatrix, boolMatrix
        self.positionMatrixStack = positionMatrixStack
        self.boolMatrixStack = boolMatrixStack

        
        self.positionMatrix = np.array(self.positionMatrixStack)
        _boolMatrix = np.array(self.boolMatrixStack)
        self.boolMatrix = np.prod(_boolMatrix, axis=0).astype('bool')


    def getPixelPositionWavelength(self):
        ''' calculate average position on the spots for different wavelength'''

        # calculate relative shift
        vectorMatrixShift10 = self.positionMatrix[1,...] - self.positionMatrix[0,...]
        vectorMatrixShift20 = self.positionMatrix[2,...] - self.positionMatrix[0,...]

        # get the mean of the shift 
        meanXShift10= np.mean(vectorMatrixShift10[1,self.boolMatrix])
        meanXShift20= np.mean(vectorMatrixShift20[1,self.boolMatrix])

        # vector for spectral to pixel calibration
        self.pixelPositionWavelength = np.array([0, meanXShift10, meanXShift20 ])

        return self.pixelPositionWavelength


    def setGridLine(self):
        ''' set the global grid with for obtaining spectral blocks'''

        # define the global gridSuperPixel 
        self.gridLine = deepcopy(self.imMoStack[0])

        wavelengthFit = np.poly1d(np.polyfit(self.pixelPositionWavelength,self.wavelengthStack, 2))
        
        
        
        return wavelengthFit(np.arange(2*self.bwidth+1)-self.bwidth)



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

#%%

    import napari
    import spectralCamera
    import numpy as np
    from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
    # load reference image
    whiteImage = np.load(spectralCamera.dataFolder + '\\' + 'filter_wo_0.npy')

    # initiate the calibration class
    myCal = CalibrateFrom3Images()

    myCal.setImageStack()

    if False:
        myCal.processImageStack()
        #myCal._setGlobalGridZero()
        myCal._saveGridStack()
    else:
        myCal._loadGridStack()


    myCal._setGlobalGridZero()
    #%% show the calibration images
    viewer = napari.Viewer()
    #viewer.add_image(myCal.mask)
    viewer.add_image(whiteImage, name='white')

    for ii,iS in enumerate(myCal.imageStack):
        viewer.add_image(iS,name = myCal.wavelengthStack[ii], opacity=0.5)


    # %% show selected points
   
    mask = np.zeros_like(myCal.imageStack[0])

    for ii,imMo in enumerate(myCal.imMoStack):
        selectPoint = (imMo.imIdx[:,0]%2 == 0 ) & (imMo.imIdx[:,1]%2 == 0 )

        mask[imMo.position[selectPoint,0].astype(int),imMo.position[selectPoint,1].astype(int)] = ii+1

    viewer.add_image(mask,name= 'position')

    # %% show zero points
    point00 = []
    for ii,imMo in enumerate(myCal.imMoStack):
        point00.append(imMo.xy00)
        print(f'for {ii} the xy00 is {imMo.xy00}')
    point00 = np.array(point00).T
    viewer.add_points(point00.T, size= 50, opacity=0.2, name= 'zero position')

    # %% show position Matrix

    myCal._setPositionMatrix()

    viewer = napari.Viewer()
    viewer.add_image(np.array(myCal.positionMatrixStack), name='position Matrix')

    #viewer.add_image(np.array(myCal.boolMatrixStack), name='position Matrix bool')



    #%%


    im525Mask = im525*0
    im525Mask[pos525[:,0].astype(int),pos525[:,1].astype(int)] = 1
    viewer.add_image(im525Mask)
    viewer.add_image(im525, name='525')

    #%%


    if True:
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































