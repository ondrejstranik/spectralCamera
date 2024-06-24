'''
class to calibrate HIS image from three narrow band images
'''

#%%
import numpy as np
from copy import deepcopy

from skimage.transform import warp
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

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
               'wavelengthStack': [602,534,705],
               'bheight': 2,
               'spectralRange': [525,700], # spectral range of the camera - determine the length of the spectral box
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

        # indexing matrix - necessary for alignment of self.imMoStack 
        self.positionMatrix = None
        self.boolMatrix = None
        self.idxMatrixOffset = None

        self.dSpectralWarpMatrix = None
        self.dSubpixelShiftMatrix = None
        self.warpMatrix = None


        self.bwidth = None
        self.bheight = self.DEFAULT['bheight']

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
       
        self.bheight = int(peakWidth//2)

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
        # internally defined parameters
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
        self.idxMatrixOffset = imIdx.min(axis=0)
        idxMatrixSize = imIdx.max(axis=0) - self.idxMatrixOffset +1

        # convert the grid position list to grid position matrix
        # this allow to operate on the grid with the same index
        positionMatrixStack = []
        boolMatrixStack = []

        for imMo in self.imMoStack:
            pointMatrix = np.zeros((2,*idxMatrixSize))
            pointMatrix[:,imMo.imIdx[:,0] -self.idxMatrixOffset[0],
                imMo.imIdx[:,1]-self.idxMatrixOffset[1]] = imMo.position.T
            boolMatrix = np.zeros(idxMatrixSize).astype('bool')
            boolMatrix[imMo.imIdx[:,0] -self.idxMatrixOffset[0],
                imMo.imIdx[:,1]-self.idxMatrixOffset[1]] = True
            
            positionMatrixStack.append(pointMatrix)
            boolMatrixStack.append(boolMatrix)

        self.positionMatrix = np.array(positionMatrixStack)
        self.boolMatrix = np.prod(np.array(boolMatrixStack), axis=0).astype('bool')

    def setGridLine(self,spectralRange= None):
        ''' set the global superGrid for obtaining spectral blocks'''

        def F(pixel,A,B,C):
            return A*np.exp(B*pixel) + C

        def iF(wavelength,A,B,C):
            return 1/B*np.log((wavelength- C)/A)

        ''' calculate average position on the spots for different wavelength'''
        # calculate relative shift
        vectorMatrixShift10 = self.positionMatrix[1,...] - self.positionMatrix[0,...]
        vectorMatrixShift20 = self.positionMatrix[2,...] - self.positionMatrix[0,...]

        # get the mean of the shift 
        meanXShift10= np.mean(vectorMatrixShift10[1,self.boolMatrix])
        meanXShift20= np.mean(vectorMatrixShift20[1,self.boolMatrix])

        # vector for spectral to pixel calibration
        self.pixelPositionWavelength = np.array([0, meanXShift10, meanXShift20 ])


        if spectralRange is None:
            spectralRange = self.DEFAULT['spectralRange']
        else:
            spectralRange = np.array(spectralRange)

        # define the size of the spectralBlock, and calibration
        # fit with polynomial ...wavelength =  const + A*1/pixel  + B/pixel**2
        wavelengthFitP,_ = curve_fit(f = F, xdata = self.pixelPositionWavelength, ydata = self.wavelengthStack, bounds=((0,-np.inf,0),(np.inf,0,np.inf)))
        self.wavelengthFit = lambda x: F(x,*wavelengthFitP)
        self.pixelFit = lambda x: iF(x,*wavelengthFitP)

        self.bwidth = int(np.abs(self.pixelFit(spectralRange[0]) - self.pixelFit(spectralRange[1]))//2)
        self.xShift = (self.pixelFit(spectralRange[0]) + self.pixelFit(spectralRange[1]))//2 # shift from the self.imMoStack[0].position

        self.wavelength = self.wavelengthFit(np.arange(2*self.bwidth+1)-self.bwidth+self.xShift)


        # define the global gridSuperPixel 
        # set only where all three calibration peak were identified 
        self.gridLine = GridSuperPixel()
        self.gridLine.setGridPosition(self.positionMatrix[0,:,self.boolMatrix] + np.array([0,self.xShift]))
        self.gridLine.xVec = self.imMoStack[0].xVec
        self.gridLine.yVec = self.imMoStack[0].yVec
        self.gridLine.imIdx = np.argwhere(self.boolMatrix) + self.idxMatrixOffset
        self.gridLine.shiftIdx00([0,0])

        # indicate spectral block outside of the image
        self.gridLine.getPositionOutsideImage(self.imageStack[0],self.bheight,self.bwidth)

    def prepareGrid(self,spectralRange=None):
        ''' prepare the grids from different calibration, so that warping can be calculated'''
        self.processImageStack()
        self._setGlobalGridZero()
        self._setPositionMatrix()
        self.setGridLine(spectralRange=spectralRange)

    def _setSpectralWarpMatrix(self):
        ''' calculate the chromatic distortion matrix from vectors shifts
        and vector for pixel to spectral calibration
        '''

        # calculate relative shift
        vectorShift10 = (self.positionMatrix[1,:,self.boolMatrix] 
                                - self.positionMatrix[0,:,self.boolMatrix])
        vectorShift20 = (self.positionMatrix[2,:,self.boolMatrix] 
                                - self.positionMatrix[0,:,self.boolMatrix])

        # get the mean of the shift 
        meanShift10= np.mean(vectorShift10, axis=0)
        meanShift20= np.mean(vectorShift20, axis=0)

        # deviation from the ideal position (y .. tilting, x ... stretching) for the two wavelength 
        dVectorShift10 = vectorShift10 -  meanShift10
        dVectorShift20 = vectorShift20 -  meanShift20

        # pixel position for the two wavelength  
        px1 = np.argmin(np.abs(self.wavelength - self.wavelengthStack[1]))
        px2 = np.argmin(np.abs(self.wavelength - self.wavelengthStack[2]))

        def linFit(px):
            # linear fit of the deviation for given pixel in the blocks
            slope = ((dVectorShift20 - dVectorShift10) / (px2 - px1) )
            return  slope*(px - px1) + dVectorShift10

        # populate the spectralWarp Matrix
        vx = np.zeros_like(self.imageStack[0])
        vy = np.zeros_like(self.imageStack[0])
        myPos = self.gridLine.getPositionInt()

        for ii in range(2*self.bheight+1):
            for jj in range(2*self.bwidth+1):
                vx[(myPos[:,0]+ii-self.bheight).astype(int),
                (myPos[:,1]+jj-self.bwidth).astype(int)] = linFit(jj)[:,1]
                vy[(myPos[:,0]+ii-self.bheight).astype(int),
                (myPos[:,1]+jj-self.bwidth).astype(int)] = linFit(jj)[:,0]

        self.dSpectralWarpMatrix = np.array([vy,vx])


    def _setSubpixelShiftMatrix(self):
        ''' calculate matrix to shift original spots on integer grids '''

        # generate blocks with constant shift 
        vx = np.zeros_like(self.imageStack[0])
        vy = np.zeros_like(self.imageStack[0])

        vectorSubpixelShift = self.gridLine.position - np.round(self.gridLine.position)
        myPos = self.gridLine.getPositionInt()

        for ii in range(2*self.bheight+1):
            for jj in range(2*self.bwidth+1+2):
                vx[(myPos[:,0]+ii-self.bheight).astype(int),
                (myPos[:,1]+jj-self.bwidth-1).astype(int)] = vectorSubpixelShift[:,1]
                vy[(myPos[:,0]+ii-self.bheight).astype(int),
                (myPos[:,1]+jj-self.bwidth-1).astype(int)] = vectorSubpixelShift[:,0]

        self.dSubpixelShiftMatrix = np.array([vy,vx])

    def setWarpMatrix(self,spectral=True, subpixel=True):
        ''' set the final warping matrix
        spectral == True ... correct for the bending of hte spectral lines
        subpixel == True ... shift the initial calibration wavelength spots on full pixels
        '''
        # define the grid points and warp matrix
        xx, yy = np.meshgrid(np.arange(self.imageStack[0].shape[1]),
                                        np.arange(self.imageStack[0].shape[0]))
        self.warpMatrix = np.array([yy,xx])

        if spectral:
            if self.dSpectralWarpMatrix is None: self._setSpectralWarpMatrix()
            self.warpMatrix = self.warpMatrix + self.dSpectralWarpMatrix

        if subpixel:
            if self.dSubpixelShiftMatrix is None: self._setSubpixelShiftMatrix()
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

        return self.gridLine.getSpectraBlock(image,bheight=self.bheight, bwidth=self.bwidth)

    def getSpectralBlockImage(self):
        ''' for visual check.
        wrapper for gridSuperPixel.getSpectralBlockImage'''

        return self.gridLine.getSpectralBlockImage(self.imageStack[0],
                                                   bheight=self.bheight,
                                                   bwidth=self.bwidth)


    def getAlignedImage(self,mySpec):
        ''' get the aligned spectral blocks as an image
        wrapper for a method from the imageMorph class
        '''
        return self.gridLine.getAlignedImage(mySpec)

    def getWYXImage(self,mySpec):
        ''' get the spectral image
        wrapper for a method from the imageMorph class
        '''
        return self.gridLine.getWYXImage(mySpec)

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

#%%

if __name__ == "__main__":
    pass
