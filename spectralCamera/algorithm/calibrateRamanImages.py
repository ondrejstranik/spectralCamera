'''
class to calibrate Raman image from one image of homogenous image of polymer polystyrene
'''
#%%

import numpy as np

from skimage import data, filters, measure, morphology
from skimage.filters import threshold_otsu, rank, median
from skimage.morphology import disk
from skimage.transform import warp

from spectralCamera.algorithm.baseCalibrate import BaseCalibrate
from spectralCamera.algorithm.gridSuperPixel import GridSuperPixel

import spectralCamera
import pickle

class CalibrateRamanImages(BaseCalibrate):
    ''' main class to calibrate Raman images '''

    DEFAULT = {'kPosition': [1002, 1659],
                'polymer': 'polystyrene',
                'classFile': 'CalibrateRamanImages.obj',
                'cosmicRayThreshold': 10000}

    def __init__(self,image=None,darkImage=None):
        ''' class definition '''

        super().__init__(**kwargs)

        self.image = image
        self.darkImage = darkImage
        
        self.whitePixel = None
        self.gridLine = None
        self.spBlock = None # indicative spectral blocks for location the two main Raman peaks

        self.bwidth = None
        self.bheight = None

        self.mask = None
        self.peakLeftPosition = None
        self.peakRightPosition = None

        self.warpMatrix = None
        self.dSubpixelShiftMatrix = None
        self.dSpectralWarpMatrix = None 


    def _identifySpectralLine(self):
        ''' identify spectral line from the image'''

        myim = self.image
        medianIm = median(myim, disk(1))

        # remove white pixels if dark Image provided
        if self.darkImage is not None:
            wpThreshold = np.mean(self.darkImage) + 4*np.std(self.darkImage)
            #wpThreshold = np.mean(self.darkImage) + 10*np.std(self.darkImage)
            self.whitePixel = (self.darkImage>wpThreshold)
            myim[self.whitePixel] = medianIm[self.whitePixel]

        # remove cosmic ray/oversaturated images
        cosmicRayMask = myim> self.DEFAULT['cosmicRayThreshold']
        myim[cosmicRayMask] = medianIm[cosmicRayMask]

        # identify the spectral line
        threshold = filters.threshold_otsu(myim)
        mask = myim > threshold

        # connect broken spectral lines
        footprint = np.ones(10)[None,:]
        mask2 = morphology.binary_closing(mask,footprint=footprint)

        # remove small spots
        mask3 = morphology.remove_small_objects(mask2, 50)

        self.mask = mask3

        # calculate properties of objects
        labels = measure.label(self.mask)
        props = measure.regionprops_table(labels, myim,
                properties=['label', 'centroid','orientation', 'axis_major_length','axis_minor_length', 'intensity_max'])


        _position = np.vstack((props['centroid-0'],props['centroid-1'])).T

        bwidth = np.round(np.mean(props['axis_major_length']//2)).astype(int)
        bheight = np.round(np.mean(props['axis_minor_length']//2)).astype(int)

        self.bheight = bheight
        self.bwidth = bwidth

        self.gridLine = GridSuperPixel()
        self.gridLine.setGridPosition(_position)
        self.gridLine.getGridInfo()
        self.gridLine.getPixelIndex()

        
        secureXOffset = 10
        self.gridLine.getPositionOutsideImage(myim,bheight=bheight,bwidth=bwidth+secureXOffset)
        self.spBlock = self.gridLine.getSpectraBlock(myim,bheight=bheight,bwidth=bwidth)

    def _getLeftMax(self):
        ''' get position of the strongest peak on the left '''

        # left spectral lines averaged over y
        mySpec = np.mean(self.spBlock,axis=1)
        # block the right side
        mySpec[:,mySpec.shape[1]//2:] = 0

        # find the max (full pixel precision)
        maxIdx = np.argmax(mySpec, axis=1)

        # define the mask with maxima
        maskMax = np.zeros_like(self.image).astype(int)
        myPositionInt = self.gridLine.getPositionInt()
        #bwidth = self.spBlock.shape[2]//2
        maskMax[myPositionInt[:,0],myPositionInt[:,1] -self.bwidth + maxIdx]= np.arange(myPositionInt.shape[0])+1

        # make larger mask of the  max and make label of it
        #footprintDisk = morphology.disk(self.spBlock.shape[1]*2)
        footprintDisk = morphology.disk(self.bheight*3)

        label = morphology.dilation(maskMax,footprintDisk)

        # find exact position of first max (x,y)
        # calculate properties of objects
        props = measure.regionprops_table(label, self.image,
                properties=['label', 'centroid_weighted'])
        self.peakLeftPosition = np.vstack((props['centroid_weighted-0'],props['centroid_weighted-1'])).T

    def _getRightMax(self):
        ''' get position of the strongest peak on the left '''

        # right spectral lines averaged over y
        mySpec = np.mean(self.spBlock,axis=1)
        mySpec[:,0:mySpec.shape[1]//2] = 0

        # find the max (full pixel precision)
        maxIdx = np.argmax(mySpec, axis=1)

        # define the mask with maxima
        maskMax = np.zeros_like(self.image).astype(int)
        myPositionInt = self.gridLine.getPositionInt()
        #bwidth = self.spBlock.shape[2]//2
        maskMax[myPositionInt[:,0],myPositionInt[:,1] -self.bwidth + maxIdx]= np.arange(myPositionInt.shape[0])+1

        # make larger mask of the  max
        #footprintDisk = morphology.disk(self.spBlock.shape[1]*2)
        footprintDisk = morphology.disk(self.bheight*3)
        label = morphology.dilation(maskMax,footprintDisk)

        # calculate properties of objects
        props = measure.regionprops_table(label, self.image,
                properties=['label', 'centroid_weighted'])
        self.peakRightPosition = np.vstack((props['centroid_weighted-0'],props['centroid_weighted-1'])).T

    def prepareGrids(self):
        ''' wrapper function. carry out individual steps to get all info about grid
        in order to get the hyperspectral image
        '''
        self._identifySpectralLine()
        self._getLeftMax()
        self._getRightMax()

        # reassign more precisely the position of the spectral lines 
        # change the position to the left peak position and add offset,
        # so that it is in approximate middle 
        dxPosition = np.mean((self.gridLine.position - self.peakLeftPosition),axis=0)[1]

        self.gridLine.position = 1*self.peakLeftPosition
        self.gridLine.position[:,1] += dxPosition

        # assign the k-vectors pixels
        bheight =self.spBlock.shape[1]//2
        bwidth =self.spBlock.shape[2]//2
        dkPosition = np.mean(self.peakRightPosition[:,1] - self.peakLeftPosition[:,1])
        self.pixelPositionK = np.array([
            bwidth - dxPosition, bwidth - dxPosition + dkPosition]
        )

    def getSpectraBlock(self, image):
        ''' get the spectral blocks out of the image
        wrapper function for the getSpectralBlock
        THIS spectralBlock are fine-tuned. The intern self.spBlock are
        just for spectral peak identification 
        '''
        #bheight =self.spBlock.shape[1]//2
        #bwidth = self.spBlock.shape[2]//2

        return self.gridLine.getSpectraBlock(image, bheight=self.bheight, bwidth=self.bwidth)

    def getWYXImage(self,mySpec):
        ''' get the spectral image
        wrapper for a method from the GridSuperPixel class
        '''
        return self.gridLine.getWYXImage(mySpec)

    def getAlignedImage(self,mySpec):
        ''' get the aligned spectral blocks as an image
        wrapper for a method from the GridSuperPixel class
        '''
        return self.gridLine.getAlignedImage(mySpec)

    def getK(self):
        ''' get the wavelength vector defined on the range of spectral block
        use linear fit on 2 calibration peaks '''

        kFit = np.poly1d(np.polyfit(self.pixelPositionK,self.DEFAULT['kPosition'], 1))
        #return kFit(np.arange(self.spBlock.shape[2]))
        return kFit(np.arange(2*self.bwidth +1))


    def getWavelength(self):
        ''' only for compatibility issue
        it will be still giving the k-numbers 
        '''
        return self.getK()


    def getSpectralImage(self,rawImage,aberrationCorrection=False,
                         whitePixelCorrection=True,
                         flatFieldCorrection=False):
        ''' get the spectral image from raw image
        whitePixelCorrection ... remove the whitePixels from rawImage
        flatFieldCorrection ... adjust the intensity of the spectral line for each hspixel
                            ... not implemented now!!!
                            ... TODO: check if it improve the image
        
        it is just wrapper function
        '''

        try:
            # remove white pixels from  image
            if  whitePixelCorrection and self.whitePixel is not None:
                medianIm = median(rawImage, disk(1))
                rawImage[self.whitePixel] = medianIm[self.whitePixel]
            # remove cosmic ray/oversaturated images
            if whitePixelCorrection:
                cosmicRayMask = rawImage> self.DEFAULT['cosmicRayThreshold']
                rawImage[cosmicRayMask] = medianIm[cosmicRayMask]
        except:
            pass

        if aberrationCorrection:
            warpedImage = self.getWarpedImage(rawImage)
        else:
            warpedImage = rawImage

        mySpec = self.getSpectraBlock(warpedImage)

        # correct for the hyperpixel intensity variation
        #if flatFieldCorrection and self.ffFactor is not None:
        #    mySpec = mySpec * self.ffFactor[:,None,None]

        return self.getWYXImage(mySpec)


    def setWarpMatrix(self,spectral=True, subpixel=True):
        ''' set the final warping matrix
        spectral == True ... correct for the bending and linear expansion of the spectral lines 
        subpixel == True ... shift the max peak spots on full pixels
        '''     

        # define the grid points and warp matrix
        xx, yy = np.meshgrid(np.arange(self.image.shape[1]),
                                        np.arange(self.image.shape[0]))
        self.warpMatrix = np.array([yy,xx])

        if spectral:
            if self.dSpectralWarpMatrix is None: self._setSpectralWarpMatrix()
            self.warpMatrix = self.warpMatrix + self.dSpectralWarpMatrix

        if subpixel:
            if self.dSubpixelShiftMatrix is None: self._setSubpixelShiftMatrix()
            self.warpMatrix = self.warpMatrix + self.dSubpixelShiftMatrix

    def _setSpectraFlatFielding(self):
        ''' correct the spectra intensity variation of the signal
        it assumes homogenous sample with homogenous illumination 
        TODO: the simple averaging do not work very well.
        do linear fit on the wavelength. (increasing with the wavelength)'''

        # get the spectral block 
        # make copy of the image
        rawImage = 1*self.image

        if  self.whitePixel is not None:
            rawImage[self.whitePixel]=0

        if self.warpMatrix is not None:
            warpedImage = self.getWarpedImage(rawImage)
        else:
            warpedImage = rawImage
        mySpec = self.getSpectraBlock(warpedImage)

        # the simplest constant (average) multiplication
        ffFactor = np.mean(mySpec,axis=(1,2))
        # ignore empty hspixels
        ffFactor[ffFactor==0] = 1
        self.ffFactor = np.mean(mySpec)/ffFactor
          


    def _setSubpixelShiftMatrix(self):
        ''' calculate warp matrix to shift left spots on integer grids
        TODO: test it, add spectral calibration as well '''

        points = self.peakLeftPosition
        pointsIdeal = np.round(points).astype(int)

        vector = pointsIdeal - points 

        self.dSubpixelShiftMatrix  = np.zeros((2,*self.image.shape)) # 0... yshift, 1 ... xshift

        bwidth = self.spBlock.shape[2]//2
        bheight = self.spBlock.shape[1]//2
        myPositionInt = self.gridLine.getPositionInt()

        for ii in range(2*bwidth+1):
            for jj in range(2*bheight+1):
                # y-shift   
                self.dSubpixelShiftMatrix[myPositionInt[self.gridLine.inside,0]*0,
                    myPositionInt[self.gridLine.inside,0]+jj-bheight,
                    myPositionInt[self.gridLine.inside,1]+ii-bwidth] = vector[self.gridLine.inside,0]
                # x-shift
                self.dSubpixelShiftMatrix[myPositionInt[self.gridLine.inside,0]*0 + 1 ,
                    myPositionInt[self.gridLine.inside,0]+jj-bheight,
                    myPositionInt[self.gridLine.inside,1]+ii-bwidth] = vector[self.gridLine.inside,1]

    def _setSpectralWarpMatrix(self):
        ''' calculate the chromatic distortion matrix from vectors shifts
        and vector for pixel to spectral calibration
        TODO: just copied code. adapt it !!!
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





    def getWarpedImage(self, image):
        ''' make the warping on the image'''
        warpedImage = warp(image, self.warpMatrix, mode='edge')        
        return warpedImage


    def process(self,spectral=True, subpixel=True):
        ''' carry out all steps to be able to get hyperspectral image '''

        self.prepareGrids()
        self.setWarpMatrix(spectral=spectral, subpixel=subpixel)
        self.setSpectraFlatFielding()


if __name__ == "__main__":

    from HSIplasmon.algorithm.io import RamanImage
    import napari
    from HSIplasmon.xywViewer import xywViewer
    from HSIplasmon.RamanViewer import RamanViewer

    if False:
        # load calibration image
        fFolder = r'g:\office\work\projects - free\23-07-07 Integral field uscope - Mariia\23-09-26 Mariia data\DATA\calibration2'
        myRI = RamanImage(fFolder).getImage()  

        myCal = CalibrateRamanImages(myRI)
        myCal.prepareGrids()

        myCal.setWarpMatrix()
    
        myCal.saveClass()
    else:
        myCal = CalibrateRamanImages(0)
        myCal = myCal.loadClass()
    
    myK = myCal.getK()

    #myWRI = myCal.getWarpedImage(myRI)

    # display warp matrix
    #viewer = napari.Viewer()
    #viewer.add_image(myCal.dSubpixelShiftMatrix)

    #spBlock = myCal.getSpectraBlock(myRI)
    #spIm = myCal.getWYXImage(spBlock)
    #alignedIm = myCal.getAlignedImage(spBlock)

    #spBlock2 = myCal.getSpectraBlock(myWRI)
    #spIm2 = myCal.getWYXImage(spBlock2)
    #alignedWIm = myCal.getAlignedImage(spBlock2)

    # display aligned images
    #viewer = napari.Viewer()
    #viewer.add_image(alignedIm)
    #viewer.add_image(alignedWIm)


    # display spectra cube
    #sViewer = xywViewer(spIm, myK)
    #sViewer2 = xywViewer(spIm2, myK)



    
    # prepare a sub selected  points
    pointsSelect = (myCal.gridLine.imIdx[:,0]%10 == 0 ) & (myCal.gridLine.imIdx[:,1]%10 == 0 )
    pointsL = myCal.peakLeftPosition[pointsSelect,:]
    pointsR = myCal.peakRightPosition[pointsSelect,:]


    features = {'pointIndex0': myCal.gridLine.imIdx[pointsSelect,0],
                'pointIndex1': myCal.gridLine.imIdx[pointsSelect,1]
                }
    text = {'string': '[{pointIndex0},{pointIndex1}]',
            'translation': np.array([-30, 0])
            }

    # display the images
    viewer = napari.Viewer()
    viewer.add_image(myCal.image)
    viewer.add_points(pointsL,features=features,text=text, size= 50, opacity=0.5)
    viewer.add_points(pointsR,features=features,text=text, size= 50, opacity=0.5)

    # display aligned image
    spBlock = myCal.getSpectraBlock(myCal.image)
    alignedIm = myCal.getAlignedImage(spBlock)

    viewer2 = napari.Viewer()
    viewer2.add_image(alignedIm)

    # display spectra cube
    spIm = myCal.getWYXImage(spBlock)
    myK = myCal.getK()
    #sViewer = xywViewer(spIm, myK)
    sViewer = RamanViewer(spIm, myK)

   
    #%% load image and display spectra cube
    fFolder = r'G:\office\work\projects - free\23-07-07 Integral field uscope - Mariia\23-09-26 Mariia data\DATA\two_beads2'
    #fFolder = r'G:\office\work\projects - free\23-07-07 Integral field uscope - Mariia\23-09-26 Mariia data\DATA\two_beads'
    #fFolder = r'G:\office\work\projects - free\23-07-07 Integral field uscope - Mariia\23-09-26 Mariia data\DATA\flurescence_beads'


    myRI2 = RamanImage(fFolder).getImage()  

    spBlock2 = myCal.getSpectraBlock(myRI2)
    spIm2 = myCal.getWYXImage(spBlock2)
    sViewer2 = xywViewer(spIm2, myK)

    #%% subtract background
    from HSIplasmon.algorithm.spectraprocess import baseline_SNIP

    # cut out the edges
    myK = myK[30:-30]
    spIm3 = spIm2[30:-30,:,:]
    mySp =spIm3.shape

    spIm3 = np.reshape(spIm3,(mySp[0],-1))
    spIm3Bcg = np.zeros_like(spIm3)
    for ii in np.arange(spIm3.shape[1]):
        spIm3Bcg[:,ii] = baseline_SNIP(spIm3[:,ii],100)
        if ii%100==0:
            print(f'baseline correction {ii} of {spIm3.shape[1]}')

    spIm3Bcg = np.nan_to_num(np.reshape(spIm3Bcg,mySp))
    spIm3 = np.reshape(spIm3,mySp)
    sViewer3 = xywViewer(spIm3Bcg, myK)
    sViewer4 = xywViewer(spIm3-spIm3Bcg, myK)


    #%% spectra clustering
    # followed:
    # https://www.kaggle.com/code/phamvanvung/nir-spectra-classification-using-pca
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

    from scipy.signal import savgol_filter
    from sklearn.decomposition import PCA as sk_pca
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    import matplotlib.pyplot as plt
    from itertools import cycle, islice
    from sklearn import cluster
    from skimage.filters import gaussian as guassian


    # PCA 
    newIm = guassian(spIm3-spIm3Bcg,sigma=1, channel_axis=0)
    
    feat = np.swapaxes(np.reshape(newIm,(mySp[0],-1)),0,1)
    #dfeat = savgol_filter(feat, 25, polyorder = 5, deriv = 1)
    nfeat = StandardScaler().fit_transform(feat)
    skpca = sk_pca(n_components=3) 
    X = skpca.fit_transform(nfeat)

    #clustering
    spectral = cluster.SpectralClustering()
        #n_clusters=5,
        #eigen_solver="arpack",
        #affinity="nearest_neighbors"
        #


    spectral.fit(X)

    y_pred = spectral.labels_.astype(int)

    # plot result
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    mybeads = np.reshape(y_pred,mySp[1:])
    intensityIm = np.sum(newIm,axis=(0))
    sViewer5 = xywViewer(newIm, myK)
    # remove the spectra averaging in the display
    sViewer5.pxAve = 0

    # add each cluster 
    colorindex = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
    def myColorMap(rgb):
        colors = np.linspace(
        start=[0,0,0, 1],
        stop=[rgb[0],rgb[1],rgb[2], 1],
        num=20,
        endpoint=True
        )
        colors[0] = np.array([rgb[0], rgb[1],rgb[2], 0])
        transparent_colormap = {
        'colors': colors,
        'name': 'red_and_green',
        'interpolation': 'linear'
        }
        return transparent_colormap


    for ii in range(np.max(y_pred)):
        partIm= intensityIm*(mybeads==ii)
        sViewer5.viewer.add_image(partIm, 
        opacity=1, 
        name=f'part{ii}', 
        colormap= myColorMap(colorindex[ii]))

    napari.run()





# %%


# %%
