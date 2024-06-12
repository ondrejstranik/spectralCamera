'''
class to calibrate Raman image from one image of homogenous image of polymer polystyrene
'''
#%%

import numpy as np

from skimage import measure, morphology
from skimage.filters import threshold_otsu, median, threshold_local
from skimage.morphology import disk
from skimage.transform import warp

from spectralCamera.algorithm.baseCalibrate import BaseCalibrate
from spectralCamera.algorithm.gridSuperPixel import GridSuperPixel


class CalibrateIFFromFilterImage(BaseCalibrate):
    ''' main class to calibrate IF camera from filter images
        It needs three images.
         One homogenous image with broad range spectrum.
         One homogenous image with narrow band spectrum at lower wavelength
         One homogenous image wi:ht narrow band spectra at hight wavelength
           '''

    DEFAULT = {'calibrationWavelength': [525, 700],
                'cosmicRayThreshold': 1e6,
                }

    def __init__(self,image=None,darkImage=None):
        ''' class definition '''

        self.image = image
        self.darkImage = darkImage

        self.filterImage = []
        self.filterWavelength = []
        
        self.whitePixel = None
        self.gridLine = None
        self.spBlock = None

        self.mask = None
        self.peakLeftPosition = None
        self.peakRightPosition = None

        self.warpMatrix = None
        self.dSubpixelShiftMatrix = None
        self.dSpectralWarpMatrix = None

    def setImage(self,image, filter=None):
        ''' set the images for the necessary for the calibration'''

        if filter is None:
            self.image = image
            return
        if filter == 'dark':
            self.darkImage = image
            return
        
        self.filterImage.append(image)
        self.filterWavelength.append(filter)

    def _identifySpectralLine(self):
        ''' identify spectral line from the image'''

        myim = self.image

        # smoothing
        medianIm = median(myim, disk(1))

        # remove white pixels if dark Image provided (white pixel obtained from dark image)
        if self.darkImage is not None:
            wpThreshold = np.mean(self.darkImage) + 4*np.std(self.darkImage)
            #wpThreshold = np.mean(self.darkImage) + 10*np.std(self.darkImage)
            self.whitePixel = (self.darkImage>wpThreshold)
            myim[self.whitePixel] = medianIm[self.whitePixel]

        # remove cosmic ray/oversaturated images
        cosmicRayMask = myim> self.DEFAULT['cosmicRayThreshold']
        myim[cosmicRayMask] = medianIm[cosmicRayMask]

        '''
        # identify the spectral line
        threshold = threshold_otsu(myim)
        mask = myim > threshold

        # connect broken spectral lines
        footprint = np.ones(10)[None,:]
        mask2 = morphology.binary_closing(mask,footprint=footprint)

        # remove small spots
        mask3 = morphology.remove_small_objects(mask2, 50)

        '''

        #identify the spectral lines
        local_thresh = threshold_local(myim, 41)
        mask = myim > local_thresh

        # connect broken spectral lines
        footprint = np.ones(10)[None,:]
        mask2 = morphology.binary_closing(mask,footprint=footprint)

        # calculate properties of objects
        labels = measure.label(mask2)
        component_sizes = np.bincount(labels.ravel())
        myArea = np.median(component_sizes)

        # remove too small and too large objects
        mySelection = np.logical_or((component_sizes < myArea/1.5),
                                    (component_sizes > myArea*1.52))
        myMask = mySelection[labels]
        mask2[myMask] = 0
        mask3 = mask2
        self.mask = mask3

        # calculate properties of objects
        labels = measure.label(mask3)
        props = measure.regionprops_table(labels, myim,
                properties=['label', 'centroid','orientation', 'axis_major_length','axis_minor_length', 'intensity_max'])

        myPosition = np.vstack((props['centroid-0'],props['centroid-1'])).T

        bwidth = np.round(np.mean(props['axis_major_length']//2)).astype(int)
        bheight = np.round(np.mean(props['axis_minor_length']//2)).astype(int)

        self.gridLine = GridSuperPixel()
        self.gridLine.setGridPosition(myPosition)
        self.gridLine.getGridInfo()
        self.gridLine.getPixelIndex()

        #remove the lines, which are too close to the edges        
        secureXOffset = 10
        self.gridLine.getPositionOutsideImage(myim,bheight=bheight,bwidth=bwidth+secureXOffset)
        self.spBlock = self.gridLine.getSpectraBlock(myim,bheight=bheight,bwidth=bwidth)

    def _getFilterPeak(self,filterImage):
        ''' get position of the strongest peak of the filter image
         the spectral lines has to be defined beforehand  '''

        bheigth =self.spBlock.shape[1]//2
        bwidth = self.spBlock.shape[2]//2
        _spBlock = self.gridLine.getSpectraBlock(filterImage,bheight=bheigth,bwidth=bwidth)

        # left spectral lines averaged over y
        mySpec = np.mean(_spBlock,axis=1)

        # find the max (full pixel precision)
        maxIdx = np.argmax(mySpec, axis=1)

        # define the mask with maxima
        # at first single pixels and give it a label
        maskMax = np.zeros_like(filterImage,dtype=int)
        myPositionInt = self.gridLine.getPositionInt()
        maskMax[myPositionInt[:,0],myPositionInt[:,1] -bwidth + maxIdx]= np.arange(myPositionInt.shape[0])+1

        # make larger mask  of the max (twice the width of the block) 
        #footprintDisk = morphology.disk(self.spBlock.shape[1]*2)
        footprintDisk = morphology.disk(self.spBlock.shape[1])

        label = morphology.dilation(maskMax,footprintDisk)

        # find exact position of first max (x,y)
        # calculate properties of objects
        props = measure.regionprops_table(label, self.image,
                properties=['label', 'centroid_weighted'])
        
        return np.vstack((props['centroid_weighted-0'],props['centroid_weighted-1'])).T
        
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
        bwidth = self.spBlock.shape[2]//2
        maskMax[myPositionInt[:,0],myPositionInt[:,1] -bwidth + maxIdx]= np.arange(myPositionInt.shape[0])+1

        # make larger mask of the  max
        footprintDisk = morphology.disk(self.spBlock.shape[1]*2)
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
        bheight =self.spBlock.shape[1]//2
        bwidth = self.spBlock.shape[2]//2

        return self.gridLine.getSpectraBlock(image, bheight=bheight, bwidth=bwidth)

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
        return kFit(np.arange(self.spBlock.shape[2]))

    def getWavelength(self):
        ''' only for compatibility issue
        it will be still giving the k-numbers 
        # TODO'''

        return self.getK()


    def getSpectralImage(self,rawImage,whitePixelCorrection=True,flatFieldCorrection=False):
        ''' get the spectral image from raw image
        whitePixelCorrection ... remove the whitePixels from rawImage
        flatFieldCorrection ... adjust the intensity of the spectral line for each hspixel
                            ... so far it does not work very good!
        
        it is just wrapper function
        '''

        try:
            if  whitePixelCorrection and self.whitePixel is not None:
                rawImage[self.whitePixel]=0
        except:
            pass

        if self.aberrationCorrection:
            warpedImage = self.getWarpedImage(rawImage)
        else:
            warpedImage = rawImage

        mySpec = self.getSpectraBlock(warpedImage)

        # correct for the hyperpixel intensity variation
        if flatFieldCorrection and self.ffFactor is not None:
            mySpec = mySpec * self.ffFactor[:,None,None]

        return self.getWYXImage(mySpec)


    def setWarpMatrix(self,spectral=True, subpixel=True):
        ''' set the final warping matrix
        spectral == True ... correct for the bending and linear expansion of the spectral lines 
        subpixel == True ... shift the max peak spots on full pixels
        '''

        if spectral or subpixel:
            self.aberrationCorrection = True
        else:
            self.aberrationCorrection = False        

        # define the grid points and warp matrix
        self.xx, self.yy = np.meshgrid(np.arange(self.image.shape[1]),
                                        np.arange(self.image.shape[0]))
        self.warpMatrix = np.array([self.yy,self.xx])

        # TODO: set SpectralWarpMatrix
        #if spectral:
        if False:
            self._setSpectralWarpMatrix()
            self.warpMatrix = self.warpMatrix + self.dSpectralWarpMatrix

        if subpixel:
            self._setSubpixelShiftMatrix()
            self.warpMatrix = self.warpMatrix + self.dSubpixelShiftMatrix

    def setSpectraFlatFielding(self):
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
