''' script to generate calibration object from three images '''

#%% import and parameter definition

import napari
import spectralCamera
import numpy as np
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer


dfolder = spectralCamera.dataFolder
whiteFileName = 'filter_wo_0.npy'
imageNameStack = None
wavelengthStack = None


#%% data loading
print('loading data')
# load reference image
whiteImage = np.load(dfolder + '\\' + whiteFileName)

myCal = CalibrateFrom3Images()

if imageNameStack is not None: myCal.imageNameStack = imageNameStack
if wavelengthStack is not None: myCal.wavelengthStack = wavelengthStack


# initiate the calibration class
myCal = CalibrateFrom3Images()
# load images
myCal.setImageStack()

#%% process calibration images
print('processing the reference images - getting the grid')
myCal.prepareGrid([450,720])

#%% visual check that grids are on proper position

# white image
viewer = napari.Viewer()
viewer.add_image(whiteImage, name='white' )

# calibration images
iS = myCal.imageStack[0] + myCal.imageStack[1] + myCal.imageStack[2]
viewer.add_image(iS,name = 'calibration images', opacity=0.5, colormap='hsv')

# located spectral peaks in the calibration images
peakLabel = np.zeros_like(myCal.imageStack[0], dtype='int')
for ii,imMo in enumerate(myCal.imMoStack):
    selectPoint = (imMo.imIdx[:,0]%2 == 0 ) & (imMo.imIdx[:,1]%2 == 0 )
    peakLabel[imMo.position[selectPoint,0].astype(int),imMo.position[selectPoint,1].astype(int)] = ii+1
viewer.add_labels(peakLabel, name='peaks', opacity=1)

# show zero points
point00 = []
for ii,imMo in enumerate(myCal.imMoStack):
    point00.append(imMo.xy00)
viewer.add_points(np.array(point00), size= 50, opacity=0.2, name= 'zero position')

# show spectral block
blockImage = myCal.getSpectralBlockImage()
viewer.add_image(blockImage*1, name='spectral block',opacity=0.2)

#%% calculate the calibration warping matrices
print('calculating the calibration -warp - matrix')

myCal.setWarpMatrix(spectral=True, subpixel=True)

#%% visual check of the warping matrices

# located spectral peaks in the calibration images
label = np.zeros_like(myCal.imageStack[0], dtype='int')
avePoint = myCal.gridLine.getPositionInt()
for ii,imMo in enumerate(myCal.imMoStack):
    px = np.argmin(np.abs(myCal.wavelength - myCal.wavelengthStack[ii]))
    label[imMo.position[:,0].astype(int),imMo.position[:,1].astype(int)] = 2
    label[avePoint[:,0].astype(int), avePoint[:,1].astype(int) -myCal.bwidth + px] = 3

viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift',colormap='turbo')
viewer2.add_image(myCal.dSpectralWarpMatrix, name='SpectralWarp',colormap='turbo')
viewer2.add_labels(label, name='peaks', opacity=1)

#%% visual check of the warped image

viewer3 = napari.Viewer()
viewer3.add_image(iS, name = 'calibration image', opacity=1,colormap='turbo')
viewer3.add_image(myCal.getWarpedImage(iS), name = 'warped calibration image', opacity=1,colormap='turbo')
viewer3.add_image(whiteImage, name='white')
viewer3.add_image(myCal.getWarpedImage(whiteImage), name='warped white')
viewer3.add_labels(label, name='peak real and ideal')


#%% show the spectral images
spImage = myCal.getSpectralImage(whiteImage,aberrationCorrection=False)
spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

spImage2 = myCal.getSpectralImage(iS, aberrationCorrection=False)
spImage2Cor = myCal.getSpectralImage(iS,aberrationCorrection=True)

viewer3 = napari.Viewer()
viewer3.add_image(spImage, name = 'white not cor', opacity=1,colormap='turbo')
viewer3.add_image(spImageCor, name = 'white cor', opacity=1,colormap='turbo')
viewer3.add_image(spImage2, name = 'peak not cor', opacity=1,colormap='turbo')
viewer3.add_image(spImage2Cor, name = 'peak cor', opacity=1,colormap='turbo')

#%%
napari.run()


#%% show hyper spectral image

sViewer = XYWViewer(np.stack((spImage2,spImage2Cor)),myCal.wavelength)
sViewer.run()


#%% save class

myCal.saveClass(classFolder=dfolder)
