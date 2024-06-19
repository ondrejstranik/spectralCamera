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
myCal.processImageStack()
myCal._setGlobalGridZero()
myCal._setPositionMatrix()
myCal.setGridLine()

#%% visual check that grids are on proper 

# white image
viewer = napari.Viewer()
viewer.add_image(whiteImage, name='white')

# calibration images
iS = myCal.imageStack[0] + myCal.imageStack[1] + myCal.imageStack[2]
viewer.add_image(iS,name = 'calibration images', opacity=0.5)

# located spectral peaks in the calibration images
mask = np.zeros_like(myCal.imageStack[0], dtype='int')
for ii,imMo in enumerate(myCal.imMoStack):
    selectPoint = (imMo.imIdx[:,0]%2 == 0 ) & (imMo.imIdx[:,1]%2 == 0 )
    mask[imMo.position[selectPoint,0].astype(int),imMo.position[selectPoint,1].astype(int)] = ii+1
viewer.add_labels(mask, name='peaks')

# show zero points
point00 = []
for ii,imMo in enumerate(myCal.imMoStack):
    point00.append(imMo.xy00)
viewer.add_points(np.array(point00), size= 5, opacity=0.2, name= 'zero position')

# show spectral block
blockImage = myCal.getSpectralBlockImage()
viewer.add_image(blockImage*1, name='spectral block',opacity=0.2)

#%% calculate the calibration warping matrices
print('calculating the calibration -warp - matrix')
myCal._setSubpixelShiftMatrix()
myCal._setSpectralWarpMatrix()
myCal.setWarpMatrix(spectral=True, subpixel=True)

#%% visual check of the warping matrices
viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift',opacity=0.2)
viewer2.add_image(myCal.dSpectralWarpMatrix, name='SpectralWarp',opacity=0.2)
viewer2.add_labels(mask, name='peaks')

#%% visual check of the warped image

# define labels of the calibration peak and ideal peak
label = np.zeros_like(myCal.imageStack[0],dtype=int)
for ii,imMo in enumerate(myCal.imMoStack):
    px = np.argmin(np.abs(myCal.wavelength - myCal.wavelengthStack[ii]))
    avePoint = myCal.gridLine.getPositionInt()
    label[imMo.position[:,0].astype(int),imMo.position[:,1].astype(int)] = ii+1
    label[avePoint[:,0].astype(int), avePoint[:,1].astype(int) -myCal.bwidth + px] = len(myCal.imMoStack) + ii +1

# show the data
viewer3 = napari.Viewer()
viewer3.add_image(iS, name = 'calibration image', opacity=0.5)
viewer3.add_image(myCal.getWarpedImage(iS), name = 'warped calibration image', opacity=0.5)
viewer3.add_image(whiteImage, name='white')
viewer3.add_image(myCal.getWarpedImage(whiteImage), name='warped white')
viewer3.add_labels(label, name='peak real and ideal')


#%%
napari.run()


#%% save class

myCal.saveClass(classFolder=dfolder)


# %% show data in spectral viewer
# only if class is saved
myCal = myCal.loadClass(dfolder + '/' +'CalibrateFrom3Images.obj')

spImage = myCal.getSpectralImage(whiteImage)
myCal.setWarpMatrix(spectral=False, subpixel=False)
spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage,spImageCor)),myCal.wavelength)
sViewer.run()


# %%

viewer2 = napari.Viewer()
viewer2.add_image(myCal.warpMatrix+100, name='warpmatrix',opacity=0.2)
napari.run()
# %%

from skimage.transform import warp

xx, yy = np.meshgrid(np.arange(whiteImage.shape[1]),
                     np.arange(whiteImage.shape[0]))
warpMatrix = np.array([yy,xx])

wM = warpMatrix + myCal.dSubpixelShiftMatrix

warpedImage = warp(whiteImage, wM,mode='edge', preserve_range=True, order=3)
spWarpedImage = myCal.getSpectralImage(warpedImage)
sViewer = XYWViewer(np.stack((spImage,spWarpedImage)),myCal.wavelength)
sViewer.run()


# %%

xx, yy = np.meshgrid(np.arange(whiteImage.shape[1]),
                     np.arange(whiteImage.shape[0]))
warpMatrix = np.array([yy,xx])

wM = warpMatrix + myCal.dSpectralWarpMatrix

warpedImage = warp(whiteImage, wM,mode='edge', preserve_range=True, order=2)

viewer4 = napari.Viewer()
viewer4.add_image(myCal.dSpectralWarpMatrix, name='warpmatrix',opacity=0.2)
viewer4.add_image(whiteImage, name='whiteImage',opacity=0.2)
viewer4.add_image(warpedImage, name='whiteImage',opacity=0.2)

napari.run()





# %% check the subpixel shift matrix

from scipy.interpolate import griddata

# calculate relative shift
vectorMatrixSubpixelShift = myCal.positionMatrix[0,...] - np.round(myCal.positionMatrix[0,...])

points0 = myCal.positionMatrix[0,:,myCal.boolMatrix]
points1 = myCal.positionMatrix[1,:,myCal.boolMatrix]
points2 = myCal.positionMatrix[2,:,myCal.boolMatrix]
vector_ = vectorMatrixSubpixelShift[:,myCal.boolMatrix].T

# put them all together
sv = np.array([myCal.bheight,0])
points_M = np.vstack((points0,points1,points2))
points_T = np.vstack((points0,points1,points2)) + sv
points_D = np.vstack((points0,points1,points2)) - sv
points = np.vstack((points_M,points_T,points_D))
vector = np.vstack((vector_,vector_,vector_,
                    vector_,vector_,vector_,
                    vector_,vector_,vector_))

# define the grid points
xx, yy = np.meshgrid(np.arange(myCal.imageStack[0].shape[1]),
                        np.arange(myCal.imageStack[0].shape[0]))

# interpolate the shift on all pixel in the image
vy = griddata(points, vector[:,0], (yy, xx), method='cubic', fill_value= 0)
vx = griddata(points, vector[:,1], (yy, xx), method='cubic', fill_value= 0)


myCal.dSubpixelShiftMatrix = np.array([vy,vx])

#%%

mask = np.zeros_like(myCal.imageStack[0], dtype='int')
for ii,imMo in enumerate(myCal.imMoStack):
    selectPoint = (imMo.imIdx[:,0]%2 == 0 ) & (imMo.imIdx[:,1]%2 == 0 )
    mask[imMo.position[selectPoint,0].astype(int),imMo.position[selectPoint,1].astype(int)] = ii+1

viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift',opacity=0.2)
blockImage = myCal.getSpectralBlockImage()
viewer2.add_image(blockImage*1, name='spectral block',opacity=0.2)
viewer2.add_labels(mask, name='peaks')



# %%

spImage = myCal.getSpectralImage(whiteImage, aberrationCorrection=False)
myCal.setWarpMatrix(spectral=False, subpixel=False)
spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage,spImageCor)),myCal.wavelength)
sViewer.run()
# %%
