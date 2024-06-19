''' script to generate calibration object from three images '''

#%% import and parameter definition

import napari
import spectralCamera
import numpy as np
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
from skimage.transform import warp
from scipy.interpolate import griddata


dfolder = spectralCamera.dataFolder
whiteFileName = 'filter_wo_0.npy'
imageNameStack = None
wavelengthStack = None


#%% data loading
print('loading data')
# load reference image
whiteImage = np.load(dfolder + '\\' + whiteFileName)

myCal = CalibrateFrom3Images()

myCal = myCal.loadClass(dfolder + '//' + 'CalibrateFrom3Images.obj')


#%% recalculate the calibration matrices

myCal._setSubpixelShiftMatrix()
myCal._setSpectralWarpMatrix()


# %% show white image  in spectral viewer

spImage = myCal.getSpectralImage(whiteImage)
myCal.setWarpMatrix(spectral=True, subpixel=True)
spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage,spImageCor)),myCal.wavelength)
sViewer.run()

# %% show white image  in 2D

spImage = myCal.getSpectralImage(whiteImage)
myCal.setWarpMatrix(spectral=False, subpixel=True)
spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage,spImageCor)),myCal.wavelength)
sViewer.run()



# %% show single wavelength image  in spectral viewer

peakImage = myCal.imageStack[0] + myCal.imageStack[1] + myCal.imageStack[2]

spImage2 = myCal.getSpectralImage(peakImage)
myCal.setWarpMatrix(spectral=True, subpixel=True)
spImageCor2 = myCal.getSpectralImage(peakImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage2,spImageCor2)),myCal.wavelength)
sViewer.run()





# %% define better subpixel shift matrix

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
vy = griddata(points, -vector[:,0], (yy, xx), method='cubic', fill_value= 0)
vx = griddata(points, vector[:,1], (yy, xx), method='cubic', fill_value= 0)


myCal.dSubpixelShiftMatrix = np.array([vy,vx])

#%% show the warp shiftMatrix

mask = np.zeros_like(myCal.imageStack[0], dtype='int')
for ii,imMo in enumerate(myCal.imMoStack):
    selectPoint = (imMo.imIdx[:,0]%2 == 0 ) & (imMo.imIdx[:,1]%2 == 0 )
    mask[imMo.position[selectPoint,0].astype(int),imMo.position[selectPoint,1].astype(int)] = ii+1

viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift',opacity=0.2)
blockImage = myCal.getSpectralBlockImage()
viewer2.add_image(blockImage*1, name='spectral block',opacity=0.2)
viewer2.add_labels(mask, name='peaks')



# %% compare the original and warped images

spImage = myCal.getSpectralImage(whiteImage, aberrationCorrection=False)
myCal.setWarpMatrix(spectral=False, subpixel=True)
spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage,spImageCor)),myCal.wavelength)
sViewer.run()

# %% generate better spectral shift matrix

def fun(w):
    slope = ((self.positionMatrix[1,:,self.boolMatrix] - self.positionMatrix[2,:,self.boolMatrix]) /
             (self.wavelengthStack[1] - self.wavelengthStack[2]))
    return  slope*(w-self.wavelengthStack[0]) + self.gridLine.getPositionInt() - self.positionMatrix[0,:,self.boolMatrix]


vx = np.zeros_like(self.imageStack[0])
vy = np.zeros_like(self.imageStack[0])

myPos = self.gridLine.getPositionInt()

for ii in range(2*self.bheight+1):
    for jj in range(2*self.bwidth+1):
        vx[(myPos[:,0]+ii-self.bheight).astype(int),
           (myPos[:,1]+jj-self.bwidth).astype(int)] = fun(self.wavelength[jj])[:,1] -jj + self.bwidth
        vy[(myPos[:,0]+ii-self.bheight).astype(int),
           (myPos[:,1]+jj-self.bwidth).astype(int)] = fun(self.wavelength[jj])[:,0]

self.dSpectralWarpMatrix2 = np.array([vy,vx])


# %%
viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSpectralWarpMatrix, name='SubpixelShift',opacity=0.2)
viewer2.add_image(myCal.dSpectralWarpMatrix2, name='SubpixelShift2',opacity=0.2)

blockImage = myCal.getSpectralBlockImage()
viewer2.add_image(blockImage*1, name='spectral block',opacity=0.2)








# %%
viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift',opacity=0.2)
viewer2.add_image(myCal.dSubpixelShiftMatrix2, name='SubpixelShift2',opacity=0.2)

blockImage = myCal.getSpectralBlockImage()
viewer2.add_image(blockImage*1, name='spectral block',opacity=0.2)
viewer2.add_labels(mask, name='peaks')
# %%

spImage = myCal.getSpectralImage(whiteImage, aberrationCorrection=False)

self.dSubpixelShiftMatrix = self.dSubpixelShiftMatrix2
myCal.setWarpMatrix(spectral=False, subpixel=True)

spImageCor = myCal.getSpectralImage(whiteImage,aberrationCorrection=True)

sViewer = XYWViewer(np.stack((spImage,spImageCor)),myCal.wavelength)
sViewer.run()
# %%
