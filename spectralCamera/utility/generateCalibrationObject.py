''' script to generate calibration object from three images '''

#%% import and parameter definition

import napari
import spectralCamera
import numpy as np
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer


dfolder = spectralCamera.dataFolder
whiteFileName = 'filter_wo_0.npy'
imageNameStack = None # default
wavelengthStack = None # default
spectralRange = [470, 770]

#%% data loading
print('loading data')
# load reference image
whiteImage = np.load(dfolder + '\\' + whiteFileName)

# initiate the calibration class
myCal = CalibrateFrom3Images(imageNameStack=imageNameStack,
                             wavelengthStack=wavelengthStack)

# load images
myCal.setImageStack()

#%% process calibration images
print('processing the reference images - getting the grid')
myCal.prepareGrid(spectralRange)

#%% visual check that grids are on proper position
print('visual check of grid and block position')

viewer = napari.Viewer()

# white image
viewer.add_image(whiteImage, name='white' )

# show spectral block
blockImage = myCal.getSpectralBlockImage()
blockImageLayer = viewer.add_image(blockImage*1, name='spectral block',opacity=0.2)

answer = ""
while answer != "y":
    answer = input(f" is spectral range {spectralRange} good [Y/N]  ").lower()
    if answer != "y":
        range1 = int(input(f"{spectralRange[0]} --> : "))
        range2 = int(input(f"{spectralRange[1]} --> : "))
        myCal.setGridLine([range1,range2])
        blockImage = myCal.getSpectralBlockImage()
        blockImageLayer.date = blockImage*1


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


#%% calculate the calibration warping matrices
print('calculating the calibration -warp - matrix')

myCal.setWarpMatrix(spectral=True, subpixel=True)

#%% visual check of the warping matrices
print('visual check of the calibration routine')

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
#napari.run()


#%% show hyper spectral image

sViewer = XYWViewer(np.stack((spImage2,spImage2Cor)),myCal.wavelength)
sViewer.run()


#%% save class
if input("Save the calibration [Y/N]? ").lower()=='y':
    print(f'calibration will be saved in folder: {dfolder}')
    myCal.saveClass(classFolder=dfolder)
