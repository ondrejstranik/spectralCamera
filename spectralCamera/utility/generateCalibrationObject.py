''' script to generate calibration object from three images '''

#%% import and parameter definition

import napari
import spectralCamera
import numpy as np
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer

import matplotlib.pyplot as plt
# set backend in the case of ipython
try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic("matplotlib","")
except:
    pass


spectralCamera.dataFolder = r'C:\Users\ostranik\Documents\GitHub\spectralCamera\spectralCamera\DATA\24-06-26-calibration'

whiteFileName = 'white_0.npy'
imageNameStack = ['filter_602_0','filter_505_0','filter_705_0']
wavelengthStack = [602,505,705]
spectralRange = [490, 750]

#%% data loading
print('loading data')
# load reference image
whiteImage = np.load(spectralCamera.dataFolder + '\\' + whiteFileName)

# initiate the calibration class
myCal = CalibrateFrom3Images(imageNameStack=imageNameStack,
                             wavelengthStack=wavelengthStack)

# load images
myCal.setImageStack()

#%% process calibration images
if False:
    print('processing the reference images - getting the grid')
    myCal.prepareGrid(spectralRange)
    myCal._saveGridStack()
else:
    print('loading the grid')
    myCal._loadGridStack()
    myCal._setPositionMatrix()
    myCal.setGridLine(spectralRange=spectralRange)
    myCal.bheight = 3


#%% visual check that grids are on proper position

fig, ax = plt.subplots()
ax.plot(myCal.wavelength)

ax.set(xlabel='pixels', ylabel='wavelength [nm]',
       title='fit pixel to wavelength')
ax.axhline(y=myCal.wavelengthStack[0])
ax.axhline(y=myCal.wavelengthStack[1])
ax.axhline(y=myCal.wavelengthStack[2])
ax.axvline(x=myCal.pixelPositionWavelength[0]+myCal.bwidth-myCal.xShift)
ax.axvline(x=myCal.pixelPositionWavelength[1]+myCal.bwidth-myCal.xShift)
ax.axvline(x=myCal.pixelPositionWavelength[2]+myCal.bwidth-myCal.xShift)

plt.show()

#%%


print('visual check of grid and block position')

viewer = napari.Viewer()

# white image
viewer.add_image(whiteImage, name='white' )

# show spectral block
blockImage = myCal.getSpectralBlockImage()*1
blockImageLayer = viewer.add_image(blockImage, name='spectral block',opacity=0.2)




answer = ""
while answer != "y":
    answer = input(f" is spectral range {spectralRange} good [Y/N]  ").lower()
    if answer != "y":
        range1 = int(input(f"{spectralRange[0]} --> : "))
        range2 = int(input(f"{spectralRange[1]} --> : "))
        spectralRange = [range1,range2]
        myCal.setGridLine(spectralRange)
        blockImage = myCal.getSpectralBlockImage()
        blockImageLayer.data = blockImage*1


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

#%% calculate 2D histogram

_y = np.reshape(np.swapaxes(spImage2, 0, 2), (-1,spImage2.shape[0]))
_nonzero = np.sum(_y,axis=1)
y = _y[_nonzero>0] 
x = np.zeros_like(y)
x = x + myCal.wavelength

_yCor = np.reshape(np.swapaxes(spImage2Cor, 0, 2), (-1,spImage2.shape[0]))
_nonzero = np.sum(_yCor,axis=1)
yCor = _yCor[_nonzero>0]

bins=[myCal.wavelength.shape[0]*2,myCal.wavelength[::-1]]
H, yedges, xedges = np.histogram2d(y.flatten(),x.flatten(),bins=bins)
HCor, yedges, xedges = np.histogram2d(yCor.flatten(),x.flatten(),bins=bins)

#%% show 2D histogram

viewer5 = napari.Viewer()
viewer5.add_image(H[::-1], name = 'raw', opacity=1,colormap='turbo')
viewer5.add_image(HCor[::-1], name = 'corrected', opacity=1,colormap='turbo')

#%% show hyper spectral image

sViewer = XYWViewer(np.stack((spImage2,spImage2Cor)),myCal.wavelength)
sViewer.run()



#%% save class
if input("Save the calibration [Y/N]? ").lower()=='y':
    print(f'calibration will be saved in folder: {spectralCamera.dataFolder}')
    myCal.saveClass(classFolder=spectralCamera.dataFolder)
