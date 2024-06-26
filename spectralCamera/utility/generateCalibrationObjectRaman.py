''' script to generate calibration object from one Raman Image '''

#%% import and parameter definition

import napari
import spectralCamera
import numpy as np
from spectralCamera.algorithm.calibrateRamanImage import CalibrateRamanImage
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer

import matplotlib.pyplot as plt
# set backend in the case of ipython
try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic("matplotlib","")
except:
    pass

dfolder = spectralCamera.dataFolder
imageFileName = 'RamanCalibrationImage.npy'
darkFileName = 'RamanDarkImage.npy'


#%% data loading
print('loading data')
# load reference image
rawImage = np.load(dfolder + '\\' + imageFileName)
darkImage = np.load(dfolder + '\\' + darkFileName)

# initiate the calibration class
myCal = CalibrateRamanImage()

# load images
myCal.setImageStack(rawImage=rawImage,darkImage=darkImage)

#%% process calibration images

print('prepare Grids')
myCal.prepareGrids()

#%%

print('visual check of grid and block position')

viewer = napari.Viewer()

# white image
viewer.add_image(myCal.image, name='raw Image', colormap='turbo' )

# show spectral block
blockImage = myCal.getSpectralBlockImage()*1
blockImageLayer = viewer.add_image(blockImage, name='spectral block',opacity=0.2)

# located spectral peaks in the calibration images
peakLabel = np.zeros_like(myCal.image, dtype='int')
selectPoint = myCal.gridLine.inside
peakLabel[myCal.peakLeftPosition[selectPoint,0].astype(int),myCal.peakLeftPosition[selectPoint,1].astype(int)] = 1
peakLabel[myCal.peakRightPosition[selectPoint,0].astype(int),myCal.peakRightPosition[selectPoint,1].astype(int)] = 2
viewer.add_labels(peakLabel, name='peaks', opacity=1)

# show zero points
viewer.add_points(myCal.gridLine.xy00, size= 50, opacity=0.2, name= 'zero position')


#%% calculate the calibration warping matrices
print('calculating the calibration -warp - matrix')

myCal.setWarpMatrix(spectral=True, subpixel=True)

#%% visual check of the warping matrices
print('visual check of the calibration routine')

# located spectral peaks in the calibration images
label = np.zeros_like(myCal.image, dtype='int')
avePoint = myCal.gridLine.getPositionInt()
selectPoint = myCal.gridLine.inside
label[myCal.peakLeftPosition[selectPoint,0].astype(int),myCal.peakLeftPosition[selectPoint,1].astype(int)] = 2
label[myCal.peakRightPosition[selectPoint,0].astype(int),myCal.peakRightPosition[selectPoint,1].astype(int)] = 2
px = np.argmin(np.abs(myCal.wavelength - myCal.DEFAULT['kPosition'][0]))
label[avePoint[selectPoint,0].astype(int), avePoint[selectPoint,1].astype(int) -myCal.bwidth + px] = 3
px = np.argmin(np.abs(myCal.wavelength - myCal.DEFAULT['kPosition'][1]))
label[avePoint[selectPoint,0].astype(int), avePoint[selectPoint,1].astype(int) -myCal.bwidth + px] = 3

viewer2 = napari.Viewer()
viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift',colormap='turbo')
viewer2.add_image(myCal.dSpectralWarpMatrix, name='SpectralWarp',colormap='turbo')
viewer2.add_labels(label, name='peaks', opacity=1)

#%% visual check of the warped image

viewer3 = napari.Viewer()
viewer3.add_image(rawImage, name = 'calibration image', opacity=1,colormap='turbo')
viewer3.add_image(myCal.getWarpedImage(rawImage), name = 'warped calibration image', opacity=1,colormap='turbo')
viewer3.add_labels(label, name='peak real and ideal')



#%% show the spectral images

myCal.setWarpMatrix(spectral=True, subpixel=True)
spImage = myCal.getSpectralImage(rawImage,aberrationCorrection=False)
spImageCor = myCal.getSpectralImage(rawImage,aberrationCorrection=True)

viewer3 = napari.Viewer()
viewer3.add_image(spImage, name = 'image not cor', opacity=1,colormap='turbo')
viewer3.add_image(spImageCor, name = 'image cor', opacity=1,colormap='turbo')

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
    print(f'calibration will be saved in folder: {dfolder}')
    myCal.saveClass(classFolder=dfolder)
