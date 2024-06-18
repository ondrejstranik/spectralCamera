''' script to generate calibration object from three images '''

#%% import and parameter definition

import napari
import spectralCamera
import numpy as np
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images

dfolder = spectralCamera.dataFolder
whiteFileName = 'filter_wo_0.npy'
imageNameStack = None
wavelengthStack = None


#%% data loading

# load reference image
whiteImage = np.load(dfolder + '\\' + 'whiteFileName')

myCal = CalibrateFrom3Images()

if imageNameStack is not None: myCal.imageNameStack = imageNameStack
if wavelengthStack is not None: myCal.wavelengthStack = wavelengthStack


# initiate the calibration class
myCal = CalibrateFrom3Images()
# load images
myCal.setImageStack()

#%% process calibration images

myCal.processImageStack()
myCal._setGlobalGridZero()
myCal._setPositionMatrix()

#%% visual check that grids are on proper 

viewer = napari.Viewer()
#viewer.add_image(myCal.mask)
viewer.add_image(whiteImage, name='white')

for ii,iS in enumerate(myCal.imageStack):
    viewer.add_image(iS,name = myCal.wavelengthStack[ii], opacity=0.5)
