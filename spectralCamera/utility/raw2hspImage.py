''' script to convert raw Image to spectral image'''

#%% import and parameter definition

import napari
import numpy as np
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
import spectralCamera

dFolder = spectralCamera.dataFolder
fileName = 'spot_0.npy'
cfile = None

#%% data loading
print('loading data')
# load reference image
image = np.load(dFolder + '\\' + fileName)

# load calibration
myCal = CalibrateFrom3Images()
myCal = myCal.loadClass(classFile=cfile)

spImage = myCal.getSpectralImage(image, aberrationCorrection=True)
wavelength = myCal.getWavelength()

#%% show the data

viewer = napari.Viewer()
viewer.add_image(image)

spViewer = XYWViewer(spImage,wavelength)
spViewer.run()

# %% save the data

np.savez(dFolder + '\\' + fileName,image=spImage,wavelength=wavelength)

# %%
