'''
script to acquire spectral calibration images with the mil camera and then
calibrate the spectral camera from them

workflow:
    1) use the live camera view + "Save Image" panel to save the three
       narrow-band filter images needed for calibration (e.g. fileName
       "filter_602", one shot per filter wavelength - saved as
       filter_602_0.npy, matching the "Calibration" panel's file name
       fields)
    2) in the "Calibration" panel, set the file names/wavelengths/spectral
       range/folder to match, then press "Show images" to check the raw
       data, "Calibrate" to run the fit, and "Save" to store the result
'''
#%%
#devices
from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera

#gui
import spectralCamera
from viscope.main import viscope
from viscope.gui.allDeviceGUI import AllDeviceGUI
from viscope.gui.saveImageGUI import SaveImageGUI
from spectralCamera.gui.calibrationGUI import CalibrationGUI

def main():
    # some global settings
    viscope.dataFolder = spectralCamera.dataFolder

    camera = MilCamera(name='MilCamera')
    camera.connect()
    camera.setParameter('exposureTime', 5)
    camera.setParameter('threadingNow', True)

    newGUI = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)

    newGUI = SaveImageGUI(viscope)
    newGUI.setDevice(camera)

    CalibrationGUI(viscope)

    viscope.run()

    camera.disconnect()

if __name__ == "__main__":
    main()
