'''
script to acquire spectral calibration images with the mil camera and then
calibrate the spectral camera from them

workflow:
    1) use the live camera view + "Save Image" panel to save the three
       narrow-band filter images needed for calibration (e.g. fileName
       "filter_602", one shot per filter wavelength - saved as
       filter_602_0.npy) plus a white reference image
    2) in the "Calibration" panel, pick each of those files (each shows up
       live in napari as soon as picked) and set the matching wavelengths/
       spectral range, then press "Calibrate" to run the fit and "Save" to
       store the result
'''
#%%
#devices
from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera

#gui
import spectralCamera
from viscope.main import viscope
from viscope.gui.cameraGUI import CameraGUI
from viscope.gui.cameraView2GUI import CameraView2GUI
from viscope.gui.saveImageGUI import SaveImageGUI
from spectralCamera.gui.calibrationGUI import CalibrationGUI

def main():
    # some global settings
    viscope.dataFolder = spectralCamera.dataFolder

    camera = MilCamera(name='MilCamera')
    camera.connect()
    camera.setParameter('exposureTime', 5)
    camera.setParameter('threadingNow', True)

    # live camera view, pyqtgraph-based (CameraView2GUI) instead of the
    # napari-based CameraViewGUI that AllDeviceGUI would normally wire up
    # for a camera device - same manual construction AllDeviceGUI does
    # internally, just swapping the viewer GUI class
    liveViewWindow = viscope.addViewerWindow()
    newGUI = CameraGUI(viscope, vWindow=liveViewWindow)
    newGUI.setDevice(camera)
    newGUI = CameraView2GUI(viscope, vWindow=liveViewWindow)
    newGUI.setDevice(camera)

    newGUI = SaveImageGUI(viscope)
    newGUI.setDevice(camera)

    CalibrationGUI(viscope)

    viscope.run()

    camera.disconnect()

if __name__ == "__main__":
    main()
