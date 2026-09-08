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
       spectral range, then press "Get Blocks" to fit the super-pixel grid
       (shown as a visual check plot plus the fitted grid, spectral peaks
       and zero points in napari - pressing it again redoes the fit from
       scratch), then "Calculate Warp" to fit the warping matrices and
       plot the full set of checks, and finally "Save" to store the result
'''
#%%
#devices
try:
    from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera
    cameraOnDevice = True
except:
    cameraOnDevice = False

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

    if cameraOnDevice:
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
