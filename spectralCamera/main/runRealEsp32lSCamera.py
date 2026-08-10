'''
script to run real mil camera
'''
#%%
#devices
from spectralCamera.instrument.camera.esp32Camera.esp32Camera import ESP32Camera
from spectralCamera.instrument.sCamera.sCamera import SCamera 
from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage

#gui
from viscope.gui.allDeviceGUI import AllDeviceGUI
from viscope.main import viscope
from spectralCamera.gui.xywViewerGUI import XYWViewerGui

def main():
    camera = ESP32Camera(name='ESP32Camera',filterType='RGGB')
    camera.connect()
    camera.setParameter('threadingNow',True)

    sCal = CalibrateRGBImage()

    sCamera = SCamera(name='spectralWebCamera')
    sCamera.connect(camera=camera)
    sCamera.setParameter('calibrationData',sCal)
    sCamera.setParameter('threadingNow',True)  

    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)

    svGui  = XYWViewerGui(viscope)
    svGui.setDevice(sCamera)

    viscope.run()

    camera.disconnect()

if __name__ == "__main__":
    main()
    
    


