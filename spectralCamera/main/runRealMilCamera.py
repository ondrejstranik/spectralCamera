'''
script to run real mil camera
'''
#%%
#devices
from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera  

#gui
import spectralCamera
from viscope.main import viscope
from viscope.gui.allDeviceGUI import AllDeviceGUI 
from viscope.gui.saveImageGUI import SaveImageGUI

def main():
    # some global settings
    viscope.dataFolder = spectralCamera.dataFolder

    camera = MilCamera(name='MilCamera')
    camera.connect()
    camera.setParameter('exposureTime', 5)
    camera.setParameter('threadingNow',True)

    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)
    newGUI = SaveImageGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()

if __name__ == "__main__":
    main()
    
    


