'''
class for live viewing spectral images
'''
#%%

import spectralCamera
from viscope.main import viscope
from viscope.gui.allDeviceGUI import AllDeviceGUI 
from viscope.gui.saveImageGUI import SaveImageGUI

import numpy as np
from pathlib import Path

class SpectralCamera():
    ''' base top class for control'''

    DEFAULT = {}

    @classmethod
    def runImageRecordingReal(cls):
        from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera  

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

    SpectralCamera.runImageRecordingReal()
    


