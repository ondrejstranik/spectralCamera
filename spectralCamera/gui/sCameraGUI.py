'''
class for tracking of plasmon peaks
'''
#%%
from pathlib import Path
import numpy as np

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui

class SCameraGUI(BaseGUI):
    ''' main class to set parameters in spectral Camera'''

    DEFAULT = {'nameGUI': 'sCamera'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        # prepare the gui of the class
        SCameraGUI.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        @magicgui(aberrationCorrection={"label": "image correction"},
                  spectralCorrection={"label": "spectral correction"},
                  dTimeCamera = {"widget_type":"Label"},
                  dTimeSCamera = {"widget_type":"Label"})
        def sCameraGui(aberrationCorrection=True,
                       spectralCorrection=True,
                       dTimeCamera=0, 
                       dTimeSCamera=0):
            if aberrationCorrection is not None:
                self.device.aberrationCorrection = aberrationCorrection
            if spectralCorrection is not None:
                self.device.spectralCorrection = spectralCorrection
            if dTimeCamera is not None:
                self.sCameraGui.dTimeCamera.value = dTimeCamera
            if dTimeSCamera is not None:
                self.sCameraGui.dTimeSCamera.value = dTimeSCamera

        # add widgets 
        self.sCameraGui = sCameraGui
        self.vWindow.addParameterGui(self.sCameraGui,name=self.DEFAULT['nameGUI'])
 
    def setDevice(self, device):
        super().setDevice(device)

        # connect the signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)
        self.device.camera.worker.yielded.connect(self.guiUpdateTimed)

        # set value in gui
        self.sCameraGui.aberrationCorrection.value = self.device.aberrationCorrection
        self.sCameraGui.spectralCorrection.value = self.device.spectralCorrection

    def updateGui(self):
        ''' update the data in gui '''
        self.sCameraGui.dTimeSCamera.value = self.device.dTime
        self.sCameraGui.dTimeCamera.value = self.device.camera.dTime



if __name__ == "__main__":
    pass


#%%