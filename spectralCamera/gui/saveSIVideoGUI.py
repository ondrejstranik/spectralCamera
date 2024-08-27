'''
class for saving video of spectral images
'''
#%%
from pathlib import Path
import numpy as np

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui

class SaveSIVideoGUI(BaseGUI):
    ''' main class to save video of spectral images'''

    DEFAULT = {'nameGUI': 'Save Video'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        if 'name' in kwargs: self.DEFAULT['nameGUI']= kwargs['name']

        # prepare the gui of the class
        SaveSIVideoGUI.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        @magicgui(filePath={"label": "Saving video Folder:","mode":'d'},
                  call_button="Start Recording"
        )
        def saveVideoGui(filePath= Path(self.viscope.dataFolder)):

            if self.device.flagSaving is False:
                self.device.savingFolder = str(filePath)
                np.save(str(filePath / 'wavelength'),self.device.wavelength)            
                self.device.flagSaving = True
                saveVideoGui.call_button.text = 'Stop Recording'
            else:
                self.device.flagSaving = False
                saveVideoGui.call_button.text = 'Start Recording'

        # add widgets 
        self.saveVideoGui = saveVideoGui
        self.vWindow.addParameterGui(self.saveVideoGui,name=self.DEFAULT['nameGUI'])
 
    def setDevice(self, device):
        super().setDevice(device)

        # set GUI
        if self.device.flagSaving:
            self.saveVideoGui.call_button.text = 'Stop Recording'
        else:
            self.saveVideoGui.call_button.text = 'Start Recording'



if __name__ == "__main__":
    pass


