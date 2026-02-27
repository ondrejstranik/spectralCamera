'''
class for gui for loading sequential saved spectral images
'''
#%%
from pathlib import Path
import numpy as np

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui

class SCameraFromFileGUI(BaseGUI):
    ''' main class to set parameters in spectral Camera from File via GUI'''

    DEFAULT = {'nameGUI': 'sCamera'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)


        # prepare the gui of the class
        SCameraFromFileGUI.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        @magicgui(auto_call= True,
                  filePath={"label": "Saving video Folder:","mode":'d'},
                  currentFileIdx = {"label": "out of 1, current #: ",
                                    "widget_type": "Slider",
                                    "min": 1,
                                    "max": 1,
                                    }
        )
        def selectFileGui(filePath= Path(self.viscope.dataFolder),
                          currentFileIdx=1):

            oldFolder = self.device.getFolder()
            if str(filePath) != str(oldFolder):
                    self.device.setFolder(str(filePath))
                    print(f'setting new folder{filePath} ')
            selectFileGui.currentFileIdx.label = f"out of {self.device.nFile}, current #: "
            selectFileGui.currentFileIdx.max = self.device.nFile
            if currentFileIdx<1: currentFileIdx=1
            self.device.startReadingImages(idx=[currentFileIdx-1])

        # add widgets 
        self.selectFileGui = selectFileGui
        self.vWindow.addParameterGui(self.selectFileGui,name=self.DEFAULT['nameGUI'])
 

    def setDevice(self, device):
        super().setDevice(device)

        # connect the signals
        #self.device.worker.yielded.connect(self.guiUpdateTimed)
        self.selectFileGui.filePath.value =str(self.device.getFolder())


    def updateGui(self):
        ''' update the data in gui '''
        pass



if __name__ == "__main__":
    pass


#%%