'''
class for gui for loading sequential saved spectral images
'''
#%%
from pathlib import Path
import numpy as np

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui
from magicgui.widgets import Container

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
                                    "step":1
                                    }
        )
        def selectFileGui(filePath= Path(self.viscope.dataFolder),
                          currentFileIdx=1):

            if filePath is not None:
                oldFolder = self.device.getFolder()
                if str(filePath) != str(oldFolder):
                        self.device.setFolder(str(filePath))
                        print(f'setting new folder{filePath} ')

            self.runFileSet.fileSetIdx.max = self.device.nFile
            self.runFileSet.fileSetIdx.value = (1,self.device.nFile)
            selectFileGui.currentFileIdx.label = f"out of {self.device.nFile}, current #: "
            selectFileGui.currentFileIdx.max = self.device.nFile
            if currentFileIdx<1: currentFileIdx=1
            self.device.startReadingImages(idx=[currentFileIdx-1])

        @magicgui(call_button= "Run",
                  fileSetIdx = {"label": "out of 1, current #: ",
                                    "widget_type": "RangeSlider",
                                    "min": 1,
                                    "max": 1,
                                    "step": 1,
                                    }
        )
        def runFileSet(fileSetIdx = (1,1)):
            
            if self.device.isReading:
                self.device.stopReadingImages()
                runFileSet.call_button.text = 'Run'
            else:
                runFileSet.call_button.text = 'Stop'
                _idx = list(range(fileSetIdx[0]-1, fileSetIdx[1] -1))
                self.device.startReadingImages(idx=_idx)
                
             

        # add widgets 
        self.selectFileGui = selectFileGui
        self.runFileSet = runFileSet

        self.container = Container(widgets=[self.selectFileGui,self.runFileSet])

        self.vWindow.addParameterGui(self.container,name=self.DEFAULT['nameGUI'])
 

    def setDevice(self, device):
        super().setDevice(device)

        # connect the signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)
        self.selectFileGui.filePath.value =str(self.device.getFolder())            

    def updateGui(self):
        ''' update the data in gui '''
        if self.device.processor == 'GUI':
            self.device.flagToProcess.set()
        if not self.device.isReading:
            self.runFileSet.call_button.text = 'Run'
        self.selectFileGui._auto_call = False
        self.selectFileGui.currentFileIdx.value = self.device.currentIdx +1
        self.selectFileGui._auto_call = True


if __name__ == "__main__":
    pass


#%%