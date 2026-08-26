'''
class for gui for loading sequential saved spectral images
'''
#%%
from pathlib import Path
import logging
import numpy as np

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui
from magicgui.widgets import Container

logger = logging.getLogger(__name__)

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
                        logger.info(f'setting new folder{filePath} ')
                # keep the widget's own display in sync - calling this
                # FunctionGui directly with a kwarg does not update its
                # bound widget's value on its own
                selectFileGui.filePath.value = str(filePath)

            # nFile can be 0 if the folder has no matching files yet; fall
            # back to a valid (min 1) range instead of crashing on it
            self.runFileSet.fileSetIdx.max = max(self.device.nFile, 1)
            self.runFileSet.fileSetIdx.value = (1, max(self.device.nFile, 1))
            selectFileGui.currentFileIdx.label = f"out of {self.device.nFile}, current #: "
            selectFileGui.currentFileIdx.max = max(self.device.nFile, 1)
            if currentFileIdx<1: currentFileIdx=1
            if self.device.nFile > 0:
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