'''
class for live viewing spectral images
'''
#%%
#import napari
#from magicgui import magicgui
#from typing import Annotated, Literal

#from qtpy.QtWidgets import QLabel, QSizePolicy, QDockWidget
#from qtpy.QtCore import Qt
from viscope.gui.baseGUI import BaseGUI
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
from timeit import default_timer as timer

#from timeit import default_timer as timer
#import napari

#import numpy as np

class XYWViewerGui(BaseGUI):
    ''' main class to show xywViewer'''

    DEFAULT = {'nameGUI': 'XYWViewer'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        self.lastUpdateTime = timer()
        self.guiUpdateTime = 0.03

        # prepare the gui of the class
        XYWViewerGui.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        self.XYWViewer = XYWViewer(show=False)
        self.viewer = self.XYWViewer.viewer

        self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI'])

    def guiUpdateTimed(self):
        ''' update gui according the update time '''
        timeNow = timer()
        if (timeNow -self.lastUpdateTime) > self.guiUpdateTime:
            self.updateGui()
            self.lastUpdateTime = timeNow    

    def setDevice(self,device):
        super().setDevice(device)
        # connect signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)
        self.vWindow.setWindowTitle(self.device.name)

    def updateGui(self):
        ''' update the data in gui '''
        # napari
        self.XYWViewer.setImage(self.device.sImage)
        self.XYWViewer.setWavelength(self.device.wavelength)



if __name__ == "__main__":
    pass

