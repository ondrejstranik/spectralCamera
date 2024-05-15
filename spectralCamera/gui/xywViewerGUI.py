'''
class for live viewing spectral images
'''
#%%
from viscope.gui.baseGUI import BaseGUI
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer

class XYWViewerGui(BaseGUI):
    ''' main class to show xywViewer'''

    DEFAULT = {'nameGUI': 'XYWViewer'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        # prepare the gui of the class
        XYWViewerGui.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        self.XYWViewer = XYWViewer(show=False)
        self.viewer = self.XYWViewer.viewer

        self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI'])

    def setDevice(self,device):
        super().setDevice(device)
        # connect signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)

    def updateGui(self):
        ''' update the data in gui '''
        # napari
        self.XYWViewer.setImage(self.device.sImage)
        self.XYWViewer.setWavelength(self.device.wavelength)



if __name__ == "__main__":
    pass

