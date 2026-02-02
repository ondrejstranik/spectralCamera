'''
class for live viewing spectral images
'''
#%%
from viscope.gui.baseGUI import BaseGUI
from spectralCamera.gui.spectralViewer.sViewer import SViewer

class SViewerGUI(BaseGUI):
    ''' main class to show SViewerGUI
    calculation of the spectra are done in the main thread of the SViewer
    '''

    DEFAULT = {'nameGUI': 'SViewer'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        # prepare the gui of the class
        SViewerGUI.__setWidget(self) 

    def __setWidget(self):
        ''' prepare the gui '''

        self.sViewer = SViewer(show=False)
        self.viewer = self.sViewer.viewer

        self.vWindow.addMainGUI(self.viewer.window._qt_window, name=self.DEFAULT['nameGUI'])

    def setDevice(self,device):
        super().setDevice(device)
        # connect signals
        self.device.worker.yielded.connect(self.guiUpdateTimed)

    def updateGui(self):
        ''' update the data in gui '''
        # napari
        self.sViewer.setWavelength(self.device.wavelength)
        self.sViewer.setImage(self.device.sImage)


if __name__ == "__main__":
    pass

