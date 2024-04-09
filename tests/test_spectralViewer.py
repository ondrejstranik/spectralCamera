''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_XYWViewer():
    ''' check if gui works'''
    from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
    import numpy as np
    im = np.random.rand(10,100,100)
    wavelength = np.arange(im.shape[0])*1.3+ 10
    sViewer = XYWViewer(im, wavelength)
    sViewer.run()

@pytest.mark.GUI
def test_TransmissionViewer():
    ''' check if gui works'''
    from spectralCamera.gui.spectralViewer.transmissionViewer import TransmissionViewer
    import numpy as np
    im = np.random.rand(10,100,100)
    wavelength = np.arange(im.shape[0])*1.3+ 10
    sViewer = TransmissionViewer(im, wavelength)
    sViewer.run()