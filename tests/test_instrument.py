''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_sCameraStatic():
    ''' check if gui works'''
    from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImages import CalibrateRGBImage
    
    camera = WebCamera(name='WebCamera')
    camera.connect()

    sCal = CalibrateRGBImage(rgbOrder='RGB')

    sCamera = SCamera(name='spectralWebCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('calibrationData',sCal)

    im = sCamera.getLastSpectralImage()
    wavelength = sCamera.getParameter('wavelength')

    sViewer = XYWViewer(im, wavelength)
    sViewer.run()
    camera.disconnect()

def test_sCamera():
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera

    import numpy as np
    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    sCamera = SCamera(name='RGBWebCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('threadingNow',True)

    for ii in range(5):
        sCamera.flagLoop.wait()
        print(f'worker loop reported: {ii+1} of 5')
        print(f' spectral Image sum: {np.sum(sCamera.sImage)}')
        sCamera.camera.flagLoop.clear()

    camera.disconnect()
    sCamera.disconnect()