''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_webCamera():
    ''' check if web camera work'''
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera

    cam = WebCamera(name='WebCamera',filterType='RGGB')
    cam.connect()
    cam.setParameter('exposureTime',300)
    cam.setParameter('nFrames', 5)

    cam._displayStreamOfImages()
    cam.disconnect()

@pytest.mark.GUI
def test_webCamera2():
    ''' check if camera works with viscope gui '''
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from viscope.main import Viscope
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera  

    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    viscope = Viscope()
    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()


@pytest.mark.GUI
def test_sCameraStatic():
    ''' check if spectral camera give a proper image'''
    from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
    
    camera = WebCamera(name='WebCamera')
    camera.connect()

    sCal = CalibrateRGBImage()

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
    ''' check if thread looping in the spectral camera is working properly '''
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