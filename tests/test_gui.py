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
def test_XYWViewerGUI():

    from viscope.main import Viscope
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui

    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    sCal = CalibrateRGBImage(rgbOrder='RGGB')

    sCamera = SCamera(name='sCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('calibrationData',sCal)

    sCamera.setParameter('threadingNow',True)

    print('starting main event loop')
    viscope = Viscope()
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)
    viscope.run()

    camera.disconnect()
    sCamera.disconnect()


