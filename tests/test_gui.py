''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_XYWViewerGUI():
    ''' testing the viewer with webcam'''
    from viscope.main import viscope
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui
    from spectralCamera.gui.sCameraGUI import SCameraGUI

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
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)
    newGUI  = SCameraGUI(viscope)
    newGUI.setDevice(sCamera)

    viscope.run()

    camera.disconnect()
    sCamera.disconnect()

@pytest.mark.GUI
def test_SViewerGUI():
    ''' testing the viewer with webcam'''
    from viscope.main import viscope
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
    from spectralCamera.gui.sViewerGUI import SViewerGui
    from spectralCamera.gui.sCameraGUI import SCameraGUI

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
    newGUI  = SViewerGui(viscope)
    newGUI.setDevice(sCamera)
    newGUI  = SCameraGUI(viscope)
    newGUI.setDevice(sCamera)

    viscope.run()

    camera.disconnect()
    sCamera.disconnect()



@pytest.mark.GUI
def test_saveSIVideoGUI():
    ''' testing the viewer with webcam'''
    from viscope.main import viscope
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera    
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui
    from spectralCamera.gui.saveSIVideoGUI import SaveSIVideoGUI

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
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)
    newGUI  = SaveSIVideoGUI(viscope)
    newGUI.setDevice(sCamera)

    viscope.run()

    camera.disconnect()
    sCamera.disconnect()