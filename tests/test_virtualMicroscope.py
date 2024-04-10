''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_simpleSpectralMicroscope():
    ''' check if virtual microscope works - show raw data'''
    from viscope.instrument.virtual.virtualCamera import VirtualCamera
    from viscope.main import Viscope
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope

    camera1 = VirtualCamera()
    camera1.connect()
    camera1.setParameter('threadingNow',True)

    vM = SimpleSpectralMicroscope()
    vM.setVirtualDevice(camera1)
    vM.connect()

    viscope = Viscope()
    viewer  = AllDeviceGUI(viscope)
    viewer.setDevice([camera1])
    
    viscope.run()

    camera1.disconnect()
    vM.disconnect()

def test_simpleSpectralMicroscope2():
    ''' check if virtual microscope works - show spectral and raw data'''
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera
    from viscope.main import Viscope
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImages import CalibrateRGBImage
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui

    #camera
    camera = VirtualCamera()
    camera.connect()
    camera.setParameter('threadingNow',True)

    #spectral camera
    sCal = CalibrateRGBImage(rgbOrder='RGB')
    sCamera = SCamera(name='sCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('calibrationData',sCal)
    sCamera.setParameter('threadingNow',True)


    # virtual microscope
    vM = SimpleSpectralMicroscope()
    vM.setVirtualDevice(camera)
    vM.connect()

    # add gui
    viscope = Viscope()
    viewer  = AllDeviceGUI(viscope)
    viewer.setDevice(camera)
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)

    # main event loop
    viscope.run()

    camera.disconnect()
    sCamera.disconnect()
    vM.disconnect()

def test_simpleSpectralMicroscope3():
    ''' reference test to test2 but with real webcam'''
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera
    from viscope.main import Viscope
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from spectralCamera.algorithm.calibrateRGBImages import CalibrateRGBImage
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui

    #camera
    camera = WebCamera()
    camera.connect()
    camera.setParameter('threadingNow',True)

    #spectral camera
    sCal = CalibrateRGBImage(rgbOrder='RGB')
    sCamera = SCamera(name='sCamera')
    sCamera.connect()
    sCamera.setParameter('camera',camera)
    sCamera.setParameter('calibrationData',sCal)
    sCamera.setParameter('threadingNow',True)

    # add gui
    viscope = Viscope()
    viewer  = AllDeviceGUI(viscope)
    viewer.setDevice(camera)
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)

    # main event loop
    viscope.run()

    camera.disconnect()
    sCamera.disconnect()
