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

@pytest.mark.GUI
def test_simpleSpectralMicroscope2():
    ''' check if virtual microscope works - show spectral and raw data'''
    #from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera
    from viscope.main import Viscope
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui

    from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope

    from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualRGBCamera

    #spectral camera system
    scs = VirtualRGBCamera(rgbOrder='RGB')
    camera = scs.camera
    sCamera = scs.sCamera

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

@pytest.mark.GUI
def test_simpleSpectralMicroscope3():
    ''' reference test to test2 but with real webcam'''
    from viscope.main import Viscope
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui

    from spectralCamera.instrument.sCamera.sCameraGenerator import RGBWebCamera

    from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope


    #spectral camera system
    scs = RGBWebCamera(rgbOrder='RGB')
    camera = scs.camera
    sCamera = scs.sCamera

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

@pytest.mark.GUI
def test_multiSpectralMicroscope():
    from viscope.instrument.virtual.virtualCamera import VirtualCamera
    #from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualFilterCamera
    from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualIFCamera

    from viscope.main import Viscope
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui
    from viscope.gui.allDeviceGUI import AllDeviceGUI

    from spectralCamera.virtualSystem.multiSpectralMicroscope import MultiSpectralMicroscope
    
    #camera
    camera2 = VirtualCamera(name='BWCamera')
    camera2.connect()
    camera2.setParameter('threadingNow',True)

    #spectral camera system
    #scs = VirtualFilterCamera()
    scs = VirtualIFCamera()

    camera = scs.camera
    sCamera = scs.sCamera

    # virtual microscope
    vM = MultiSpectralMicroscope()
    vM.setVirtualDevice(sCamera=sCamera, camera2=camera2)
    vM.connect()

    # main event loop
    viscope = Viscope()
    newGUI  = XYWViewerGui(viscope)
    newGUI.setDevice(sCamera)

    viewer  = AllDeviceGUI(viscope)
    viewer.setDevice([camera,camera2])

    viscope.run()

    sCamera.disconnect()
    camera.disconnect()
    camera2.disconnect()
    vM.disconnect()