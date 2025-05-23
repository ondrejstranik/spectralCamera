''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_webCamera():
    ''' check if web camera work'''
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera

    cam = WebCamera(name='WebCamera',filterType='RGGB')
    cam.connect()
    cam.setParameter('exposureTime',300)
    cam.setParameter('nFrame', 5)

    cam._displayStreamOfImages()
    cam.disconnect()

@pytest.mark.GUI
def test_webCamera2():
    ''' check if camera works with viscope gui '''
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from viscope.main import viscope
    from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera  

    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()

@pytest.mark.GUI
def test_milCamera():
    ''' check if mil camera works with viscope gui '''
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from viscope.main import viscope
    from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera  

    camera = MilCamera(name='MilCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()

def test_milCamera2():
    ''' check if mil camera works with viscope gui '''
    import spectralCamera
    from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera  
    from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from viscope.main import viscope

    camera = MilCamera(name='MilCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    sCal = CalibrateFrom3Images()
    sCal = sCal.loadClass()

    sCamera = SCamera(name='spectralWebCamera')
    sCamera.connect(camera=camera)
    sCamera.setParameter('calibrationData',sCal)



    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()


@pytest.mark.GUI
def test_PFCamera():
    ''' check if PFCamera works'''
    from spectralCamera.instrument.camera.pfCamera.pFCamera import PFCamera

    cam = PFCamera(name='pfCamera')
    cam.connect()
    cam.setParameter('exposureTime',300)
    cam.setParameter('nFrame', 5)

    cam._displayStreamOfImages()
    cam.disconnect()

@pytest.mark.GUI
def test_PFCamera2():
    ''' check if camera works with viscope gui '''
    from viscope.gui.allDeviceGUI import AllDeviceGUI
    from viscope.main import viscope
    from spectralCamera.instrument.camera.pfCamera.pFCamera import PFCamera 

    cam = PFCamera(name='pfCamera')
    cam.connect()
    cam.setParameter('exposureTime',300)
    #cam.setParameter('nFrame', 5)
    cam.setParameter('threadingNow',True)

    adGui  = AllDeviceGUI(viscope)
    adGui.setDevice(cam)
    viscope.run()

    cam.disconnect()

@pytest.mark.GUI
def test_calibratePFImage():
    ''' check if calibration of photon focus camera work'''
    from spectralCamera.instrument.camera.pfCamera.pFCamera import PFCamera 
    from spectralCamera.algorithm.calibratePFImage import CalibratePFImage
    from spectralCamera.instrument.sCamera.sCamera import SCamera
    from viscope.main import viscope
    from spectralCamera.gui.xywViewerGUI import XYWViewerGui
    from viscope.gui.allDeviceGUI import AllDeviceGUI    

    camera = PFCamera(name='pfCamera')
    camera.connect()
    camera.setParameter('exposureTime',300)
    camera.setParameter('threadingNow',True)

    sCal = CalibratePFImage()

    sCamera = SCamera(name='spectralPFCamera')
    sCamera.connect(camera=camera)
    sCamera.setParameter('calibrationData',sCal)
    sCamera.setParameter('threadingNow',True)    

    adGui  = AllDeviceGUI(viscope)
    adGui.setDevice(camera)
    
    svGui  = XYWViewerGui(viscope)
    svGui.setDevice(sCamera)

    viscope.run()

    camera.disconnect()
    sCamera.disconnect()




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
    sCamera.connect(camera=camera)
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
    sCamera.connect(camera)
    sCamera.setParameter('threadingNow',True)

    for ii in range(5):
        sCamera.flagLoop.wait()
        print(f'worker loop reported: {ii+1} of 5')
        print(f' spectral Image sum: {np.sum(sCamera.sImage)}')
        sCamera.camera.flagLoop.clear()

    camera.disconnect()
    sCamera.disconnect()
# %%
