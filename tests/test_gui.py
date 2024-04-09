''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_webCamera():
    ''' check if gui works'''
    from spectralCamera.instrument.webCamera.webCamera import WebCamera

    cam = WebCamera()
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
    from spectralCamera.instrument.webCamera.webCamera import WebCamera

    camera = WebCamera(name='WebCamera')
    camera.connect()
    camera.setParameter('threadingNow',True)

    viscope = Viscope()
    newGUI  = AllDeviceGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()