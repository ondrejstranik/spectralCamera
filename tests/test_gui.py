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

def test_webCamera2():
    ''' check if camera works with viscope gui '''
    from viscope.gui.cameraViewGUI import CameraViewGUI
    from spectralCamera.instrument.webCamera.webCamera import WebCamera
    from viscope.main import Viscope

    camera = WebCamera(name='WebCamera')
    camera.connect()

    viscope = Viscope()
    newGUI  = CameraViewGUI(viscope)
    newGUI.setDevice(camera)
    viscope.run()

    camera.disconnect()