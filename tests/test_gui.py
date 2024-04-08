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

