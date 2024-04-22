''' camera unitest '''

import pytest

@pytest.mark.GUI
def test_calibrateFilterImage():
    ''' check if the FilterCalibration put the channels in proper position'''
    from spectralCamera.algorithm.calibrateFilterImage import CalibrateFilterImage
    import napari
    import numpy as np

    # generate raw image (consisting of superpixels)
    singleImage = np.ones((40,50))
    spec = np.arange(9)
    rawImage = np.zeros((3*40,3*50))
    for ii in range(3):
        for jj in range(3):
            rawImage[ii::3,jj::3] = singleImage*spec[ii*3+jj]

    myCal = CalibrateFilterImage(order=3)

    spImage = myCal.getSpectralImage(rawImage)

    mv = napari.Viewer()
    mv.add_image(rawImage)
    mv.add_image(spImage)
    napari.run() 


def test_calibrateIFImage():
    ''' check  getSpectralImage method '''

    from viscope.instrument.virtual.virtualCamera import VirtualCamera
    from spectralCamera.algorithm.calibrateIFImage import CalibrateIFImage
    
    camera = VirtualCamera()
    camera.connect()
    camera.setParameter('threadingNow',True)

    sCal = CalibrateIFImage(camera=camera)

    rawImage = camera.getLastImage()
    sCal.getSpectralImage(rawImage)

    camera.disconnect() 
