import pytest

def test_disperseIntoBlock():
    ''' check the virtual dispersion intoBlock '''
    from spectralCamera.algorithm.calibrateFilterImage import CalibrateFilterImage
    import napari
    import numpy as np
    from spectralCamera.virtualSystem.component.component2 import Component2

    # generate raw image (consisting of superpixels)
    singleImage = np.ones((40,50))
    spec = np.arange(9)
    rawImage = np.zeros((3*40,3*50))
    for ii in range(3):
        for jj in range(3):
            rawImage[ii::3,jj::3] = singleImage*spec[ii*3+jj]

    # spectral image
    myCal = CalibrateFilterImage(order=3)
    spImage = myCal.getSpectralImage(rawImage)

    # dispersed image 
    diImage = Component2.disperseIntoBlock(spImage,np.array((3,3)))

    # they should be equal
    assert rawImage == pytest.approx(diImage)

@pytest.mark.GUI
def test_disperseIntoLines():
    ''' visual check of the line dispersion '''
    import napari
    import numpy as np
    from spectralCamera.virtualSystem.component.component2 import Component2
    
    
    spImage = np.random.rand(30,5,10) +1
    oFrame =Component2.disperseIntoLines(spImage, gridVector=[4,10])

    viewer = napari.view_image(oFrame)

    napari.run()


@pytest.mark.GUI
def test_getModifiedSpectraImage():
    ''' check the spectral resampling '''

    from spectralCamera.virtualSystem.component.sample2 import Sample2
    from spectralCamera.virtualSystem.component.component2 import Component2
    from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
    import numpy as np

    sample = Sample2()
    sample.setSpectralDisk()    

    newWavelength = np.array([400,500,700]) 
    newImage = Component2.getModifiedSpectraImage(sample.get(),sample.getWavelength(),newWavelength)

    sViewer = XYWViewer(sample.get(),sample.getWavelength())
    sViewer2 = XYWViewer(newImage,newWavelength)

    sViewer.run()