#%%
''' 
class for identification of plasmonic spots
'''
import HSIplasmon as hsi

from skimage.filters import threshold_otsu, threshold_local, gaussian, rank
from skimage.morphology import binary_closing, square, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
import skimage as ski

import numpy as np


class spotIdentification:
    ''' identification of plasmon spots '''
    DEFAULT = {'radius':15}

    def __init__(self,spectralImage):
        ''' initialization '''

        self.spotPosition = None
        self.spectralImage = spectralImage
        self.spotRadius = None

    def getPosition(self):
        ''' get the position of plasmonic spots '''
        
        # make contrast image by sum projection
        # TODO: (optional) improve contrast by selecting only part of the spectral
        _sumIm = np.sum(self.spectralImage, axis=0)
        _sumIm = _sumIm/_sumIm.max()
        sumIm = ski.util.img_as_uint(gaussian(_sumIm))

        # use local otsu to threshold image
        footprint = disk(self.DEFAULT['radius'])
        local_otsu = rank.otsu(sumIm, footprint)
        mask = sumIm < local_otsu

        # calculate properties of objects
        labels = label(mask)
        props = regionprops_table(labels,
                properties=['label', 'centroid', 'eccentricity', 'equivalent_diameter_area'])

        # subselect the objects
        # based on size and eccentricity
        dAve = np.median(props['equivalent_diameter_area'])
        dEcc = np.median(props['eccentricity'])

        selSpots = ((props['equivalent_diameter_area']< 2*dAve) & 
                    (props['equivalent_diameter_area']> dAve/2) &
                    (props['eccentricity'] < 1.5*dEcc))

        _myPosition = np.vstack((props['centroid-0'],props['centroid-1'])).T
        self.spotPosition = _myPosition[selSpots]

        self.spotRadius = dAve/2

        return self.spotPosition

    def getRadius(self):
        ''' get the average radius of spots '''
        return self.spotRadius

if __name__ == "__main__":

    import napari

    # load reference image
    container = np.load(hsi.dataFolder + '/plasmonicArray.npz')
    spImage = container['arr_0']
    #sViewer = SpectraViewerModel2(container['arr_0'], container['arr_1'])

    # identify the spot
    sI = spotIdentification(spImage)
    myPosition = sI.getPosition()
    myRadius = sI.getRadius()

    # show the result
    viewer = napari.Viewer()
    viewer.add_image(spImage)
    viewer.add_points(myPosition, face_color= 'red', size=myRadius*2)




    