'''
class to calibrate RGB Images
'''

import numpy as np


import napari


class CalibrateIFImage():
    ''' main class to calibrate virtual integral field images '''

    DEFAULT = {'wavelengthRange': np.array([400,800]), # 
               'order' : 3 } # number of filter channels Order**2 (aligned in a square)

    def __init__(self,order=None, wavelengthRange=None):
        ''' initialise the class '''
        
        if order is None:
            self.order = self.DEFAULT['order']
        else:
            self.order = order     

        if wavelengthRange is None:
            self.wavelengthRange = self.DEFAULT['wavelengthRange']
        else:
            self.wavelengthRange = wavelengthRange

        self.wavelength = np.linspace(self.wavelengthRange[0],self.wavelengthRange[1],self.order**2)

    def getSpectralImage(self,rawImage,**kwargs):
        ''' get the spectral image from raw image'''


        # crop to the proper size
        myShape = np.array(np.shape(rawImage))//self.order*self.order
        #WYXImage = rawImage[0:myShape[0]+1,0:myShape[1]+1]
        WYXImage = rawImage[0:myShape[0],0:myShape[1]]


        WYXImage = np.reshape(WYXImage,
                    (myShape[0]//self.order,self.order,myShape[1]//self.order,self.order))
        WYXImage = np.swapaxes(WYXImage,1,2)
        WYXImage = np.reshape(WYXImage,
                    (myShape[0]//self.order,myShape[1]//self.order,self.order**2))
        WYXImage = np.swapaxes(WYXImage,1,2)
        WYXImage = np.swapaxes(WYXImage,0,1)


        return  WYXImage

    def getWavelength(self):
        ''' get the wavelengths '''
        return self.wavelength

if __name__ == "__main__":

    pass
































