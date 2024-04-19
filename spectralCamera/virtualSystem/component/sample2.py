"""
class to generate virtual sample

@author: ostranik
"""
#%%

import numpy as np
from skimage import data
from skimage.transform import resize
from viscope.virtualSystem.component.sample import Sample
from skimage.draw import disk


class Sample2(Sample):
    ''' class to define a sample object of the microscope'''
    DEFAULT = {}
    
    def __init__(self,*args, **kwargs):
        ''' initialisation '''
        super().__init__(*args, **kwargs)
        self.wavelength = None

    def setSpectralAstronaut(self,samplePixelSize=None,
                        sampleSize= None,
                        photonRateMax= None,
                        samplePosition = None,
                        wavelength  = None):
        ''' define the sample.
        sample ... spatial distribution of photon rates [#/s/pixelSize^2] (no noise)'''

        DEFAULT = {'photonRateMax':1e6,
                    'samplePixelSize':1, # um
                    'sampleSize': (200,400),
                    'samplePosition': np.array([0,0,0]), # pixels
                    'wavelength': np.array([400,500,600])} # nm

        self.pixelSize=DEFAULT['samplePixelSize'] if samplePixelSize is None else samplePixelSize
        self.size=DEFAULT['sampleSize'] if sampleSize is None else sampleSize
        self.position=DEFAULT['samplePosition'] if samplePosition is None else samplePosition
        self.wavelength = DEFAULT['wavelength'] if wavelength is None else wavelength


        photonRateMax=DEFAULT['photonRateMax'] if photonRateMax is None else photonRateMax        

        # define
        _sample = np.moveaxis(data.astronaut(),-1, 0)

        # resize 
        _sample = resize(_sample, (_sample.shape[0],*self.size))

        # normalise
        _sample = _sample/np.max(_sample)*photonRateMax

        self.data = _sample

    def setSpectralDisk(self,samplePixelSize=None,
                        sampleSize= None,
                        photonRateMax= None,
                        samplePosition = None,
                        wavelength  = None):

        DEFAULT = {'photonRateMax':1e6,
                    'samplePixelSize':1, # um
                    'sampleSize': (200,400),
                    'samplePosition': np.array([0,0,0]),  # pixels
                    'wavelength': np.arange(400,800,10)}

        self.pixelSize=DEFAULT['samplePixelSize'] if samplePixelSize is None else samplePixelSize
        self.size=DEFAULT['sampleSize'] if sampleSize is None else sampleSize
        self.position=DEFAULT['samplePosition'] if samplePosition is None else samplePosition
        self.wavelength = DEFAULT['wavelength'] if wavelength is None else wavelength

        photonRateMax=DEFAULT['photonRateMax'] if photonRateMax is None else photonRateMax        

        _sample = np.zeros((self.wavelength.shape[0],*self.size))

        # fixed disk properties
        # x,y,radius,amplitude,central wavelength, standard deviation
        diskList = [[10,20,5,1,400,80],
                [50,80,20,1,700,10],
                [200,300,50,0.4,550, 80]]
        
        for _disk in diskList: 
            rr, cc = disk((_disk[0],_disk[1]), _disk[2], shape=_sample.shape[1:])
            _sample[:,rr,cc] = _disk[3]*np.exp(-(self.wavelength-_disk[4])**2/2/_disk[5]**2)[:,None]

        # normalise
        _sample = _sample/np.max(_sample)*photonRateMax

        self.data = _sample

    def setCalibrationImage(self,samplePixelSize=None,
                        sampleSize= None,
                        photonRateMax= None,
                        samplePosition = None,
                        wavelength  = None,
                        calibrationWavelength = None):

        DEFAULT = {'photonRateMax':1e6,
                    'samplePixelSize':1, # um
                    'sampleSize': (200,400),
                    'samplePosition': np.array([0,0,0]),  # pixels
                    'wavelength': np.arange(400,800,10),
                    'calibrationWavelength': np.array([500,700])}

        self.pixelSize=DEFAULT['samplePixelSize'] if samplePixelSize is None else samplePixelSize
        self.size=DEFAULT['sampleSize'] if sampleSize is None else sampleSize
        self.position=DEFAULT['samplePosition'] if samplePosition is None else samplePosition
        self.wavelength = DEFAULT['wavelength'] if wavelength is None else wavelength
        self.calibrationWavelength = DEFAULT['calibrationWavelength'] if calibrationWavelength is None else calibrationWavelength       

        photonRateMax=DEFAULT['photonRateMax'] if photonRateMax is None else photonRateMax        

        # give constant spectral  background
        _sample = np.ones((self.wavelength.shape[0],*self.size))

        # adjust calibration wavelength on the whole pixels
        cW0idx = np.argmin(np.abs(self.wavelength-self.calibrationWavelength[0]))
        cW1idx = np.argmin(np.abs(self.wavelength-self.calibrationWavelength[1]))
        self.calibrationWavelength = self.wavelength[[cW0idx,cW1idx]]

        # set the two calibration wavelength
        _sample[cW0idx,...]= 5
        _sample[cW1idx,...]= 3

        # normalise
        _sample = _sample/np.max(_sample)*photonRateMax

        self.data = _sample

    def getWavelength(self):
        ''' get wavelength range '''
        return self.wavelength

#%%

if __name__ == '__main__':

    import napari

    sample = Sample2()
    sample.setSpectralDisk()
    # load multichannel image in one line
    viewer = napari.view_image(sample.get())
    napari.run()

