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
    ''' class to define a sample object of the microscope - spectrally resolved'''
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

    def setSpectralCell(self,samplePixelSize=None,
                        sampleSize= None,
                        photonRateMax= None,
                        samplePosition = None,
                        cellSize = None,
                        cellDistance = None,
                        wavelength = None):
        ''' define the sample.
        sample ... spectral distribution of photon rates [#/s/pixelSize^2] (no noise)
        cellSize ... (mean) cell radius [um]
        cellDistance ... average distance between neighbouring cells [um]
        the cytoplasm, nucleus and protein puncta are spectrally coloured by the
        emission spectrum (gaussian peak, given by its position and width) of a
        fluorophore commonly used to label the corresponding cell part:
        cytoplasm ... EGFP, nucleus ... DAPI, protein ... mCherry'''

        PROTEIN_NUMBER = 8 # number of protein puncta per cell
        PROTEIN_SIZE = 0.1 # protein radius [um] (100 nm)

        # fluorophore emission peak position and standard deviation [nm]
        CYTOPLASM_FLUOROPHORE = (509,17) # EGFP, generic cytoplasmic label
        NUCLEUS_FLUOROPHORE = (461,21) # DAPI, nuclear stain
        PROTEIN_FLUOROPHORE = (610,23) # mCherry, protein tag

        DEFAULT = {'photonRateMax':1e6,
                    'samplePixelSize':0.1, # um
                    'sampleSize': (1000,1000),
                    'samplePosition': np.array([0,0,0]), # pixels
                    'cellSize': 5, # radius of a cell [um]
                    'cellDistance': 11, # average distance between cells [um]
                    'wavelength': np.arange(400,800,10)} # nm

        self.pixelSize=DEFAULT['samplePixelSize'] if samplePixelSize is None else samplePixelSize
        self.size=DEFAULT['sampleSize'] if sampleSize is None else sampleSize
        self.position=DEFAULT['samplePosition'] if samplePosition is None else samplePosition
        self.wavelength = DEFAULT['wavelength'] if wavelength is None else wavelength

        photonRateMax=DEFAULT['photonRateMax'] if photonRateMax is None else photonRateMax
        cellSize=DEFAULT['cellSize'] if cellSize is None else cellSize
        cellDistance=DEFAULT['cellDistance'] if cellDistance is None else cellDistance

        # cell radius, average spacing and protein size, converted from [um] to [pixel]
        cellRadiusPixel = cellSize/self.pixelSize
        cellDistancePixel = cellDistance/self.pixelSize
        proteinRadiusPixel = PROTEIN_SIZE/self.pixelSize

        # number of cells expected for the given average spacing over the sample area
        cellNumber = max(1, int(round((self.size[0]*self.size[1])/cellDistancePixel**2)))

        # normalised gaussian emission spectrum of the fluorophore labelling each cell part
        cytoplasmSpectrum = np.exp(-(self.wavelength-CYTOPLASM_FLUOROPHORE[0])**2/2/CYTOPLASM_FLUOROPHORE[1]**2)
        nucleusSpectrum = np.exp(-(self.wavelength-NUCLEUS_FLUOROPHORE[0])**2/2/NUCLEUS_FLUOROPHORE[1]**2)
        proteinSpectrum = np.exp(-(self.wavelength-PROTEIN_FLUOROPHORE[0])**2/2/PROTEIN_FLUOROPHORE[1]**2)

        # define, cells as randomly placed, non-overlapping, randomly sized and
        # irregularly shaped blobs, each with a brighter nucleus and a few small proteins
        _sample = np.zeros((self.wavelength.shape[0],*self.size))

        rng = np.random.default_rng()
        placedCenter = []
        placedRadius = []
        maxAttempt = 100
        for _ in range(cellNumber):
            for _ in range(maxAttempt):
                center = np.array([rng.uniform(0,self.size[0]), rng.uniform(0,self.size[1])])
                radius = rng.uniform(0.7,1.3)*cellRadiusPixel
                rr,cc,boundingRadius = self._irregularBlob(center,radius,rng)

                fitsInImage = (center[0]-boundingRadius>=0 and center[1]-boundingRadius>=0
                                and center[0]+boundingRadius<=self.size[0]-1
                                and center[1]+boundingRadius<=self.size[1]-1)

                noOverlap = all(np.linalg.norm(center-otherCenter) >= max(boundingRadius+otherRadius,cellDistancePixel)
                                for otherCenter,otherRadius in zip(placedCenter,placedRadius))

                if fitsInImage and noOverlap:
                    break
            else:
                # could not find a spot that fits fully in the image and is far
                # enough from the other cells, skip this cell
                continue

            placedCenter.append(center)
            placedRadius.append(boundingRadius)

            # cytoplasm, an irregular (non-circular) blob
            _sample[:,rr,cc] += rng.uniform(0.2,0.4)*cytoplasmSpectrum[:,None]

            # nucleus, slightly off-centre, also irregularly shaped
            nucleusRadius = radius*rng.uniform(0.35,0.5)
            nucleusCenter = center+rng.uniform(-0.15,0.15,2)*radius
            rrN,ccN,_ = self._irregularBlob(nucleusCenter,nucleusRadius,rng,amplitude=0.1)
            _sample[:,rrN,ccN] += rng.uniform(0.5,0.8)*nucleusSpectrum[:,None]

            # small proteins scattered through the cytoplasm, placed on pixels that
            # are actually part of the cell so none end up outside its boundary
            if len(rr)>0:
                proteinIndex = rng.integers(0,len(rr),size=PROTEIN_NUMBER)
                for idx in proteinIndex:
                    proteinCenter = (rr[idx],cc[idx])
                    rrP,ccP = disk(proteinCenter, proteinRadiusPixel, shape=self.size)
                    _sample[:,rrP,ccP] += rng.uniform(0.3,0.6)*proteinSpectrum[:,None]

        # normalise
        if np.max(_sample)>0:
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

