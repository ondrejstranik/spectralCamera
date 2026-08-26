"""
class FileSILCTFVideo

@author: ostranik
"""
#%%

import numpy as np
from pathlib import Path
import re
import logging
import tifffile

logger = logging.getLogger(__name__)

class FileSILCTFVideo:
    ''' class to load a time series of spectral images acquired with a tunable filter (LCTF),
    where every spectral frame is stored as a set of single-wavelength tiff files.

    file naming convention (example): imLCTFatWL500Frame12.tiff
        WL<number>    ... wavelength of the image
        Frame<number> ... index of the spectral frame (time point)

    all information (wavelength, frame index, acquisition time) is extracted
    from the tiff files present in the folder, acquisition time is taken
    from the file creation time '''

    DEFAULT = {'nameSet'  : {
                            'image' : '*.tif*',
                            'regexp': r'WL(?P<wl>\d+).*Frame(?P<frame>\d+)\.tif+$'}
    }

    def __init__(self,folder=None, **kwargs):
        ''' initialisation '''

        # data container
        self.folder = '' if folder is None else folder

    def setFolder(self,folder):
        self.folder = str(folder)

    def _groupFilesByFrame(self,folder=None):
        ''' scan the folder and group the tiff files by frame number,
        each group is a list of (wavelength,fileName) sorted by wavelength
        return dict frame:list[(wavelength,fileName)] '''
        if folder is not None: self.folder = folder

        vfolder = Path(self.folder)
        fileList = vfolder.glob(self.DEFAULT['nameSet']['image'])
        pattern = re.compile(self.DEFAULT['nameSet']['regexp'], re.IGNORECASE)

        frameDict = {}
        for f in fileList:
            m = pattern.search(f.name)
            if m is None:
                continue
            wl = int(m.group('wl'))
            frame = int(m.group('frame'))
            frameDict.setdefault(frame,[]).append((wl,f.name))

        for frame in frameDict:
            frameDict[frame].sort(key=lambda x: x[0])

        return frameDict

    def loadWavelength(self,folder=None):
        ''' loading wavelength, extracted from the file names present in the folder.
        wavelength 0 is a dark-current reference image, not an actual
        acquisition wavelength, and is excluded '''
        frameDict = self._groupFilesByFrame(folder)

        wlSet = {wl for group in frameDict.values() for wl,_ in group if wl != 0}
        wavelength = np.array(sorted(wlSet))
        return wavelength

    def loadImage(self,fileNameGroup, folder=None):
        ''' loading the spectral image from a group of single-wavelength tiff files
        fileNameGroup: list of fileNames sorted according to the wavelength.
        the file at wavelength 0 is a dark-current image: it is subtracted
        from every other wavelength of the frame and excluded from the
        returned image stack '''
        if folder is not None: self.folder = folder

        pattern = re.compile(self.DEFAULT['nameSet']['regexp'], re.IGNORECASE)

        try:
            darkImage = None
            images = []
            for fName in fileNameGroup:
                wl = int(pattern.search(fName).group('wl'))
                _image = tifffile.imread(str(Path(self.folder) / fName)).astype(float)
                if wl == 0:
                    darkImage = _image
                else:
                    images.append(_image)

            if darkImage is not None:
                images = [_image - darkImage for _image in images]

            sImage = np.zeros((len(images),*images[0].shape))
            for ii,_image in enumerate(images):
                sImage[ii,...] = _image

        except:
            logger.exception('error in class FileSILCTFVideo, function loadImage - could not load image')
            return

        return sImage

    def getImageInfo(self,folder=None):
        ''' getting list of spectral frames (each is a list of the single-wavelength
        fileNames sorted according to the wavelength) and the corresponding
        fileTime (creation time of the first file of the frame, in ns from epoch)
        return ( fileName:list[list], fileTime:np.array)'''
        frameDict = self._groupFilesByFrame(folder)

        frameIdx = sorted(frameDict.keys())
        fileName = [[name for _,name in frameDict[frame]] for frame in frameIdx]
        fileTime = np.array([int(1e9*(Path(self.folder) / group[0]).stat().st_ctime)
                              for group in fileName])

        return (fileName, fileTime)

    def loadAllImage(self,folder=None):
        ''' load all spectral frames of the video sequence
         return (allImage:np.array .... frame indexing = first index,
                wavelength: np.array
                time: np.array .. frame time)
        '''
        if folder is not None: self.folder= folder

        wavelength = self.loadWavelength()
        (fileName, fileTime) = self.getImageInfo()

        for ii,fGroup in enumerate(fileName):
            _image = self.loadImage(fGroup)
            if ii == 0:
                allImage = np.zeros((len(fileName),*_image.shape))
            allImage[ii,...] = _image

        return (allImage,wavelength,fileTime)


#%%
if __name__ == '__main__':
    pass
