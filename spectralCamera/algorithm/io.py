'''
helper class to read raman images
'''

from dataclasses import dataclass
import tifffile
import numpy

@dataclass
class RamanImage():
    fFolder: str
    fSignalName: str ='a{}.tif'
    fBcgName: str='dark{}.tif'
    fNumbers = [1,2,3,4,5,6]
        
    def getImage(self):
        for ii in self.fNumbers:
            _myIm = (tifffile.imread(self.fFolder + '/' + self.fSignalName.format(ii)).astype(int) - 
                    tifffile.imread(self.fFolder + '/' + self.fBcgName.format(ii)).astype(int))
            if ii==1:
                myIm = _myIm
            else:
                myIm += _myIm        
        return myIm

if __name__ == "__main__":

    import napari

    fFolder = r'G:\office\work\projects - free\23-07-07 Integral field uscope - Mariia\23-09-26 Mariia data\DATA\calibration'

    myRI = RamanImage(fFolder)

    viewer = napari.Viewer()
    viewer.add_image(myRI.getImage())
    napari.run()

