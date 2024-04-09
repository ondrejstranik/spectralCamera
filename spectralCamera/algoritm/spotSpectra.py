'''
class for calculating spot spectra from 3D spectral cube
'''
import sys, os
sys.path.append(os.getcwd() + '\\code')


import numpy as np


class SpotSpectra:
    ''' class for calculating spot spectra '''

    def __init__(self,wxyImage,spotPosition=[]):
        ''' initialization of the parameters '''

        # parameters of the mask
        self.pxBcg = 3
        self.pxAve = 3
        self.pxSpace = 1

        self.wxyImage = wxyImage  # spectral Image
        self.maskSize = None # total size of the mask
        self.maskSpot = None # weights for calculation of spots spectra
        self.maskBcg = None # weight for calculation of background spectra
        self.maskImage = None
        self.spotPosition = spotPosition

        self.spectraRawSpot = []
        self.spectraRawBcg = []
        self.spectraSpot = []

        self.setMask()
        
        self.calculateSpectra()

    def setMask(self,pxAve=None,pxBcg= None, pxSpace = None):
        ''' set the geometry of spots and bcg mask  and calculate spectra'''

        if pxAve is not None:
            self.pxAve = pxAve
        if pxBcg is not None:
            self.pxBcg = pxBcg
        if pxSpace is not None:
            self.pxSpace = pxSpace
        self.maskSize = 2*(self.pxBcg + self.pxAve + self.pxSpace) + 1 

        xx, yy = np.meshgrid(np.arange(self.maskSize) - self.maskSize//2, (np.arange(self.maskSize) - self.maskSize//2))
        maskR = np.sqrt(xx**2 + yy**2)

        self.maskSpot = maskR<self.pxAve
        self.maskBcg = (maskR>(self.pxAve+self.pxSpace)) & (maskR<self.pxAve+self.pxSpace + self.pxBcg)
        
        # set mask image
        self.maskImage = 0*self.wxyImage[0,:,:]
        for myspot in self.spotPosition:
            self.maskImage[int(myspot[0])-self.maskSize//2:int(myspot[0])+self.maskSize//2+1,
                            int(myspot[1])-self.maskSize//2:int(myspot[1])+self.maskSize//2+1] = \
                            self.maskSpot*2
            self.maskImage[int(myspot[0])-self.maskSize//2:int(myspot[0])+self.maskSize//2+1,
                            int(myspot[1])-self.maskSize//2:int(myspot[1])+self.maskSize//2+1] += \
                            self.maskBcg*1

        self.calculateSpectra()

    def setSpot(self, spotPosition):
        ''' set position of the spots  and calculate spectra'''
        self.spotPosition = spotPosition

        self.setMask()
        self.calculateSpectra()

    def setImage(self, wxyImage):
        ''' set the spectra image and calculate image and calculate spectra'''
        self.wxyImage = wxyImage

        self.calculateSpectra()

    def calculateSpectra(self):
        ''' calculate the spectra '''

        self.spectraRawSpot = []
        self.spectraRawBcg = []
        self.spectraSpot = []

        maskSpotFlatten = self.maskSpot.flatten()
        maskBcgFlatten = self.maskBcg.flatten()

        for myspot in self.spotPosition:
            # image of the single spots with surrounding 
            myAreaImage = self.wxyImage[:,int(myspot[0])-self.maskSize//2:int(myspot[0])+self.maskSize//2+1,
                            int(myspot[1])-self.maskSize//2:int(myspot[1])+self.maskSize//2+1]
            
            myAreaImageFlatten = myAreaImage.reshape(myAreaImage.shape[0],-1)

            spectraRawSpot = np.mean(myAreaImageFlatten[:,maskSpotFlatten], axis=1)
            spectraRawBcg = np.mean(myAreaImageFlatten[:,maskBcgFlatten], axis=1)
            spectraSpot = spectraRawSpot/spectraRawBcg

            self.spectraRawSpot.append(spectraRawSpot)        
            self.spectraRawBcg.append(spectraRawBcg)        
            self.spectraSpot.append(spectraSpot)        

    def getMask(self):
        ''' return the image of the mask of spots and backgound '''
        return self.maskImage

    def getT(self):
        ''' return trasmission spectra of the spots '''
        return self.spectraSpot

    def getA(self):
        ''' return absorption spectra of the spots '''
        return (1 - np.array(self.spectraSpot)).tolist()




if __name__ == "__main__":

        # load the image
        container = np.load(os.getcwd() + '\\code\\Data\\plasmonicArray.npz')
        wxyImage = container['arr_0']
        w = container['arr_1']
        mySpot = [[118,113], [151,108]]

        mySS = SpotSpectra(wxyImage,spotPosition=mySpot)

        # show images
        import napari
        viewer = napari.Viewer()
        viewer.add_image(np.sum(wxyImage, axis=0))
        viewer.add_image(mySS.maskImage)
        napari.run()


        # show spectra
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(w, np.array(mySS.spectraSpot).T)
        ax.set_title('Spectra')

        fig, ax = plt.subplots()
        ax.plot(w, np.array(mySS.spectraRawSpot).T)
        ax.plot(w, np.array(mySS.spectraRawBcg).T)

        ax.set_title('Raw Spectra')




        plt.show()















