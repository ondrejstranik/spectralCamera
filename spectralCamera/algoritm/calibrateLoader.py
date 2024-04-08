'''
class to load calibration objects
'''
#%%

from pathlib import Path
import pickle

class CalibrateLoader():
    ''' main class to load calibration objects (for spectral imaging) '''

    @classmethod
    def load(cls,classFile=None):
        ''' load the class itself from file '''
        # import all types of calibration objects
        from HSIplasmon.algorithm.calibrateRamanImages import CalibrateRamanImages
        from HSIplasmon.algorithm.calibrateFrom3Images import CalibrateFrom3Images
        from HSIplasmon.algorithm.calibrateRGBImages import CalibrateRGBImage

        # it is necessary in order to unpickle not only from __main__
        # alternatively use a package dill
        import __main__
        __main__.CalibrateRamanImages = CalibrateRamanImages
        __main__.CalibrateFrom3Images = CalibrateFrom3Images
        # not loading this class from file , it is only initiated
        #__main__.CalibrateRGBImage = CalibrateRGBImage

        fullFile = str(classFile)

        path= Path(fullFile)
        if path.is_file():
            myObject = pickle.load(open(fullFile, 'rb'))
        else:
            if fullFile in ['W', 'RGGB','RGB']:
                myObject = CalibrateRGBImage(fullFile)
            else:
                myObject = CalibrateRGBImage('W') #  default black/white

        return myObject

if __name__ == "__main__":

    print(f'loading default calibration class')
    cal = CalibrateLoader.load()
    print(f'class type: {type(cal).__name__}')

    from HSIplasmon.algorithm.calibrateRamanImages import CalibrateRamanImages
    import HSIplasmon as hsi
    print(f'loading CalibrateRamanImges')
    cal = CalibrateLoader.load(hsi.dataFolder + '\\' + CalibrateRamanImages.DEFAULT['classFile'])
    print(f'class type: {type(cal).__name__}')




    
