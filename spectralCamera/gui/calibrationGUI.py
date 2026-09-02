'''
GUI to generate a spectral calibration object from three narrow-band images
'''
#%%
from pathlib import Path
import numpy as np
import napari

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui
from magicgui.widgets import Container
from superqt.utils._qthreading import create_worker

import spectralCamera
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images


class CalibrationGUI(BaseGUI):
    ''' GUI to set the parameters of and run a 3-image spectral calibration
    (see spectralCamera.algorithm.calibrateFrom3Images.CalibrateFrom3Images).
    File-based tool, not tied to any live camera device - no setDevice()
    call is needed or expected. '''

    DEFAULT = {'nameGUI': 'Calibration'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        self.myCal = None
        self.viewer = None

        CalibrationGUI.__setWidget(self)

    def __setWidget(self):
        ''' prepare the gui '''

        imageNameStack = CalibrateFrom3Images.DEFAULT['imageNameStack']
        wavelengthStack = CalibrateFrom3Images.DEFAULT['wavelengthStack']
        spectralRange = CalibrateFrom3Images.DEFAULT['spectralRange']

        dataFolder = Path(spectralCamera.dataFolder)

        @magicgui(call_button=False,
                  folder={"label": "calibration folder", "mode": 'd'},
                  fileName1={"label": "image 1 file (red)", "mode": 'r', "filter": "*.npy"},
                  wavelength1={"label": "image 1 wavelength [nm]"},
                  fileName2={"label": "image 2 file (green)", "mode": 'r', "filter": "*.npy"},
                  wavelength2={"label": "image 2 wavelength [nm]"},
                  fileName3={"label": "image 3 file (blue)", "mode": 'r', "filter": "*.npy"},
                  wavelength3={"label": "image 3 wavelength [nm]"},
                  spectralRangeMin={"label": "spectral range min [nm]"},
                  spectralRangeMax={"label": "spectral range max [nm]"},
                  whiteFileName={"label": "white image file", "mode": 'r', "filter": "*.npy"})
        def settingsGui(folder=dataFolder,
                         fileName1=dataFolder / (imageNameStack[0] + '.npy'), wavelength1=wavelengthStack[0],
                         fileName2=dataFolder / (imageNameStack[1] + '.npy'), wavelength2=wavelengthStack[1],
                         fileName3=dataFolder / (imageNameStack[2] + '.npy'), wavelength3=wavelengthStack[2],
                         spectralRangeMin=spectralRange[0], spectralRangeMax=spectralRange[1],
                         whiteFileName=dataFolder / 'white_0.npy'):
            pass

        @magicgui(call_button='Show images')
        def showImagesGui():
            self._showImages()

        @magicgui(call_button='Calibrate',
                  status={"widget_type": "Label"},
                  bwidth={"widget_type": "Label"},
                  bheight={"widget_type": "Label"},
                  wavelengthRange={"widget_type": "Label"})
        def calibrateGui(status='', bwidth='', bheight='', wavelengthRange=''):
            self._calibrate()

        @magicgui(call_button='Save',
                  saveFolder={"label": "save folder", "mode": 'd'})
        def saveGui(saveFolder=Path(spectralCamera.dataFolder)):
            self._save(saveFolder)

        self.settingsGui = settingsGui
        self.showImagesGui = showImagesGui
        self.calibrateGui = calibrateGui
        self.saveGui = saveGui

        self.container = Container(widgets=[self.settingsGui,
                                             self.showImagesGui,
                                             self.calibrateGui,
                                             self.saveGui],
                                    labels=False)

        self.vWindow.addParameterGui(self.container, name=self.DEFAULT['nameGUI'])

    def _getSettings(self):
        ''' read the current values from the settings panel.

        fileName1/2/3/whiteFileName are file pickers (full paths) for
        convenience, but CalibrateFrom3Images.setImageStack() (and the
        loads in this class) expect bare names without extension, joined
        with the separate calibration folder - so only the file stem is
        used here; the picked file's own directory is ignored and it must
        actually live in the calibration folder for loading to work. '''
        s = self.settingsGui
        folder = str(s.folder.value)
        imageNameStack = [Path(s.fileName1.value).stem, Path(s.fileName2.value).stem, Path(s.fileName3.value).stem]
        wavelengthStack = [s.wavelength1.value, s.wavelength2.value, s.wavelength3.value]
        spectralRange = [s.spectralRangeMin.value, s.spectralRangeMax.value]
        whiteFileName = Path(s.whiteFileName.value).stem if s.whiteFileName.value else ''
        return folder, imageNameStack, wavelengthStack, spectralRange, whiteFileName

    def _getViewer(self):
        ''' get or create the napari viewer used to show images/results '''
        if self.viewer is None:
            self.viewer = napari.Viewer()
        return self.viewer

    def _showImages(self):
        ''' show the raw calibration images in napari, each of the three
        filter images in its own color (R/G/B) with additive blending so
        they can be visually compared/overlaid, plus the white reference
        image (if available) in grayscale. '''
        folder, imageNameStack, wavelengthStack, _, whiteFileName = self._getSettings()

        viewer = self._getViewer()

        colormaps = ['red', 'green', 'blue']
        for name, wavelength, colormap in zip(imageNameStack, wavelengthStack, colormaps):
            try:
                image = np.load(folder + '/' + name + '.npy')
            except FileNotFoundError:
                print(f'image not found: {folder}/{name}.npy (skipped)')
                continue
            viewer.add_image(image, name=f'{name} ({wavelength} nm)',
                              colormap=colormap, blending='additive')

        if whiteFileName:
            try:
                whiteImage = np.load(folder + '/' + whiteFileName + '.npy')
                viewer.add_image(whiteImage, name=f'{whiteFileName} (white)',
                                  colormap='gray', blending='additive')
            except FileNotFoundError:
                print(f'white image not found: {folder}/{whiteFileName}.npy (skipped)')

        print(f'showing raw calibration images from folder: {folder}')

    def _calibrate(self):
        ''' start the calibration in a background worker thread, so the
        heavy computation (grid fitting, curve_fit, warp matrices) doesn't
        block the GUI's event loop. Status/results are only ever touched
        from _onCalibrateStarted/_onCalibrateFinished/_onCalibrateError,
        which run back on the GUI thread via the worker's Qt signals -
        CalibrationGUI is a QObject (via BaseGUI), so Qt safely queues
        those calls across threads instead of running them on the worker
        thread, which would not be safe for GUI/napari widgets. '''
        folder, imageNameStack, wavelengthStack, spectralRange, whiteFileName = self._getSettings()

        self.calibrateGui.status.value = 'calibrating...'
        self.calibrateGui.call_button.enabled = False

        worker = create_worker(self._runCalibration,
                                folder, imageNameStack, wavelengthStack, spectralRange, whiteFileName,
                                _start_thread=True,
                                _connect={'started': self._onCalibrateStarted,
                                          'returned': self._onCalibrateFinished,
                                          'errored': self._onCalibrateError})

        # keep a reference so the worker/thread isn't garbage-collected mid-run
        self._calibrateWorker = worker

    def _runCalibration(self, folder, imageNameStack, wavelengthStack, spectralRange, whiteFileName):
        ''' pure computation, runs on the worker thread - must not touch
        any GUI or napari widgets directly. Also loads the white image
        (if given) here, since that's just file I/O and safe off-thread.

        Passes folder explicitly to setImageStack() rather than mutating
        the shared spectralCamera.dataFolder global - this runs on a
        worker thread, and writing a module-level global from there would
        race against any other code (e.g. another button's callback) that
        reads/writes it on the GUI thread at the same time. '''
        myCal = CalibrateFrom3Images(imageNameStack=imageNameStack,
                                      wavelengthStack=wavelengthStack)
        myCal.setImageStack(folder=folder)
        myCal.prepareGrid(spectralRange)
        myCal.setWarpMatrix(spectral=True, subpixel=True)

        whiteImage = None
        if whiteFileName:
            try:
                whiteImage = np.load(folder + '/' + whiteFileName + '.npy')
            except FileNotFoundError:
                print(f'white image not found: {folder}/{whiteFileName}.npy (skipped)')

        return myCal, whiteImage

    def _onCalibrateStarted(self):
        ''' runs on the GUI thread when the worker actually starts '''
        print('calibrating...')

    def _onCalibrateFinished(self, result):
        ''' runs on the GUI thread with the worker's return value '''
        myCal, whiteImage = result
        self.myCal = myCal

        self.calibrateGui.bwidth.value = str(myCal.bwidth)
        self.calibrateGui.bheight.value = str(myCal.bheight)
        self.calibrateGui.wavelengthRange.value = f'{myCal.wavelength.min():.1f} - {myCal.wavelength.max():.1f} nm'
        self.calibrateGui.status.value = 'finished'
        self.calibrateGui.call_button.enabled = True

        # visual check: calibration image with the fitted spectral grid
        # overlaid, same as the manual visual check in
        # utility/generateCalibrationObject.py - the warped white image is
        # used as the "calibration image" when available (a flat-field
        # reference makes the grid alignment much easier to judge by eye
        # than the raw filter-sum image), falling back to the unwarped
        # first calibration image otherwise.
        if whiteImage is not None:
            calibrationImage = myCal.getWarpedImage(whiteImage)
        else:
            calibrationImage = myCal.imageStack[0]

        viewer = self._getViewer()
        blockImage = myCal.getSpectralBlockImage() * 1
        viewer.add_image(calibrationImage, name='calibration image')
        viewer.add_image(blockImage, name='fitted spectral grid', opacity=0.3)

        print('calibration finished')

    def _onCalibrateError(self, exc):
        ''' runs on the GUI thread if _runCalibration raised '''
        self.calibrateGui.status.value = f'error: {exc}'
        self.calibrateGui.call_button.enabled = True
        print(f'calibration failed: {exc}')

    def _save(self, saveFolder):
        ''' save the calibration object to file '''
        if self.myCal is None:
            print('run Calibrate before Save')
            return

        self.myCal.saveClass(classFolder=str(saveFolder))
        print(f'calibration saved in folder: {saveFolder}')


if __name__ == "__main__":
    pass

#%%
