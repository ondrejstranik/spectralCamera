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
    call is needed or expected.

    Each of the three filter images and the white reference is picked via
    its own file dialog; as soon as one is picked it is loaded and shown
    live in napari, and kept in memory (self.rawImages/self.whiteImage) so
    Calibrate transfers the already-loaded arrays straight into
    CalibrateFrom3Images instead of re-loading by name from a shared
    folder. '''

    DEFAULT = {'nameGUI': 'Calibration'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        self.myCal = None
        self.viewer = None

        # loaded image data, kept here so Calibrate can transfer it
        # directly instead of reloading it by name from a folder
        self.rawImages = [None, None, None]
        self.whiteImage = None

        # napari layers, kept so re-selecting a file updates the existing
        # layer in place instead of adding a duplicate one
        self._rawLayers = [None, None, None]
        self._whiteLayer = None

        CalibrationGUI.__setWidget(self)

    def __setWidget(self):
        ''' prepare the gui '''

        imageNameStack = CalibrateFrom3Images.DEFAULT['imageNameStack']
        wavelengthStack = CalibrateFrom3Images.DEFAULT['wavelengthStack']
        spectralRange = CalibrateFrom3Images.DEFAULT['spectralRange']

        dataFolder = Path(spectralCamera.dataFolder)

        @magicgui(call_button=False,
                  fileName1={"label": "image 1 file", "mode": 'r', "filter": "*.npy"},
                  wavelength1={"label": "image 1 wavelength [nm]"},
                  fileName2={"label": "image 2 file", "mode": 'r', "filter": "*.npy"},
                  wavelength2={"label": "image 2 wavelength [nm]"},
                  fileName3={"label": "image 3 file", "mode": 'r', "filter": "*.npy"},
                  wavelength3={"label": "image 3 wavelength [nm]"},
                  spectralRangeMin={"label": "spectral range min [nm]"},
                  spectralRangeMax={"label": "spectral range max [nm]"},
                  whiteFileName={"label": "white image file", "mode": 'r', "filter": "*.npy"})
        def settingsGui(fileName1=dataFolder / (imageNameStack[0] + '.npy'), wavelength1=wavelengthStack[0],
                         fileName2=dataFolder / (imageNameStack[1] + '.npy'), wavelength2=wavelengthStack[1],
                         fileName3=dataFolder / (imageNameStack[2] + '.npy'), wavelength3=wavelengthStack[2],
                         spectralRangeMin=spectralRange[0], spectralRangeMax=spectralRange[1],
                         whiteFileName=dataFolder / 'white_0.npy'):
            pass

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
        self.calibrateGui = calibrateGui
        self.saveGui = saveGui

        self.container = Container(widgets=[self.settingsGui,
                                             self.calibrateGui,
                                             self.saveGui],
                                    labels=False)

        self.vWindow.addParameterGui(self.container, name=self.DEFAULT['nameGUI'])

        # live-update: load+show a raw image every time its file picker
        # changes, and do the same once now for whatever the defaults
        # point at (harmless no-op print if a default file doesn't exist).
        # A wavelength change alone can also change which image is
        # shortest/middle/longest, so it must trigger a color/name refresh
        # too, even without a new file being picked.
        fileWidgets = [settingsGui.fileName1, settingsGui.fileName2, settingsGui.fileName3]
        wavelengthWidgets = [settingsGui.wavelength1, settingsGui.wavelength2, settingsGui.wavelength3]

        for index, fileWidget in enumerate(fileWidgets):
            fileWidget.changed.connect(lambda path, index=index: self._onRawImageSelected(index, path))
            self._onRawImageSelected(index, fileWidget.value)

        for wavelengthWidget in wavelengthWidgets:
            wavelengthWidget.changed.connect(lambda value: self._refreshRawLayerColorsAndNames())

        settingsGui.whiteFileName.changed.connect(self._onWhiteImageSelected)
        self._onWhiteImageSelected(settingsGui.whiteFileName.value)

    def _getViewer(self):
        ''' get or create the napari viewer used to show images/results '''
        if self.viewer is None:
            self.viewer = napari.Viewer()
        return self.viewer

    def _onRawImageSelected(self, index, path):
        ''' load and (re)display one of the three filter images as soon as
        it is picked, and keep the array for Calibrate to use directly. '''
        try:
            image = np.load(str(path))
        except (FileNotFoundError, ValueError):
            print(f'image not found or not readable: {path} (skipped)')
            return

        self.rawImages[index] = image

        viewer = self._getViewer()
        if self._rawLayers[index] is not None and self._rawLayers[index] in viewer.layers:
            self._rawLayers[index].data = image
        else:
            self._rawLayers[index] = viewer.add_image(image, blending='additive')

        self._refreshRawLayerColorsAndNames()

    def _refreshRawLayerColorsAndNames(self):
        ''' color and name of the three raw-image layers are based on the
        relative order of their wavelengths, not which file slot they're
        in - shortest wavelength shown in blue, middle in green, longest
        in red, each layer named after its own wavelength. Called whenever
        a wavelength value changes or a new image is picked, since either
        can change the shortest/middle/longest ordering. '''
        s = self.settingsGui
        wavelengths = [s.wavelength1.value, s.wavelength2.value, s.wavelength3.value]

        # rank 0 (shortest wavelength) -> blue, 1 (middle) -> green, 2 (longest) -> red
        colorByRank = ['blue', 'green', 'red']
        order = sorted(range(3), key=lambda i: wavelengths[i])
        colorForIndex = {index: colorByRank[rank] for rank, index in enumerate(order)}

        viewer = self._getViewer()
        for index, layer in enumerate(self._rawLayers):
            if layer is not None and layer in viewer.layers:
                layer.colormap = colorForIndex[index]
                layer.name = f'{wavelengths[index]} nm'

    def _onWhiteImageSelected(self, path):
        ''' load and (re)display the white reference image as soon as it
        is picked, and keep the array for Calibrate to use directly. '''
        if not path:
            self.whiteImage = None
            return

        try:
            image = np.load(str(path))
        except (FileNotFoundError, ValueError):
            print(f'white image not found or not readable: {path} (skipped)')
            self.whiteImage = None
            return

        self.whiteImage = image

        viewer = self._getViewer()
        if self._whiteLayer is not None and self._whiteLayer in viewer.layers:
            self._whiteLayer.data = image
        else:
            self._whiteLayer = viewer.add_image(image, name='white',
                                                 colormap='gray', blending='additive')

    def _calibrate(self):
        ''' start the calibration in a background worker thread, so the
        heavy computation (grid fitting, curve_fit, warp matrices) doesn't
        block the GUI's event loop. Status/results are only ever touched
        from _onCalibrateStarted/_onCalibrateFinished/_onCalibrateError,
        which run back on the GUI thread via the worker's Qt signals -
        CalibrationGUI is a QObject (via BaseGUI), so Qt safely queues
        those calls across threads instead of running them on the worker
        thread, which would not be safe for GUI/napari widgets.

        The three raw images and the white image are transferred as
        already-loaded arrays (see self.rawImages/self.whiteImage) rather
        than reloaded by name/folder - they were loaded once already, as
        soon as each was picked. '''
        if any(image is None for image in self.rawImages):
            print('select all three calibration images before calibrating')
            return

        s = self.settingsGui
        wavelengthStack = [s.wavelength1.value, s.wavelength2.value, s.wavelength3.value]
        spectralRange = [s.spectralRangeMin.value, s.spectralRangeMax.value]

        self.calibrateGui.status.value = 'calibrating...'
        self.calibrateGui.call_button.enabled = False

        worker = create_worker(self._runCalibration,
                                list(self.rawImages), wavelengthStack, spectralRange, self.whiteImage,
                                _start_thread=True,
                                _connect={'started': self._onCalibrateStarted,
                                          'returned': self._onCalibrateFinished,
                                          'errored': self._onCalibrateError})

        # keep a reference so the worker/thread isn't garbage-collected mid-run
        self._calibrateWorker = worker

    def _runCalibration(self, imageStack, wavelengthStack, spectralRange, whiteImage):
        ''' pure computation, runs on the worker thread - must not touch
        any GUI or napari widgets directly. imageStack/whiteImage are
        already-loaded arrays transferred straight from the GUI, so no
        file access happens here at all. '''
        myCal = CalibrateFrom3Images(wavelengthStack=wavelengthStack)
        myCal.setImageStack(imageStack=imageStack, wavelengthStack=wavelengthStack)
        myCal.prepareGrid(spectralRange)
        myCal.setWarpMatrix(spectral=True, subpixel=True)

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
