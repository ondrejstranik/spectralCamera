'''
GUI to generate a spectral calibration object from three narrow-band images
'''
#%%
from pathlib import Path
import numpy as np
import napari
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from viscope.gui.baseGUI import BaseGUI
from magicgui import magicgui
from magicgui.widgets import Container
from superqt.utils._qthreading import create_worker

import spectralCamera
from spectralCamera.algorithm.calibrateFrom3Images import CalibrateFrom3Images
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer


def _isEmptyPath(path):
    ''' True for an unset file-picker value. A magicgui FileEdit's empty
    state is Path(''), but pathlib normalises that to Path('.') - which
    is truthy and stringifies to '.', not '' - so a plain "not path" or
    "path == ''" check does not catch it. '''
    return (not path) or (str(path) in ('', '.'))


class CalibrationGUI(BaseGUI):
    ''' GUI to set the parameters of and run a 3-image spectral calibration
    (see spectralCamera.algorithm.calibrateFrom3Images.CalibrateFrom3Images).
    File-based tool, not tied to any live camera device - no setDevice()
    call is needed or expected.

    Mirrors the step-by-step workflow of utility/generateCalibrationObject.py:
        1) pick the three filter images + white reference (each shows up
           live in napari as soon as picked)
        2) "Get Blocks" - fit the super-pixel grid to the calibration
           images (CalibrateFrom3Images.prepareGrid) and show the visual
           checks: pixel-to-wavelength fit plot, the fitted spectral block
           grid over the images, and the located spectral peaks/zero
           points. Pressing it again redoes the whole thing from scratch.
        3) "Calculate Warp" - fit the warping matrices
           (CalibrateFrom3Images.setWarpMatrix) and show the same set of
           checks as the end of generateCalibrationObject.py: the warp
           matrices themselves, raw vs warped images, corrected vs
           uncorrected spectral images, their 2D histograms, and the
           resulting hyperspectral cube.
        4) "Save" - store the calibration object.

    Nothing is read from disk until the user either picks a file or
    presses "Get Blocks" - no image is loaded just because a file picker
    happens to show a default path. '''

    DEFAULT = {'nameGUI': 'Calibration'}

    def __init__(self, viscope, **kwargs):
        ''' initialise the class '''
        super().__init__(viscope, **kwargs)

        self.myCal = None
        self.viewer = None

        # extra napari viewers opened by Calculate Warp, closed and
        # re-created every time the button is pressed again
        self._warpViewers = []

        # loaded image data, kept here so Get Blocks can transfer it
        # directly instead of reloading it by name from a folder
        self.rawImages = [None, None, None]
        self.whiteImage = None

        # napari layers, kept so re-selecting a file / re-running Get
        # Blocks updates the existing layer in place instead of adding a
        # duplicate one
        self._rawLayers = [None, None, None]
        self._whiteLayer = None
        self._blockLayer = None
        self._peakLayer = None
        self._zeroLayer = None

        # matplotlib "pixel to wavelength" check figure, reused in place
        self._checkFig = None
        self._checkAx = None
        self._checkHistAx = None

        CalibrationGUI.__setWidget(self)

    def __setWidget(self):
        ''' prepare the gui '''

        wavelengthStack = CalibrateFrom3Images.DEFAULT['wavelengthStack']
        spectralRange = CalibrateFrom3Images.DEFAULT['spectralRange']

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
        def settingsGui(fileName1=Path(''), wavelength1=wavelengthStack[0],
                         fileName2=Path(''), wavelength2=wavelengthStack[1],
                         fileName3=Path(''), wavelength3=wavelengthStack[2],
                         spectralRangeMin=spectralRange[0], spectralRangeMax=spectralRange[1],
                         whiteFileName=Path('')):
            pass

        @magicgui(call_button='Get Blocks',
                  status={"widget_type": "Label"},
                  bwidth={"widget_type": "Label", "label": "box width [px]"},
                  bheight={"widget_type": "Label", "label": "box height [px]"},
                  wavelengthRange={"widget_type": "Label"},
                  dispersion={"widget_type": "Label", "label": "average dispersion"},
                  blockDistance={"widget_type": "Label", "label": "average block distance [px]"})
        def getBlocksGui(status='', bwidth='', bheight='', wavelengthRange='', dispersion='', blockDistance=''):
            self._getBlocks()

        @magicgui(call_button='Calculate Warp',
                  status={"widget_type": "Label"})
        def warpGui(status=''):
            self._calculateWarp()

        @magicgui(call_button='Save',
                  saveFolder={"label": "save folder", "mode": 'd'})
        def saveGui(saveFolder=Path(spectralCamera.dataFolder)):
            self._save(saveFolder)

        self.settingsGui = settingsGui
        self.getBlocksGui = getBlocksGui
        self.warpGui = warpGui
        self.saveGui = saveGui

        self.container = Container(widgets=[self.settingsGui,
                                             self.getBlocksGui,
                                             self.warpGui,
                                             self.saveGui],
                                    labels=False)

        self.vWindow.addParameterGui(self.container, name=self.DEFAULT['nameGUI'])

        # calculating the warp before there are any blocks makes no sense
        self.warpGui.call_button.enabled = False

        # live-update: load+show a raw image every time its file picker is
        # actually changed by the user - nothing is loaded just because a
        # default path happens to be pre-filled at start-up. A wavelength
        # change alone can also change which image is shortest/middle/
        # longest, so it must trigger a color/name refresh too, even
        # without a new file being picked.
        fileWidgets = [settingsGui.fileName1, settingsGui.fileName2, settingsGui.fileName3]
        wavelengthWidgets = [settingsGui.wavelength1, settingsGui.wavelength2, settingsGui.wavelength3]

        for index, fileWidget in enumerate(fileWidgets):
            fileWidget.changed.connect(lambda path, index=index: self._onRawImageSelected(index, path))

        for wavelengthWidget in wavelengthWidgets:
            wavelengthWidget.changed.connect(lambda value: self._refreshRawLayerColorsAndNames())

        settingsGui.whiteFileName.changed.connect(self._onWhiteImageSelected)

        # open the napari window right away rather than waiting for the
        # first image/result to be shown in it
        self._getViewer()

    def _getViewer(self):
        ''' get or create the napari viewer used to show images/results '''
        if self.viewer is None:
            self.viewer = napari.Viewer()
        return self.viewer

    def _onRawImageSelected(self, index, path):
        ''' load and (re)display one of the three filter images as soon as
        it is picked, and keep the array for Get Blocks to use directly. '''
        if _isEmptyPath(path):
            self.rawImages[index] = None
            return

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
        is picked, and keep the array for Get Blocks to use directly. '''
        if _isEmptyPath(path):
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

    # ------------------------------------------------------------------
    # Get Blocks - fit the super-pixel grid (CalibrateFrom3Images.prepareGrid)
    # ------------------------------------------------------------------

    def _getBlocks(self):
        ''' fit the spectral super-pixel grid and show the visual checks,
        in a background worker thread so the heavy computation (peak
        finding, grid fitting) doesn't block the GUI's event loop.
        Re-running it (button pressed again) redoes everything from
        scratch - a fresh CalibrateFrom3Images object is built and the
        plot/layers are updated in place.

        Any of the three filter images or the white image that hasn't
        been loaded yet (file picker never touched by the user) is loaded
        here, from whatever path is currently shown in its file picker -
        this is the first point anything gets read from disk by default. '''
        s = self.settingsGui
        fileWidgets = [s.fileName1, s.fileName2, s.fileName3]
        for index, fileWidget in enumerate(fileWidgets):
            if self.rawImages[index] is None:
                self._onRawImageSelected(index, fileWidget.value)
        if self.whiteImage is None:
            self._onWhiteImageSelected(s.whiteFileName.value)

        if any(image is None for image in self.rawImages):
            print('select all three calibration images before getting blocks')
            return

        wavelengthStack = [s.wavelength1.value, s.wavelength2.value, s.wavelength3.value]
        spectralRange = [s.spectralRangeMin.value, s.spectralRangeMax.value]

        # CalibrateFrom3Images treats the first image/wavelength as the
        # center reference (zero pixel shift) that the other two are
        # measured relative to - so whichever of the three wavelengths is
        # the middle one has to be moved to slot 0, regardless of which
        # file picker it was actually entered into
        order = sorted(range(3), key=lambda i: wavelengthStack[i])
        centerIndex = order[1]
        reorderedIndex = [centerIndex] + [i for i in range(3) if i != centerIndex]

        imageStack = [self.rawImages[i] for i in reorderedIndex]
        wavelengthStack = [wavelengthStack[i] for i in reorderedIndex]

        self.getBlocksGui.status.value = 'getting blocks...'
        self.getBlocksGui.call_button.enabled = False
        self.warpGui.call_button.enabled = False

        worker = create_worker(self._runGetBlocks,
                                imageStack, wavelengthStack, spectralRange,
                                _start_thread=True,
                                _connect={'started': self._onGetBlocksStarted,
                                          'returned': self._onGetBlocksFinished,
                                          'errored': self._onGetBlocksError})

        # keep a reference so the worker/thread isn't garbage-collected mid-run
        self._getBlocksWorker = worker

    def _runGetBlocks(self, imageStack, wavelengthStack, spectralRange):
        ''' pure computation, runs on the worker thread - must not touch
        any GUI, matplotlib or napari widgets directly. '''
        myCal = CalibrateFrom3Images(wavelengthStack=wavelengthStack)
        myCal.setImageStack(imageStack=imageStack, wavelengthStack=wavelengthStack)
        myCal.prepareGrid(spectralRange)

        blockImage = myCal.getSpectralBlockImage() * 1

        # located spectral peaks - every spot found in any of the three
        # calibration images, marked white on a raster image rather than
        # as individual point markers (there are far too many spots for
        # that to render/stay readable)
        peakMask = np.zeros_like(myCal.imageStack[0])
        allPeakPositions = np.vstack([imMo.position for imMo in myCal.imMoStack])
        peakMask[allPeakPositions[:, 0].astype(int), allPeakPositions[:, 1].astype(int)] = 1

        # grid [0,0] / zero position of each of the three calibration images
        point00 = np.array([imMo.xy00 for imMo in myCal.imMoStack])

        # per-spot relative x-position (dispersion direction only, in the
        # block's local pixel coordinates) of every matched grid point in
        # each of the three images against image 0 - these are exactly
        # the (un-averaged) values setGridLine's dispersion curve_fit is
        # built from (it fits against their mean, myCal.pixelPositionWavelength)
        xInBox = (myCal.positionMatrix[:, 1, myCal.boolMatrix]
                  - myCal.positionMatrix[0, 1, myCal.boolMatrix])
        dispersionFitPositions = (xInBox + myCal.bwidth - myCal.xShift).flatten()

        return {'myCal': myCal, 'blockImage': blockImage, 'peakMask': peakMask,
                'dispersionFitPositions': dispersionFitPositions, 'point00': point00}

    def _onGetBlocksStarted(self):
        ''' runs on the GUI thread when the worker actually starts '''
        print('getting blocks...')

    def _onGetBlocksFinished(self, result):
        ''' runs on the GUI thread with the worker's return value '''
        myCal = result['myCal']
        self.myCal = myCal

        # average dispersion = mean wavelength step per pixel across the
        # fitted (generally slightly non-linear) pixel-to-wavelength curve
        averageDispersion = np.mean(np.abs(np.diff(myCal.wavelength)))

        # average distance between neighboring blocks = length of the
        # lattice's x basis vector (grid spacing along the row direction)
        xVec = myCal.imMoStack[0].xVec
        blockDistance = np.sqrt(xVec[0]**2 + xVec[1]**2)

        self.getBlocksGui.bwidth.value = str(2 * myCal.bwidth + 1)
        self.getBlocksGui.bheight.value = str(2 * myCal.bheight + 1)
        self.getBlocksGui.wavelengthRange.value = f'{myCal.wavelength.min():.1f} - {myCal.wavelength.max():.1f} nm'
        self.getBlocksGui.dispersion.value = f'{averageDispersion:.3f} nm/px'
        self.getBlocksGui.blockDistance.value = f'{blockDistance:.2f} px'
        self.getBlocksGui.status.value = 'finished'
        self.getBlocksGui.call_button.enabled = True
        self.warpGui.call_button.enabled = True
        self.warpGui.status.value = ''

        self._plotWavelengthCheck(myCal, result['dispersionFitPositions'])

        # visual check: fitted spectral grid over the images, same as the
        # manual visual check in utility/generateCalibrationObject.py
        viewer = self._getViewer()

        if self._blockLayer is not None and self._blockLayer in viewer.layers:
            self._blockLayer.data = result['blockImage']
        else:
            self._blockLayer = viewer.add_image(result['blockImage'], name='spectral block', opacity=0.2)

        # located spectral peaks (every spot), shown white
        if self._peakLayer is not None and self._peakLayer in viewer.layers:
            self._peakLayer.data = result['peakMask']
        else:
            self._peakLayer = viewer.add_image(result['peakMask'], name='peaks',
                                                colormap='gray', blending='additive', opacity=1)

        if self._zeroLayer is not None and self._zeroLayer in viewer.layers:
            self._zeroLayer.data = result['point00']
        else:
            self._zeroLayer = viewer.add_points(result['point00'], size=50, opacity=0.2, name='zero position')

        print('getting blocks finished')

    def _onGetBlocksError(self, exc):
        ''' runs on the GUI thread if _runGetBlocks raised '''
        self.getBlocksGui.status.value = f'error: {exc}'
        self.getBlocksGui.call_button.enabled = True
        print(f'getting blocks failed: {exc}')

    def _plotWavelengthCheck(self, myCal, dispersionFitPositions):
        ''' visual check that the pixel-to-wavelength fit and the grid are
        on the proper position - same plot as the manual visual check in
        utility/generateCalibrationObject.py, plus a gray box marking the
        spectral block (its pixel width, 0 to 2*bwidth+1, against the
        wavelengthRange actually achieved by the fit) and, below it - on
        the same, shared pixel axis - a histogram of every calibration
        spot's relative x-position (dispersion direction) in the block:
        the un-averaged values the dispersion curve fit itself is built
        from, so the three clusters in the histogram should line up with
        the three vertical marker lines above them. Reuses the same
        matplotlib figure on every call instead of piling up a new window
        each time Get Blocks is pressed. '''
        if self._checkFig is None or not plt.fignum_exists(self._checkFig.number):
            self._checkFig, (self._checkAx, self._checkHistAx) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        boxWidth = 2 * myCal.bwidth + 1
        wavelengthMin = myCal.wavelength.min()
        wavelengthMax = myCal.wavelength.max()

        ax = self._checkAx
        ax.cla()

        # gray box: pixel extent of the block (x, index 0 to boxWidth-1 -
        # same range as myCal.wavelength, which has boxWidth entries)
        # against the wavelength range actually achieved by the fit (y)
        ax.add_patch(Rectangle((0, wavelengthMin), boxWidth - 1, wavelengthMax - wavelengthMin,
                                facecolor='gray', edgecolor='none', alpha=0.3, zorder=0))

        ax.plot(myCal.wavelength, zorder=2)
        ax.set(ylabel='wavelength [nm]', title='fit pixel to wavelength')
        ax.set_xlim(0, boxWidth - 1)
        ax.axhline(y=myCal.wavelengthStack[0])
        ax.axhline(y=myCal.wavelengthStack[1])
        ax.axhline(y=myCal.wavelengthStack[2])
        ax.axvline(x=myCal.pixelPositionWavelength[0] + myCal.bwidth - myCal.xShift)
        ax.axvline(x=myCal.pixelPositionWavelength[1] + myCal.bwidth - myCal.xShift)
        ax.axvline(x=myCal.pixelPositionWavelength[2] + myCal.bwidth - myCal.xShift)

        # histogram of the relative peak x-position (dispersion direction
        # only) in the block, for the points the dispersion fit uses
        histAx = self._checkHistAx
        histAx.cla()
        histAx.hist(dispersionFitPositions, bins=100, color='gray')
        histAx.set(xlabel='pixels', ylabel='count', title='dispersion fit points')

        self._checkFig.tight_layout()
        self._checkFig.canvas.draw_idle()
        plt.show(block=False)

    # ------------------------------------------------------------------
    # Calculate Warp - fit the warping matrices (CalibrateFrom3Images.setWarpMatrix)
    # ------------------------------------------------------------------

    def _calculateWarp(self):
        ''' fit the warping matrices and show the same set of checks as
        the end of utility/generateCalibrationObject.py, in a background
        worker thread. Requires Get Blocks to have run first. '''
        if self.myCal is None:
            print('run Get Blocks before Calculate Warp')
            return

        self.warpGui.status.value = 'calculating warp...'
        self.warpGui.call_button.enabled = False

        worker = create_worker(self._runCalculateWarp,
                                self.myCal, self.whiteImage,
                                _start_thread=True,
                                _connect={'started': self._onCalculateWarpStarted,
                                          'returned': self._onCalculateWarpFinished,
                                          'errored': self._onCalculateWarpError})

        # keep a reference so the worker/thread isn't garbage-collected mid-run
        self._warpWorker = worker

    def _runCalculateWarp(self, myCal, whiteImage):
        ''' pure computation, runs on the worker thread - must not touch
        any GUI, matplotlib or napari widgets directly. '''
        myCal.setWarpMatrix(spectral=True, subpixel=True)

        iS = myCal.imageStack[0] + myCal.imageStack[1] + myCal.imageStack[2]

        # measured peak position (2) vs the ideal grid position at that
        # wavelength's pixel column (3)
        label = np.zeros_like(myCal.imageStack[0], dtype='int')
        avePoint = myCal.gridLine.getPositionInt()
        for ii, imMo in enumerate(myCal.imMoStack):
            px = np.argmin(np.abs(myCal.wavelength - myCal.wavelengthStack[ii]))
            label[imMo.position[:, 0].astype(int), imMo.position[:, 1].astype(int)] = 2
            label[avePoint[:, 0].astype(int), avePoint[:, 1].astype(int) - myCal.bwidth + px] = 3

        spImage = spImageCor = None
        if whiteImage is not None:
            spImage = myCal.getSpectralImage(whiteImage, aberrationCorrection=False)
            spImageCor = myCal.getSpectralImage(whiteImage, aberrationCorrection=True)

        spImage2 = myCal.getSpectralImage(iS, aberrationCorrection=False)
        spImage2Cor = myCal.getSpectralImage(iS, aberrationCorrection=True)

        # 2D histogram of the spectral image (raw and aberration-corrected)
        _y = np.reshape(np.swapaxes(spImage2, 0, 2), (-1, spImage2.shape[0]))
        y = _y[np.sum(_y, axis=1) > 0]
        x = np.zeros_like(y) + myCal.wavelength

        _yCor = np.reshape(np.swapaxes(spImage2Cor, 0, 2), (-1, spImage2.shape[0]))
        yCor = _yCor[np.sum(_yCor, axis=1) > 0]

        bins = [myCal.wavelength.shape[0] * 2, myCal.wavelength[::-1]]
        H, _, _ = np.histogram2d(y.flatten(), x.flatten(), bins=bins)
        HCor, _, _ = np.histogram2d(yCor.flatten(), x.flatten(), bins=bins)

        return {'myCal': myCal, 'whiteImage': whiteImage, 'iS': iS, 'label': label,
                'spImage': spImage, 'spImageCor': spImageCor,
                'spImage2': spImage2, 'spImage2Cor': spImage2Cor,
                'H': H, 'HCor': HCor}

    def _onCalculateWarpStarted(self):
        ''' runs on the GUI thread when the worker actually starts '''
        print('calculating warp...')

    def _onCalculateWarpFinished(self, result):
        ''' runs on the GUI thread with the worker's return value.
        Rebuilds the extra napari windows from scratch, closing whatever
        was left over from a previous Calculate Warp run first. '''
        self.warpGui.status.value = 'finished'
        self.warpGui.call_button.enabled = True

        myCal = result['myCal']
        whiteImage = result['whiteImage']
        iS = result['iS']
        label = result['label']

        for viewer in self._warpViewers:
            try:
                viewer.close()
            except RuntimeError:
                pass
        self._warpViewers = []

        # warp matrices themselves + measured/ideal peak positions
        viewer2 = napari.Viewer(title='warp matrices')
        viewer2.add_image(myCal.dSubpixelShiftMatrix, name='SubpixelShift', colormap='turbo')
        viewer2.add_image(myCal.dSpectralWarpMatrix, name='SpectralWarp', colormap='turbo')
        viewer2.add_labels(label, name='peaks', opacity=1)
        self._warpViewers.append(viewer2)

        # raw vs warped calibration/white image
        viewer3 = napari.Viewer(title='warped images')
        viewer3.add_image(iS, name='calibration image', opacity=1, colormap='turbo')
        viewer3.add_image(myCal.getWarpedImage(iS), name='warped calibration image', opacity=1, colormap='turbo')
        if whiteImage is not None:
            viewer3.add_image(whiteImage, name='white')
            viewer3.add_image(myCal.getWarpedImage(whiteImage), name='warped white')
        viewer3.add_labels(label, name='peak real and ideal')
        self._warpViewers.append(viewer3)

        # aberration-corrected vs not, for white reference and for the peaks
        viewer4 = napari.Viewer(title='spectral images')
        if result['spImage'] is not None:
            viewer4.add_image(result['spImage'], name='white not cor', opacity=1, colormap='turbo')
            viewer4.add_image(result['spImageCor'], name='white cor', opacity=1, colormap='turbo')
        viewer4.add_image(result['spImage2'], name='peak not cor', opacity=1, colormap='turbo')
        viewer4.add_image(result['spImage2Cor'], name='peak cor', opacity=1, colormap='turbo')
        self._warpViewers.append(viewer4)

        # 2D histogram (wavelength vs intensity), raw and corrected
        viewer5 = napari.Viewer(title='2D histogram')
        viewer5.add_image(result['H'][::-1], name='raw', opacity=1, colormap='turbo')
        viewer5.add_image(result['HCor'][::-1], name='corrected', opacity=1, colormap='turbo')
        self._warpViewers.append(viewer5)

        # resulting hyperspectral cube, raw and corrected
        self.sViewer = XYWViewer(np.stack((result['spImage2'], result['spImage2Cor'])), myCal.wavelength)
        self.sViewer.run()

        print('calculating warp finished')

    def _onCalculateWarpError(self, exc):
        ''' runs on the GUI thread if _runCalculateWarp raised '''
        self.warpGui.status.value = f'error: {exc}'
        self.warpGui.call_button.enabled = True
        print(f'calculating warp failed: {exc}')

    def _save(self, saveFolder):
        ''' save the calibration object to file '''
        if self.myCal is None:
            print('run Get Blocks (and Calculate Warp) before Save')
            return

        self.myCal.saveClass(classFolder=str(saveFolder))
        print(f'calibration saved in folder: {saveFolder}')


if __name__ == "__main__":
    pass

#%%
