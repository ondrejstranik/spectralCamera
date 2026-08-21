'''
class for viewing spectra in specta-spatial image
'''
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget, QApplication
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal
from qtpy.QtCore import QTimer
from viscope.gui.napariViewer.napariViewer import NapariViewer
from qtpy.QtCore import QObject
from spectralCamera.algorithm.spotSpectraSimple import SpotSpectraSimple
import traceback
from timeit import default_timer as timer

import napari

import numpy as np

pg.setConfigOptions(useOpenGL=True,antialias=False)


class SViewer(QObject):
    ''' main class for viewing point spectra in spectral images'''
    DEFAULT = {'nameGUI':'SViewer',
               'maxNLine': 200} # maxNLine ... max number of line plotted in the graph

    sigUpdateData = Signal()
    sigSelectionChanged = Signal()
    sigColorChanged = Signal()

    def __init__(self,image=None, wavelength= None, **kwargs):
        ''' initialise the class '''
    
        super().__init__()

        # data parameters
        self.spotSpectra = SpotSpectraSimple(image)

        # set parameters
        if image is not None:
            self.spotSpectra.setImage(image)  # spectral 3D image
        else:
            self.spotSpectra.setImage(np.zeros((2,2,2)))
        if wavelength is not None:
            self.spotSpectra.wavelength = wavelength
        else:
            self.spotSpectra.wavelength = np.arange(self.spotSpectra.image.shape[0]) 


        # add napari
        if 'show' in kwargs:
            self.viewer = NapariViewer(show=kwargs['show'])
        else:
            self.viewer = NapariViewer()
        self.spectraLayer = None # new layer in napari
        self.pointLayer = None # new layer in napari
        self.window_menu = None # window in the napari bar menu

        # pyqt
        if not hasattr(self, 'dockWidgetParameter'):
            self.dockWidgetParameter = None 
        if not hasattr(self, 'dockWidgetData'):
            self.dockWidgetData = None 

        # spectra widget
        self.spectraGraph = None
        self.linePlotList = []
        self.penList = []
        self.maxNLine = SViewer.DEFAULT['maxNLine']

        # per-spot metadata, shape-compatible with plim.algorithm.spotData.
        # SpotData.table. Gets replaced-by-alias (plasmonViewer.table =
        # positionTrack.sD.table) by the owning GUI once a real SpotData
        # exists; this default just keeps SViewer usable stand-alone.
        self.table = {'name': [], 'color': [], 'visible': []}

        # set this qui of this class
        SViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the gui '''

        # set napari viewer
        # add image layer
        self.spectraLayer = self.viewer.add_image(self.spotSpectra.image, rgb=False, colormap="gray",
                                            name='SpectraCube', blending='additive')
        # keep the contrast limits auto-scaling to the data on every update
        self.spectraLayer._keep_auto_contrast = True
        # best-effort: also press the actual 'continuous' auto-contrast
        # button so its icon in the layer controls panel matches the state
        # set above. The path into napari's private Qt widgets differs by
        # napari version (e.g. controls.autoScaleBar pre-0.6 vs.
        # controls._contrast_limits_control.auto_scale_bar from 0.6.6), so
        # don't let a lookup failure on either break the auto-contrast itself
        try:
            controls = self.viewer.window._qt_viewer.controls.widgets[self.spectraLayer]
            autoScaleBar = getattr(controls, 'autoScaleBar', None) or controls._contrast_limits_control.auto_scale_bar
            autoScaleBar._auto_btn.setChecked(True)
        except (KeyError, AttributeError):
            pass
        # add point layer
        self.pointLayer = self.viewer.add_points(name='points', size=5, face_color='red')
        # annotate each point with its index
        self.pointLayer.features = {'names': []}
        self.pointLayer.text = {
            'string': '{names}',
            'size': 12,
            'color': 'green',
            'translation': np.array([-5, 0])}
        # add text overlay
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.text = ' nm'
        # set active layer of napari
        self.viewer.layers.selection.active = self.spectraLayer
        # find Window menu in napari
        menuBar = self.viewer.window._qt_window.menuBar()
        for action in menuBar.actions():
            if action.text().replace("&", "") == "Window":
                self.window_menu = action.menu()
                break

        # add widget spectraGraph
        self.spectraGraph = pg.PlotWidget()
        self.spectraGraph.setTitle(f'Spectra')
        styles = {'color':'r', 'font-size':'20px'}
        self.spectraGraph.setLabel('left', 'Intensity', units='a.u.')
        self.spectraGraph.setLabel('bottom', 'Wavelength ', units= 'nm')
        # speed up drawing
        #self.spectraGraph.disableAutoRange()
        # pre allocate lines and pens for the graph
        for ii in range(self.maxNLine):
            self.linePlotList.append(self.spectraGraph.plot())
            self.linePlotList[-1].hide()
            self._speedUpLineDrawing(self.linePlotList[-1])
            self.penList.append(pg.mkPen(width=1))

        # add dock widget and tabify it
        dw = self.viewer.window.add_dock_widget(self.spectraGraph, name = 'spectra')
        if self.dockWidgetData is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetData,dw)
        self.dockWidgetData = dw
        # register the graph in menu
        if self.window_menu is not None:
            self.window_menu.addAction(dw.toggleViewAction())

        # by default, make the spectra graph as wide as the image canvas.
        # deferred to let Qt/napari finish laying out the window first, since
        # the width read out during construction is not yet the final one;
        # resizeDocks() does not work here because the spectra dock is alone
        # in its area (nothing to redistribute space with), so the width is
        # instead forced with a temporary fixed width, then released so the
        # dock stays user-resizable
        def _matchSpectraGraphWidth():
            width = self.viewer.window._qt_window.width() // 2
            self.dockWidgetData.setFixedWidth(width)
            QApplication.instance().processEvents()
            self.dockWidgetData.setMinimumWidth(0)
            self.dockWidgetData.setMaximumWidth(16777215)
            # the image canvas has just been resized by the width adjustment
            # above; reset the view now so the image fits the final canvas
            # size instead of the (wider) one it had before this ran
            self.viewer.reset_view()
        QTimer.singleShot(0, _matchSpectraGraphWidth)

        # connect events
        # connect changes of the slicer in the viewer
        self.viewer.dims.events.current_step.connect(self.updateTextOverlay)
        # connect changes in data in this layer for update in main thread.
        # pointChanged() and sigUpdateData both need to react to the SAME
        # "did the data actually change" check; done once here in a single
        # handler, since pointChanged() updates spotSpectra.spotPosition to
        # match pointLayer.data as a side effect - if that check were done
        # separately for each reaction (as two independent connections),
        # whichever ran first would make spotSpectra.spotPosition catch up,
        # causing the second one to always see "no change" and never fire
        self.pointLayer.events.data.connect(lambda: self._onPointDataChanged())
        self.pointLayer._face.events.current_color.connect(self.colorChanged)

        # connect signal for a change of a point's color - kept separate from
        # sigUpdateData (points added/moved/deleted) since it needs different
        # handling downstream (e.g. reapplying current_color to the selection)
        self.pointLayer._face.events.current_color.connect(lambda: self.sigColorChanged.emit())

        # connect signal for a change of the spot selection (e.g. clicking points in napari)
        self.pointLayer.selected_data.events.items_changed.connect(
            lambda *_: self.sigSelectionChanged.emit())

        # 'v' toggles visibility of the currently selected spot(s). Bound on
        # self.viewer (not pointLayer) so it fires regardless of which layer
        # is currently active - pointLayer usually isn't (spectraLayer is
        # kept active for its own controls), so a binding scoped to
        # pointLayer would rarely trigger. Verified this doesn't collide
        # with napari's own default 'v' ('toggle_selected_visibility', which
        # hides/shows the whole active layer) even with the points layer
        # explicitly made active - the viewer-level keymap takes this key
        # first regardless.
        self.viewer.bind_key('v', lambda viewer: self.toggleVisibility(), overwrite=True)

    def _onPointDataChanged(self):
        ''' react once to a points-layer data change (add/move/delete),
        checked a single time so both reactions agree on whether anything
        actually changed '''
        if not np.array_equal(self.spotSpectra.spotPosition, self.pointLayer.data):
            self.pointChanged()
            self.sigUpdateData.emit()

    def colorChanged(self):
        ''' change the color of the spectral with the change of the point color
        very cumbersome way due to the internal processes in napari'''
        idx = list(self.pointLayer.selected_data)

        # record the picked color into self.table (aliased to SpotData.table
        # once wired up by the owning GUI) for the currently selected point(s)
        try:
            hexColor = '#{:02x}{:02x}{:02x}'.format(
                *(np.asarray(self.pointLayer._face.current_color[:3]) * 255).astype(int))
            for ii in idx:
                self.table['color'][ii] = hexColor
        except Exception:
            print('error updating table color from current_color')
            traceback.print_exc()

        # it is necessary to remember it
        _aux = self.pointLayer.face_color[idx]

        # this allow to draw spectral lines with proper color
        # however it will stop redrawing the points with a new color
        self.pointLayer.face_color[idx] = self.pointLayer._face.current_color
        self.redraw(modified='point')
        # therefore the face_colors are set back only to be put internally to new values
        self.pointLayer.face_color[idx] = _aux

    def syncPointsFromTable(self):
        ''' push self.table['color']/['visible']/['name'] onto pointLayer.
        Call whenever the table was edited via an external widget (e.g.
        SignalWidget/InfoWidget) rather than via the viewer itself. '''
        rgb = self.table.get('color')
        if not rgb:
            return
        vis = self.table['visible']
        _color = [rgb[ii] + 'ff' if vis[ii] == 'True' else rgb[ii] + '00' for ii in range(len(rgb))]

        # defensive guard in case this bulk push ever fires current_color as
        # a side effect (not observed directly, but this costs nothing and
        # prevents it from being misread as a genuine colour pick made in
        # the viewer, which would otherwise re-enter colorChanged())
        with self.pointLayer._face.events.current_color.blocker():
            self.pointLayer.face_color = _color

        try:
            self.pointLayer.features = {'names': list(self.table['name'])}
        except Exception:
            print('error updating point annotations from table')
            traceback.print_exc()

    def toggleVisibility(self):
        ''' toggle visibility of the currently selected spot(s) - bound to
        the 'v' key. Writes into self.table['visible'] (aliased to
        SpotData.table once wired up by the owning GUI), pushes the result
        onto the viewer, and notifies the rest of the GUI via sigUpdateData
        (the same signal used for point add/move/delete, since the
        downstream handling - resync widgets, redraw - is identical). '''
        idx = list(self.pointLayer.selected_data)
        if not idx:
            return
        for ii in idx:
            self.table['visible'][ii] = 'False' if self.table['visible'][ii] == 'True' else 'True'
        self.syncPointsFromTable()
        self.sigUpdateData.emit()

    def drawSpectraGraph(self):
        ''' draw all lines in the spectraGraph '''

        # if there is no points then do not continue
        try:
            nSig = len(self.spotSpectra.getSpectra())
        except:
            return
    
        self.spectraGraph.setUpdatesEnabled(False)

        # loop over all data lines
        for ii in np.arange(nSig):
            # hide the line for spots marked not-visible (e.g. via the 'v'
            # key toggle), same convention as SignalWidget.drawGraph()
            try:
                isVisible = self.table['visible'][ii] == 'True'
            except (IndexError, KeyError, TypeError):
                isVisible = True
            try:
                self.penList[ii].setColor(QColor.fromRgbF(*list(
                    self.pointLayer.face_color[ii])))
            except:
                print('error occurred in drawSpectraGraph - could not set color')
                traceback.print_exc()
            try:
                self.linePlotList[ii].setData(self.spotSpectra.wavelength,
                                              self.spotSpectra.getSpectra()[ii],
                                              pen = self.penList[ii])
                self.linePlotList[ii].show() if isVisible else self.linePlotList[ii].hide()
            except:
                print('error occurred in drawSpectraGraph - could not set data')
                traceback.print_exc()
                
        # hide extra lines
        for ii in np.arange(self.maxNLine - nSig):
            self.linePlotList[ii+nSig].hide()

        self.spectraGraph.setUpdatesEnabled(True)

    def calculateSpectra(self):
        ''' calculate the spectra '''
        self.spotSpectra.calculateSpectra()

    def updatePointAnnotations(self):
        ''' keep the point label annotation in sync with the current points '''
        self.pointLayer.features = {'names': list(self.table['name'])}

    def _growTableToPointCount(self):
        ''' resize self.table's per-spot lists in place to match
        pointLayer.data, preserving existing name/visible entries by
        position and defaulting new ones. color is always re-read fresh
        from pointLayer.face_color (napari has already correctly reindexed
        it on add/delete; name/visible have no such array to read back
        from, so a point deleted from the middle can still shift name/
        visible out of alignment - a pre-existing limitation, not
        introduced here). self.table is mutated by key, never rebound, so
        any alias to it (e.g. SpotData.table) stays intact. '''
        nSpot = len(self.pointLayer.data)
        oldName = self.table.get('name') or []
        oldVisible = self.table.get('visible') or []
        self.table['color'] = ['#{:02x}{:02x}{:02x}'.format(*c) for c in
                                (self.pointLayer.face_color[:, :3] * 255).astype(int)]
        self.table['name'] = [oldName[ii] if ii < len(oldName) else str(ii) for ii in range(nSpot)]
        self.table['visible'] = [oldVisible[ii] if ii < len(oldVisible) else 'True' for ii in range(nSpot)]

    def pointChanged(self):
        ''' updates the points, calculate spectra and draw the spectra'''
        print('recalculating mask')
        self.spotSpectra.setSpot(self.pointLayer.data)
        self.spotSpectra.setMask()

        self._growTableToPointCount()
        self.updatePointAnnotations()

        self.calculateSpectra()
        self.redraw(modified='point')

    def updateTextOverlay(self):
        ''' update wavelength overlay info text in viewer'''
        try:
            myw = self.spotSpectra.wavelength[int(self.viewer.dims.point[0])]
        except:
            myw = 0
        
        self.viewer.text_overlay.text = f' {myw} nm'

    def setImage(self, image):
        ''' set the image. it recalculate the spectra'''
        self.spotSpectra.setImage(image)
        self.calculateSpectra()
        self.redraw(modified='image')

    def setWavelength(self, wavelength):
        ''' set wavelength '''        
        self.spotSpectra.setWavelength(wavelength)

    def redraw(self,modified='all'):
        ''' only redraw the images, spectra. It does not recalculate it '''
        start = timer()
        if (modified=='image') or (modified=='all'):
            newImage = self.spotSpectra.getImage()
            resetView = self.spectraLayer.data.shape != newImage.shape
            self.spectraLayer.data = newImage
            if resetView:
                self.viewer.reset_view()
            self.drawSpectraGraph()
#        if (modified=='point') or (modified=='all'):
        if (modified=='point'):
            self.drawSpectraGraph()
        end = timer()
        print(f'viewer redraw evaluation time {end -start} s')

    def _speedUpLineDrawing(self,line):
        ''' set parameter of a line in a pyqtplot so that it is quicker'''
        line.setDownsampling(auto=True)
        line.setClipToView(True)
        line.setSkipFiniteCheck(True)
        return line

    def run(self):
        ''' start napari engine '''
        napari.run()

if __name__ == "__main__":
    pass

        














