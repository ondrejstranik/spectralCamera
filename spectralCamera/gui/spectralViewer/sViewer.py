'''
class for viewing spectra in specta-spatial image
'''
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal
from viscope.gui.napariViewer.napariViewer import NapariViewer
from qtpy.QtCore import QObject
from spectralCamera.algorithm.spotSpectraSimple import SpotSpectraSimple
import traceback

import napari

import numpy as np

pg.setConfigOptions(useOpenGL=True,antialias=False)


class SViewer(QObject):
    ''' main class for viewing point spectra in spectral images'''
    DEFAULT = {'nameGUI':'SViewer',
               'maxNLine': 200} # maxNLine ... max number of line plotted in the graph

    sigUpdateData = Signal()

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
        
        # set this qui of this class
        SViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the gui '''

        # set napari viewer
        # add image layer
        self.spectraLayer = self.viewer.add_image(self.spotSpectra.image, rgb=False, colormap="gray", 
                                            name='SpectraCube', blending='additive')
        # add point layer
        self.pointLayer = self.viewer.add_points(name='points', size=5, face_color='red')
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
        self.viewer.window._qt_window.resizeDocks([dw], [500], Qt.Vertical)
        # register the graph in menu
        if self.window_menu is not None:
            self.window_menu.addAction(dw.toggleViewAction())


        # connect events
        # connect changes of the slicer in the viewer
        self.viewer.dims.events.current_step.connect(self.updateTextOverlay)
        # connect changes in data in this layer for update in main tread
        self.pointLayer.events.data.connect(self.updateMask)
        self.pointLayer._face.events.current_color.connect(self.updateColor)

        # connect signal for a changes in the points and their color
        self.pointLayer._face.events.current_color.connect(lambda: self.sigUpdateData.emit())
        self.pointLayer.events.data.connect(lambda: self.sigUpdateData.emit())

    def updateColor(self):
        ''' change the color of the spectral with the change of the point color
        very cumbersome way due to the internal processes in napari'''
        # it is necessary to remember it 
        _aux = self.pointLayer.face_color[list(self.pointLayer.selected_data)]

        # this allow to draw spectral lines with proper color
        # however it will stop redrawing the points with a new color
        self.pointLayer.face_color[list(self.pointLayer.selected_data)] = self.pointLayer._face.current_color
        self.drawSpectraGraph()

        # therefore the face_colors are set back only to be put internally to new values
        self.pointLayer.face_color[list(self.pointLayer.selected_data)] = _aux

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
                self.linePlotList[ii].show()
            except:
                print('error occurred in drawSpectraGraph - could not set data')
                traceback.print_exc()
                
        # hide extra lines
        for ii in np.arange(self.maxNLine - nSig):
            self.linePlotList[ii+nSig].hide()

        self.spectraGraph.setUpdatesEnabled(True)


    def updateMask(self):
        ''' if points changed than update mask, spectra and graph'''
        # if old one is not the new one
        if not np.array_equal(self.spotSpectra.spotPosition,self.pointLayer.data):
            print('recalculating mask')
            self.spotSpectra.setSpot(self.pointLayer.data)
            self.drawSpectraGraph()

    def updateTextOverlay(self):
        ''' update spectra histogram title'''
        try:
            myw = self.spotSpectra.wavelength[int(self.viewer.dims.point[0])]
        except:
            myw = 0
        
        self.viewer.text_overlay.text = f' {myw} nm'

    def setImage(self, image):
        ''' set the image. only if the dimensions of the image changed, then mask is recalculated'''
        # check if the mask has to be recalculated
        calculateMask = True
        try:
            if self.spotSpectra.image.shape == image.shape:
                calculateMask = False
        except:
            pass
        if calculateMask: 
            print('image dimensions not equal, recalculating mask')
            self.spotSpectra.setImageSpot(image,self.pointLayer.data)
        else:
            self.spotSpectra.setImage(image)

        self.spectraLayer.data = image
        self.drawSpectraGraph()

    def setWavelength(self, wavelength):
        ''' set wavelength '''        
        self.spotSpectra.setWavelength(wavelength)

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

        














