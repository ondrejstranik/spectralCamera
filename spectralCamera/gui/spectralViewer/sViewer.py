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

    def __init__(self,xywImage=None, wavelength= None, **kwargs):
        ''' initialise the class '''
    
        super().__init__()

        # data parameter
        if xywImage is not None:
            self.xywImage=xywImage  # spectral 3D image
        else:
            self.xywImage = np.zeros((2,2,2))

        if wavelength is not None:
            self.wavelength = wavelength
        else:
            self.wavelength = np.arange(self.xywImage.shape[0]) 

        # calculated parameters
        self.spotSpectra = SpotSpectraSimple(self.xywImage)
        self.pointSpectra = []

        # add napari
        if 'show' in kwargs:
            self.viewer = NapariViewer(show=kwargs['show'])
        else:
            self.viewer = NapariViewer()
        self.spectraLayer = None
        self.pointLayer = None

        # pyqt
        if not hasattr(self, 'dockWidgetParameter'):
            self.dockWidgetParameter = None 
        if not hasattr(self, 'dockWidgetData'):
            self.dockWidgetData = None 

        # spectra widget
        self.spectraGraph = None
        self.lineplotList = []
        self.penList = []
        self.maxNLine = SViewer.DEFAULT['maxNLine']
        
        # set this qui of this class
        SViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the gui '''

        # set napari viewer
        window_height = self.viewer.window._qt_window.sizeHint().height()
        window_width = self.viewer.window._qt_window.sizeHint().width()
        # add image layer
        self.spectraLayer = self.viewer.add_image(self.xywImage, rgb=False, colormap="gray", 
                                            name='SpectraCube', blending='additive')
        # add point layer
        self.pointLayer = self.viewer.add_points(name='points', size=5, face_color='red')
        # add text overlay
        self.viewer.text_overlay.visible = True


        # set active layer of napari
        self.viewer.layers.selection.active = self.spectraLayer

        # add widget spectraGraph
        self.spectraGraph = pg.PlotWidget()
        # speed up drawing
        #self.spectraGraph.disableAutoRange()
        self.spectraGraph.setTitle(f'Spectra')
        styles = {'color':'r', 'font-size':'20px'}
        self.spectraGraph.setLabel('left', 'Intensity', units='a.u.')
        self.spectraGraph.setLabel('bottom', 'Wavelength ', units= 'nm')
        dw = self.viewer.window.add_dock_widget(self.spectraGraph, name = 'spectra')
        # pre allocate lines and pens for the graph
        for ii in range(self.maxNLine):
            self.lineplotList.append(self.spectraGraph.plot())
            self.lineplotList[-1].hide()
            self._speedUpLineDrawing(self.lineplotList[-1])
            self.penList.append(pg.mkPen(width=1))

        # register the graph in menu
        menuBar = self.viewer.window._qt_window.menuBar()
        # ---- Find Window menu ----
        window_menu = None
        for action in menuBar.actions():
            if action.text().replace("&", "") == "Window":
                window_menu = action.menu()
                break
        if window_menu is not None:
            window_menu.addAction(dw.toggleViewAction())

        #dw.setMaximumHeight(window_height)
        #dw.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        # tabify the widget
        if self.dockWidgetData is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetData,dw)
        self.dockWidgetData = dw
        self.viewer.window._qt_window.resizeDocks([dw], [500], Qt.Vertical)

        # connect events in napari
        # connect changes of the slicer in the viewer
        self.viewer.dims.events.current_step.connect(self.updateTextOverlay)
        
        # connect changes in data in this layer for update in main tread
        self.pointLayer.events.data.connect(self.updateSpectra)
        self.pointLayer._face.events.current_color.connect(self.colorChange)

        # connect signal for a changes in the points and their color
        self.pointLayer._face.events.current_color.connect(lambda: self.sigUpdateData.emit())
        self.pointLayer.events.data.connect(lambda: self.sigUpdateData.emit())


    def calculateSpectra(self):
        ''' calculate the spectra at the given points'''
        print('recalculating mask')
        self.spotSpectra.setSpot(self.pointLayer.data)
        self.pointSpectra = self.spotSpectra.getSpectra()

    def colorChange(self):
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

        # if there is no pointSpectra then do not continue
        try:
            nSig = len(self.pointSpectra)
        except:
            return
    
        # pointSpectra
        for ii in np.arange(nSig):
            try:
                self.penList[ii].setColor(QColor.fromRgbF(*list(
                    self.pointLayer.face_color[ii])))
            except:
                print('error occurred in drawSpectraGraph - pointSpectra')
                traceback.print_exc()
            try:
                self.lineplotList[ii].setData(self.wavelength, self.pointSpectra[ii], pen = self.penList[ii])
                self.lineplotList[ii].show()
            except:
                print('error occurred in drawSpectraGraph - pointSpectra')
                print(f'point spectra {self.pointSpectra[ii]}')
                traceback.print_exc()
                
        # hide extra lines
        for ii in np.arange(self.maxNLine - nSig):
            self.lineplotList[ii+nSig].hide()


    def updateSpectraGraph(self):
        ''' only for back compatibility only -  use instead drawSpectraGraph '''
        self.drawSpectraGraph()

    def updateSpectra(self, event):
        ''' update spectra (calculate and draw) after the data were changed '''
        if not np.array_equal(self.spotSpectra.spotPosition,self.pointLayer.data):
            self.calculateSpectra()
            self.drawSpectraGraph()

    def updateTextOverlay(self):
        ''' update spectra histogram title'''
        
        try:
            myw = self.wavelength[int(self.viewer.dims.point[0])]
        except:
            myw = 0
        
        self.viewer.text_overlay.text = f' {myw} nm'

    def setImage(self, image):
        ''' set the image. only if the dimensions of the image changed, then mask is recalculated'''
        # check if the mask has to be recalculated
        calculateMask = True
        try:
            if self.spotSpectra.wxyImage.shape == image.shape:
                calculateMask = False
        except:
                print('some error in shape')

        self.xywImage = image
        self.spectraLayer.data = self.xywImage
        self.spotSpectra.setImage(self.xywImage)
        if calculateMask: 
            self.calculateSpectra()
            print('dimension did not equal')
        else:
            self.pointSpectra = self.spotSpectra.getSpectra()

        self.updateSpectraGraph()

    def setWavelength(self, wavelength):
        ''' set wavelength '''        
        self.wavelength = wavelength
        if len(wavelength)!= self.xywImage.shape[0]:
            print('number of wavelength is not equal to image spectral channels')

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

        














