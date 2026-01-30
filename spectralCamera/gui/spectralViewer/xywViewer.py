'''
class for viewing spots's plasmon resonance
'''
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget
from qtpy.QtCore import Qt
from qtpy.QtCore import Signal
from viscope.gui.napariViewer.napariViewer import NapariViewer
from qtpy.QtCore import QObject

import napari

import numpy as np

pg.setConfigOptions(useOpenGL=True,antialias=False)


class XYWViewer(QObject):
    ''' main class for viewing spectral images'''
    DEFAULT = {'nameGUI':'XYWViewer',
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

        #spectral processing parameters
        self.pxAve = 3 # radius of the area of spectra averaging

        # calculated parameters
        self.pointSpectra = [] # list of spectra
        self.spectraHistogramValue = None
        self.spectraHistogramBin = None

        # napari
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
        self.maxNLine = XYWViewer.DEFAULT['maxNLine']
        
        # histogram widget
        self.spectraHistogram = None
        self.spectraBarGraph = None

        # set this qui of this class
        XYWViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the gui '''

        window_height = self.viewer.window._qt_window.sizeHint().height()
        window_width = self.viewer.window._qt_window.sizeHint().width()

        # add image layer
        self.spectraLayer = self.viewer.add_image(self.xywImage, rgb=False, colormap="gray", 
                                            name='SpectraCube', blending='additive')
        # add point layer
        self.pointLayer = self.viewer.add_points(name='points', size=5, face_color='red')

        
  
        # set some parameters of napari
        #self.spectraLayer._keep_auto_contrast = True
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
        # pre allocate lines for the graph
        for ii in range(self.maxNLine):
            self.lineplotList.append(self.spectraGraph.plot())
            self.lineplotList[-1].hide()
            self._speedUpLineDrawing(self.lineplotList[-1])

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

        # add spectra histogram widget
        self.spectraHistogram = pg.PlotWidget()
        self.spectraHistogram.setTitle(f'Spectra Histogram')
        styles = {'color':'r', 'font-size':'20px'}
        self.spectraHistogram.setLabel('left', '#', units='a.u.')
        self.spectraHistogram.setLabel('bottom', 'pixel value', units= 'a.u.')
        #self.spectraGraph.setFixedHeight(300)
        dw = self.viewer.window.add_dock_widget(self.spectraHistogram, name='histogram spectrum')
        #dw.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        #dw.setMaximumHeight(window_height)
        # tabify the widget
        self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetData,dw)
        self.dockWidgetData = dw
        self.viewer.window._qt_window.resizeDocks([dw], [500], Qt.Vertical)
        # register the graph in menu
        if window_menu is not None:
            window_menu.addAction(dw.toggleViewAction())

        self.calculateSpectraHistogram()
        self.drawSpectraHistogram()

        # connect events in napari
        # connect changes of the slicer in the viewer
        self.viewer.dims.events.current_step.connect(self.updateHistogram)
       # connect changes in data in this layer
        self.pointLayer.events.data.connect(self.updateSpectra)
        self.pointLayer._face.events.current_color.connect(self.colorChange)
        self.pointLayer._face.events.current_color.connect(lambda: self.sigUpdateData.emit())
        self.pointLayer.events.data.connect(lambda: self.sigUpdateData.emit())


    def calculateSpectra(self):
        ''' calculate the spectra at the given points'''
        self.pointSpectra = []

        # calculate pointSpectra
        if self.spectraLayer.data.ndim ==3:
            myPoints = self.pointLayer.data
            for ii in np.arange(myPoints.shape[0]):
                try:
                    temp = np.sum(
                        self.spectraLayer.data[
                        :,
                        int(myPoints[ii,0])-self.pxAve:int(myPoints[ii,0])+self.pxAve+1,
                        int(myPoints[ii,1])-self.pxAve:int(myPoints[ii,1])+self.pxAve+1
                        ], axis = (1,2)) / (2*self.pxAve+1)**2
                    self.pointSpectra.append(temp)
                except:
                    self.pointSpectra.append(0*self.wavelength)

        if self.spectraLayer.data.ndim ==4:
            myPoints = self.pointLayer.data
            for ii in np.arange(myPoints.shape[0]):
                try:
                    temp = np.sum(
                        self.spectraLayer.data[int(self.viewer.dims.point[0]),
                        :,
                        int(myPoints[ii,0])-self.pxAve:int(myPoints[ii,0])+self.pxAve+1,
                        int(myPoints[ii,1])-self.pxAve:int(myPoints[ii,1])+self.pxAve+1
                        ], axis = (1,2)) / (2*self.pxAve+1)**2
                    self.pointSpectra.append(temp)
                except:
                    self.pointSpectra.append(0*self.wavelength)
    
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
    
        # define pen object
        mypen = QPen()
        mypen.setWidth(0)

        # pointSpectra
        for ii in np.arange(nSig):
            try:
                mypen.setColor(QColor.fromRgbF(*list(
                    self.pointLayer.face_color[ii])))
            except:
                pass

            try:
                self.lineplotList.setData(self.wavelength, self.pointSpectra[ii])
                self.lineplotList[ii].show()
            except:
                print('error occurred in drawSpectraGraph - pointSpectra')
                
        # hide extra lines
        for ii in np.arange(self.maxNLine - nSig):
            self.lineplotList[ii+nSig].hide()


    def updateSpectraGraph(self):
        ''' only for back compatibility only -  use instead drawSpectraGraph '''
        self.drawSpectraGraph()

    def updateSpectra(self):
        ''' update spectra after the data were changed '''
        self.calculateSpectra()
        self.drawSpectraGraph()

    def calculateSpectraHistogram(self):
        ''' calculate histogram of given spectral channel '''

        if self.spectraLayer.data.ndim ==3:
            try:
                (self.spectraHistogramValue, self.spectraHistogramBin) = np.histogram(
                    self.spectraLayer.data[int(self.viewer.dims.point[0]),:,:])
            except:
                self.spectraHistogramBin = np.arange(2)
                self.spectraHistogramValue = 0*self.spectraHistogramBin[0:-1]

        if self.spectraLayer.data.ndim ==4:
            try:
                (self.spectraHistogramValue, self.spectraHistogramBin) = np.histogram(
                    self.spectraLayer.data[int(self.viewer.dims.point[0]),
                    int(self.viewer.dims.point[1]),:,:])
            except:
                self.spectraHistogramBin = np.arange(2)
                self.spectraHistogramValue = 0*self.spectraHistogramBin[0:-1]


    def drawSpectraHistogram(self):
        ''' draw spectral histogram '''

        self.spectraHistogram.clear()
        self.spectraBarGraph = pg.BarGraphItem(x = self.spectraHistogramBin[0:-1], 
        height = self.spectraHistogramValue, 
        width= self.spectraHistogramBin[1]- self.spectraHistogramBin[0])
        self.spectraHistogram.addItem(self.spectraBarGraph)

    def updateSpectraHistogram(self):
        ''' update spectra histogram values '''
        try:
            self.spectraBarGraph.setOpts(x0=self.spectraHistogramBin[0:-1], 
            height=self.spectraHistogramValue,
            width= self.spectraHistogramBin[1]- self.spectraHistogramBin[0])
        except:
            print('error occurred in updateSpectraHistogram')

    def updateSpectraHistogramTitle(self):
        ''' update spectra histogram title'''
        if self.spectraLayer.data.ndim ==3:
            try:
                myw = self.wavelength[int(self.viewer.dims.point[0])]
                self.spectraHistogram.setTitle(f'Spectra Histogram - {myw} nm')
            except:
                print('error occurred in updateSpectraHistogramTitle')
        if self.spectraLayer.data.ndim ==4:
            try:
                myw = self.wavelength[int(self.viewer.dims.point[1])]
                self.spectraHistogram.setTitle(f'Spectra Histogram - {myw} nm')
            except:
                print('error occurred in updateSpectraHistogramTitle')


    def updateHistogram(self):
        ''' update histogram values '''
        self.calculateSpectraHistogram()
        self.updateSpectraHistogram()
        self.updateSpectraHistogramTitle()

    def setImage(self, image):
        ''' set the image '''
        self.xywImage = image
        self.spectraLayer.data = self.xywImage
        #self.viewer.reset_view()
        self.calculateSpectra()
        self.updateSpectraGraph()
        self.updateHistogram()

    def setWavelength(self, wavelength):
        ''' set wavelength '''        
        self.wavelength = wavelength
        if len(wavelength)!= self.xywImage.shape[0]:
            print('number of wavelength is not equal to image spectral channels')
        self.updateSpectraHistogramTitle()

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

        














