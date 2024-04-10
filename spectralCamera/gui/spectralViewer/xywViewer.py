'''
class for viewing spots's plasmon resonance
'''
import napari
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy
from qtpy.QtCore import Qt

import numpy as np


class XYWViewer():
    ''' main class for viewing spectral images'''

    def __init__(self,xywImage=None, wavelength= None, **kwargs):
        ''' initialise the class '''
    
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
            self.viewer = napari.Viewer(show=kwargs['show'])
        else:
            self.viewer = napari.Viewer()

        self.spectraLayer = None
        self.pointLayer = None

        # pyqt
        if not hasattr(self, 'dockWidgetParameter'):
            self.dockWidgetParameter = None 
        if not hasattr(self, 'dockWidgetData'):
            self.dockWidgetData = None 


        self.spectraGraph = None
        self.lineplotList = []
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
        self.spectraLayer._keep_auto_contrast = True
        self.viewer.layers.selection.active = self.spectraLayer

        # add widget spectraGraph
        self.spectraGraph = pg.plot()
        self.spectraGraph.setTitle(f'Spectra')
        styles = {'color':'r', 'font-size':'20px'}
        self.spectraGraph.setLabel('left', 'Intensity', units='a.u.')
        self.spectraGraph.setLabel('bottom', 'Wavelength ', units= 'nm')
        dw = self.viewer.window.add_dock_widget(self.spectraGraph, name = 'spectra')
        #dw.setMaximumHeight(window_height)
        #dw.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        # tabify the widget
        if self.dockWidgetData is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetData,dw)
        self.dockWidgetData = dw
        self.viewer.window._qt_window.resizeDocks([dw], [500], Qt.Vertical)

        # add spectra histogram widget
        self.spectraHistogram = pg.plot()
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

        self.calculateSpectraHistogram()
        self.drawSpectraHistogram()

        # connect events in napari
        # connect changes of the slicer in the viewer
        self.viewer.dims.events.current_step.connect(self.updateHistogram)
       # connect changes in data in this layer
        self.pointLayer.events.data.connect(self.updateSpectra)
        #self.pointLayer.events.data.connect(self.drawSpectraGraph)

    def calculateSpectra(self):
        ''' calculate the spectra at the given points'''
        self.pointSpectra = []
              
        # calculate pointSpectra
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
        
    def drawSpectraGraph(self):
        ''' draw all new lines in the spectraGraph '''
        # remove all lines
        self.spectraGraph.clear()
        self.lineplotList = []

        try:
            # pointSpectra
            for ii in np.arange(len(self.pointSpectra)):
                mypen = QPen(QColor.fromRgbF(*list(
                    self.pointLayer.face_color[ii])))
                mypen.setWidth(0)
                lineplot = self.spectraGraph.plot(pen= mypen)
                lineplot.setData(self.wavelength, self.pointSpectra[ii])
                self.lineplotList.append(lineplot)
        except:
            print('error occurred in drawSpectraGraph - pointSpectra')

    def updateSpectraGraph(self):
        ''' update the lines in the spectra graph '''

        myPoints = self.pointLayer.data
        try:
            # pointSpectra
            for ii in np.arange(len(self.pointSpectra)):
                myline = self.lineplotList[ii]
                mypen = QPen(QColor.fromRgbF(*list(
                    self.pointLayer.face_color[ii])))
                mypen.setWidth(0)
                myline.setData(self.wavelength,self.pointSpectra[ii], pen = mypen)
        except:
            print('error occured in update_spectraGraph - points')

    def updateSpectra(self):
        ''' update spectra after the data were changed '''
        self.calculateSpectra()
        self.drawSpectraGraph()

    def calculateSpectraHistogram(self):
        ''' calculate histogram of given spectral channel '''
        try:
            (self.spectraHistogramValue, self.spectraHistogramBin) = np.histogram(
                self.spectraLayer.data[int(self.viewer.dims.point[0]),:,:])
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
        try:
            myw = self.wavelength[int(self.viewer.dims.point[0])]
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

    def run(self):
        ''' start napari engine '''
        napari.run()

if __name__ == "__main__":
    import pytest
    #retcode = pytest.main(['tests/test_spectralViewer.py::test_XYWViewer'])

        














