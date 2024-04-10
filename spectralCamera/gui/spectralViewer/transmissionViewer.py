'''
class for viewing spots's plasmon resonance
'''
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer

import napari
import time
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPen
from qtpy.QtWidgets import QLabel, QSizePolicy
from qtpy.QtCore import Qt

from magicgui import magicgui
from enum import Enum
from typing import Annotated, Literal

import numpy as np


class TransmissionViewer(XYWViewer):
    ''' main class viewing transmission '''

    def __init__(self,xywImage=None, wavelength= None,**kwargs):
        ''' initialise the class '''

        super().__init__(xywImage=xywImage, wavelength= wavelength, **kwargs)

        # calculated parameters
        self.bcgSpectra = [] # list of background spectra

        #gui parameters
        self.bcgLayer = None
        self.lineplotList2 = []
        self.showRawSpectra = True
        self.spectraParameterGui = None

        # set gui
        TransmissionViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the qui '''

        # set pyqt
        @magicgui(auto_call= 'True')
        def spectraParameterGui(
            showRawSpectra: bool = self.showRawSpectra,
            pxAve: int = self.pxAve,
            ):

            if ((pxAve != self.pxAve) or (showRawSpectra != self.showRawSpectra) or
                (spectraOffset != self.spectraOffset)):
                self.pxAve = pxAve
                self.showRawSpectra = showRawSpectra
                self.updateSpectra()

        self.spectraParameterGui = spectraParameterGui

        # add widget setParameterGui
        dw = self.viewer.window.add_dock_widget(self.spectraParameterGui, name ='view param', area='bottom')
        #dw.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        #dw.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)

        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter,dw)
        self.dockWidgetParameter = dw
        self.viewer.window._qt_window.resizeDocks([dw], [100], Qt.Vertical)

        # set napari
        # add background layer
        self.bcgLayer = self.viewer.add_points(name='bcg_points', size=5)

        # connect changes in data in this layer
        self.bcgLayer.events.data.connect(self.updateSpectra)

    def calculateSpectra(self):
        ''' calculate spectra at the given points'''
        super().calculateSpectra()
        
        self.bcgSpectra = []

        # calculate bcgSpectra
        myPoints = self.bcgLayer.data
        for ii in np.arange(myPoints.shape[0]):
            try:
                temp = np.sum(
                    self.spectraLayer.data[
                    :,
                    int(myPoints[ii,0])-self.pxAve:int(myPoints[ii,0])+self.pxAve+1,
                    int(myPoints[ii,1])-self.pxAve:int(myPoints[ii,1])+self.pxAve+1
                    ], axis = (1,2)) / (2*self.pxAve+1)**2
                self.bcgSpectra.append(temp)
            except:
                self.bcgSpectra.append(0*self.wavelength)

        if self.showRawSpectra == False:
            try:
                for ii in np.arange(len(self.pointSpectra)):
                    self.pointSpectra[ii] = 1 - self.pointSpectra[ii]/self.bcgSpectra[0]
            except:
                print('could not normalise the spectra')

    def drawSpectraGraph(self):
        ''' draw all new lines in the spectraGraph '''
        super().drawSpectraGraph()

        self.lineplotList2 = []                

        if self.showRawSpectra:
            try:
                # bcgSpectra
                for ii in np.arange(len(self.bcgSpectra)):
                    mypen = QPen(QColor.fromRgbF(*list(
                        self.bcgLayer.face_color[ii])))
                    mypen.setWidth(0)
                    lineplot = self.spectraGraph.plot(pen= mypen)
                    lineplot.setData(self.wavelength, self.bcgSpectra[ii])
                    self.lineplotList2.append(lineplot)
            except:
                print('error occurred in drawSpectraGraph - bcgSpectra')

        # set Title
        if self.showRawSpectra:
            self.spectraGraph.setTitle(f'Spectra')
            self.spectraGraph.setLabel('left', 'Intensity', units='a.u.')        
        else:
            self.spectraGraph.setTitle(f'1 - Transmission')
            self.spectraGraph.setLabel('left', 'percentage', units='a.u.')

    def updateSpectraGraph(self):
        ''' update the lines in the spectra graph '''
        super().updateSpectraGraph()

        if self.showRawSpectra:

            myPoints = self.bcgLayer.data
            try:
                # pointSpectra
                for ii in np.arange(len(self.bcgSpectra)):
                    myline = self.lineplotList2[ii]
                    mypen = QPen(QColor.fromRgbF(*list(
                        self.bcgLayer.face_color[ii])))
                    mypen.setWidth(0)
                    myline.setData(self.wavelength,self.bcgSpectra[ii], pen = mypen)
            except:
                print('error occurred in update_spectraGraph - bcgSpectra')


if __name__ == "__main__":
    im = np.random.rand(10,100,100)
    wavelength = np.arange(im.shape[0])*1.3+ 10
    sViewer = TransmissionViewer(im, wavelength)
    sViewer.run()














