'''
class for viewing spots's plasmon resonance
'''
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer
from spectralCamera.algorithm.spotSpectra import SpotSpectra

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


class SpotSpectraViewer(XYWViewer):
    ''' class viewing spectra of plasmon spots'''
 
    def __init__(self, xywImage=None, wavelength= None, **kwargs):
        ''' initialise the class '''

        super().__init__(xywImage=xywImage, wavelength= wavelength, **kwargs)

        # calculated parameters

        #gui parameters
        self.maskLayer = None # layer of the mask with spots and bcg area
        self.showRawSpectra = True
        self.spectraParameterGui = None

        self.spotSpectra = SpotSpectra(self.xywImage)

        # set gui
        SpotSpectraViewer._setWidget(self)

    def _setWidget(self):
        ''' prepare the qui '''

        # set napari
        # add mask layer
        # define a colormap with transparent 0 values
        colors = np.linspace(
            start=[0, 1, 0, 1],
            stop=[1, 0, 0, 1],
            num=3,
            endpoint=True
        )
        colors[0] = np.array([0., 1., 0., 0])
        transparentRedGreen_colormap = {
        'colors': colors,
        'name': 'red_and_green',
        'interpolation': 'linear'
        }
        self.maskLayer = self.viewer.add_image(self.spotSpectra.getMask(), name='spot_mask',
        colormap=transparentRedGreen_colormap, opacity = 0.5)

        # set pyqt
        @magicgui(auto_call= 'True')
        def spectraParameterGui(
            showRawSpectra: bool = self.showRawSpectra,
            pxAve: int = self.spotSpectra.pxAve,
            pxBcg: int = self.spotSpectra.pxBcg,
            pxSpace: int = self.spotSpectra.pxSpace
            ):

            self.spotSpectra.pxAve = pxAve
            self.spotSpectra.pxBcg = pxBcg
            self.spotSpectra.pxSpace = pxSpace
            self.showRawSpectra = showRawSpectra
            self.updateSpectra()

        self.spectraParameterGui = spectraParameterGui

        # add widget setParameterGui
        dw = self.viewer.window.add_dock_widget(self.spectraParameterGui, name ='view param', area='bottom')
        if self.dockWidgetParameter is not None:
            self.viewer.window._qt_window.tabifyDockWidget(self.dockWidgetParameter,dw)
        self.dockWidgetParameter = dw


    def calculateSpectra(self):
        ''' calculate spectra at the given points'''

        self.spotSpectra.setSpot(self.pointLayer.data)

        self.pointSpectra = self.spotSpectra.getA()

    def updateSpectra(self):
        super().updateSpectra()

        # update the mask in napari
        self.maskLayer.data = self.spotSpectra.getMask()

    def drawSpectraGraph(self):
        ''' draw all new lines in the spectraGraph '''

        self.lineplotList = []                
        self.spectraGraph.clear()

        if self.showRawSpectra:
            try:
                for ii in np.arange(len(self.pointSpectra)):
                    mypen = QPen(QColor.fromRgbF(*list(
                        self.pointLayer.face_color[ii])))
                    mypen.setWidth(0)
                    # raw spot
                    lineplot = self.spectraGraph.plot(pen= mypen)
                    lineplot.setData(self.wavelength, self.spotSpectra.spectraRawSpot[ii])
                    self.lineplotList.append(lineplot)
                    # raw bcg
                    lineplot = self.spectraGraph.plot(pen= mypen)
                    lineplot.setData(self.wavelength, self.spotSpectra.spectraRawBcg[ii])
                    self.lineplotList.append(lineplot)
            except:
                print('error occurred in drawSpectraGraph')
        else:
            for ii in np.arange(len(self.pointSpectra)):
                mypen = QPen(QColor.fromRgbF(*list(
                    self.pointLayer.face_color[ii])))
                mypen.setWidth(0)
                lineplot = self.spectraGraph.plot(pen= mypen)
                lineplot.setData(self.wavelength, self.pointSpectra[ii])
                self.lineplotList.append(lineplot)
            # print('error occurred in drawSpectraGraph')            
        # set Title
        if self.showRawSpectra:
            self.spectraGraph.setTitle(f'Spectra')
            self.spectraGraph.setLabel('left', 'Intensity', units='a.u.')        
        else:
            self.spectraGraph.setTitle(f'1 - Transmission')
            self.spectraGraph.setLabel('left', 'percentage', units='a.u.')

    def updateSpectraGraph(self):
        ''' update the lines in the spectra graph '''
        if self.showRawSpectra:
            try:
                for ii in np.arange(len(self.pointSpectra)):
                    myline = self.lineplotList[2*ii]
                    mypen = QPen(QColor.fromRgbF(*list(
                        self.pointLayer.face_color[ii])))
                    mypen.setWidth(0)
                    myline.setData(self.wavelength,self.spotSpectra.spectraRawSpot[ii], pen = mypen)
                    myline = self.lineplotList[2*ii+1]
                    myline.setData(self.wavelength,self.spotSpectra.spectraRawBcg[ii], pen = mypen)
            except:
                print('error occurred in updateSpectraGraph')
        else:
            try:
                for ii in np.arange(len(self.pointSpectra)):
                    myline = self.lineplotList[ii]
                    mypen = QPen(QColor.fromRgbF(*list(
                        self.pointLayer.face_color[ii])))
                    mypen.setWidth(0)
                    myline.setData(self.wavelength,self.pointSpectra[ii], pen = mypen)
            except:
                print('error occurred in updateSpectraGraph')



if __name__ == "__main__":
        
    im = np.random.rand(10,100,100)
    wavelength = np.arange(im.shape[0])*1.3+ 10
    sViewer = SpotSpectraViewer(im, wavelength)
    sViewer.run()














