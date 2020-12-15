# -*- coding: utf-8 -*-
"""
This is a GUI that runs a neural network on the input data and displays
original data and the result in a scrollable form.
v1 was made using matplotlib. Unfortunately that is very slow, so here
I try to use pyqtgraph for faster performance

Created on Mon Oct  5 12:18:48 2020

@author: stepp
"""
import os
import re
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox,
                             QLabel, QProgressBar, QPushButton, QSlider,
                             QWidget)
from skimage import transform
from tensorflow import keras

from NNfeeder import prepareNNImages
from NNio import loadTifFolder, loadTifStack
from QtImageViewerMerge import QtImageViewerMerge
from SatsGUI import SatsGUI

# Adjust for different screen sizes
QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


class NNGui(QWidget):
    """Used to visualize and examine data taken using an adaptive temporal sampling approach.
    Used for Mito/Drp and Bacteria/FtsZ and a neural network tought to detect division events
    in mitochondria (see Mito2Drp1.py). Uses QtImageViewerMerge to visualize the channels and
    SATS_GUI to visualize the temporal sampling of the data.
    Loads stacks of interleaved images of the two channels or folders that were populated using
    the ATS/smartiSIM approach in combination with network_Watchdog.py
    Originally based on TifStackViewer

    Args:
        QWidget ([type]): [description]
    """

    frameChanged = pyqtSignal([], [int])

    def __init__(self, app):
        QWidget.__init__(self)

        # Handle to the image stack tiffcapture object.
        self._tiffCaptureHandle = None
        self.currentFrameIndex = None

        # Image frame viewer.
        self.viewerOrig = QtImageViewerMerge()
        self.imageItemDrpOrig = self.viewerOrig.addImage()
        self.imageItemMitoOrig = self.viewerOrig.addImage()
        self.viewerOrig.setLUT(self.imageItemDrpOrig, 'reds')
        self.viewerOrig.setLUT(self.imageItemMitoOrig, 'grey')

        self.viewerProc = QtImageViewerMerge()
        self.imageItemDrpProc = self.viewerProc.addImage()
        self.imageItemMitoProc = self.viewerProc.addImage()
        self.viewerProc.setLUT(self.imageItemDrpProc, 'reds')
        self.viewerProc.setLUT(self.imageItemMitoProc, 'grey')

        self.viewerNN = QtImageViewerMerge()
        self.imageItemNN = self.viewerNN.addImage()
        self.viewerNN.setLUT(self.imageItemNN, 'inferno')
        self.loadBox = QGroupBox()

        # Connect the viewers
        self.viewerOrig.viewBox.setXLink(self.viewerProc.viewBox)
        self.viewerOrig.viewBox.setYLink(self.viewerProc.viewBox)
        self.viewerNN.viewBox.setXLink(self.viewerProc.viewBox)
        self.viewerNN.viewBox.setYLink(self.viewerProc.viewBox)

        # Slider and arrow buttons for frame traversal.
        self.sliderBox = QGroupBox()
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setMinimumHeight(20)
        self.prevFrameButton = QPushButton("<")
        self.nextFrameButton = QPushButton(">")

        # loadBox content: Buttons for load model and data
        self.modelButton = QPushButton("load model")
        self.dataButton = QPushButton("load data")
        self.orderButton = QPushButton("order: Drp first")
        self.currentFrameLabel = QLabel('Frame')
        self.loadingStatusLabel = QLabel('')
        # progress bar for loading data
        self.progress = QProgressBar(self)

        self.outputPlot = SatsGUI()

        pen = pg.mkPen(color='#AAAAAA', style=Qt.DashLine)
        self.frameLine = pg.InfiniteLine(pos=0.5, angle=90, pen=pen)
        self.outputPlot.plotItem1.addItem(self.frameLine)

        # Connect functions to the interactive elements
        self.modelButton.clicked.connect(self.loadModel)
        self.dataButton.clicked.connect(self.loadData)
        self.orderButton.clicked.connect(self.orderChange)
        self.prevFrameButton.clicked.connect(self.prevFrame)
        self.nextFrameButton.clicked.connect(self.nextFrame)

        self.frameSlider.sliderPressed.connect(self.startTimer)
        self.frameSlider.sliderReleased.connect(self.stopTimer)

        # Layout.
        grid = QGridLayout(self)
        grid.addWidget(self.viewerOrig, 0, 0)
        grid.addWidget(self.viewerProc, 0, 1)
        grid.addWidget(self.viewerNN, 0, 2)
        grid.addWidget(self.outputPlot, 1, 0, 1, 2)
        grid.addWidget(self.loadBox, 1, 2)
        grid.addWidget(self.sliderBox, 2, 0, 1, 3)

        gridprogress = QGridLayout(self.sliderBox)
        gridprogress.addWidget(self.prevFrameButton, 0, 0)
        gridprogress.addWidget(self.frameSlider, 0, 1)
        gridprogress.addWidget(self.nextFrameButton, 0, 3)
        gridprogress.setContentsMargins(0, 0, 0, 0)

        gridBox = QGridLayout(self.loadBox)
        gridBox.addWidget(self.modelButton, 0, 0)
        gridBox.addWidget(self.dataButton, 0, 1)
        gridBox.addWidget(self.orderButton, 0, 2)
        gridBox.addWidget(self.loadingStatusLabel, 1, 0, 1, 3)
        gridBox.addWidget(self.progress, 2, 0, 1, 3)
        gridBox.addWidget(self.currentFrameLabel, 3, 0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        # init variables
        self.mode = None
        self.model = None
        self.imageDrpOrig = None
        self.imageMitoOrig = None
        self.nnOutput = None
        self.mitoDataFull = None
        self.drpDataFull = None
        self.maxPos = None
        self.nnRecalculated = None

        self.app = app
        self.order = 1
        self.linePen = pg.mkPen(color='#AAAAAA')

    def loadData(self):
        """load tif stack or ATS folder into the GUI. Reports progress using a textbox and the
        progress bar.
        """
        # load images
        self.progress.setValue(0)
        nnImageSize = 128
        pixelCalib = 56  # nm per pixel
        resizeParam = pixelCalib/81  # no unit

        # Loading different types of data
        # order 0 is mito first
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/')
        self.loadingStatusLabel.setText('Loading images from files into arrays')
        if not re.match(r'img_channel\d+_position\d+_time', os.path.basename(fname[0])) is None:
            print("Folder mode")
            self.mode = 'folder'
            self.imageDrpOrig, self.imageMitoOrig, self.nnOutput = loadTifFolder(
                os.path.dirname(fname[0]), resizeParam, self.order, self.progress, self.app)
        else:
            self.mode = 'stack'
            print("Stack mode")
            self.imageDrpOrig, self.imageMitoOrig = loadTifStack(fname[0], self.order)

        print(fname[0])

        print(self.imageMitoOrig.shape)
        # Do NN for all images
        frameNum = self.imageMitoOrig.shape[0]
        postSize = round(self.imageMitoOrig.shape[1]*resizeParam)
        self.nnOutput = np.zeros((frameNum, postSize, postSize))
        self.mitoDataFull = np.zeros_like(self.nnOutput)
        self.drpDataFull = np.zeros_like(self.nnOutput)
        outputData = []
        self.maxPos = []
        self.nnRecalculated = np.zeros(frameNum)
        # set up the progress bar
        self.progress.setRange(0, frameNum-1)

        self.loadingStatusLabel.setText('Processing frames and running the network')
        for frame in range(0, self.imageMitoOrig.shape[0]):
            inputData, positions = prepareNNImages(
                self.imageMitoOrig[frame, :, :],
                self.imageDrpOrig[frame, :, :], nnImageSize)

            # Do the NN calculation if there is not already a file there
            if self.model == 'folder' and np.max(self.nnOutput[frame]) > 0:
                nnDataPres = 1
            else:
                nnDataPres = 0
                outputPredict = self.model.predict_on_batch(inputData)
                self.nnRecalculated[frame] = 1

            i = 0
            st0 = positions['stitch']
            st1 = None if st0 == 0 else -st0
            for pos in positions['px']:
                if nnDataPres == 0:
                    self.nnOutput[frame, pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] =\
                                outputPredict[i, st0:st1, st0:st1, 0]
                self.mitoDataFull[frame, pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] = \
                    inputData[i, st0:st1, st0:st1, 0]
                self.drpDataFull[frame, pos[0]+st0:pos[2]-st0, pos[1]+st0:pos[3]-st0] = \
                    inputData[i, st0:st1, st0:st1, 1]
                i += 1

            outputData.append(np.max(self.nnOutput[frame, :, :]))
            self.maxPos.append(list(zip(*np.where(self.nnOutput[frame] == outputData[-1]))))
            self.progress.setValue(frame)
            self.app.processEvents()

        self.loadingStatusLabel.setText('Resize the original frames to fit the output')
        print(postSize)
        imageDrpOrigScaled = np.zeros((self.imageDrpOrig.shape[0], postSize, postSize))
        imageMitoOrigScaled = np.zeros((self.imageDrpOrig.shape[0], postSize, postSize))
        for i in range(0, self.imageMitoOrig.shape[0]):
            imageDrpOrigScaled[i] = transform.rescale(self.imageDrpOrig[i], resizeParam)
            imageMitoOrigScaled[i] = transform.rescale(self.imageMitoOrig[i], resizeParam)
            self.progress.setValue(i)
            self.app.processEvents()
        self.imageDrpOrig = np.array(imageDrpOrigScaled)
        self.imageMitoOrig = np.array(imageMitoOrigScaled)
        print(self.imageMitoOrig.shape)

        # Make a rolling mean of the output Data
        # N = 10
        # outputDataSmooth = np.ones(len(outputData))
        # outputDataSmooth[0:N] = np.ones(N)*np.mean(outputData[0:N])
        # for x in range(N, len(outputData)):
        #     outputDataSmooth[x] = np.sum(outputData[x-N:x])/N
        self.app.processEvents()
        self.outputPlot.deleteRects()
        self.outputPlot.frames.setData([])
        self.outputPlot.nnframeScatter.setData([])
        if self.mode == 'stack':
            self.outputPlot.nnline.setData(outputData)
            self.outputPlot.scatter.setData(range(0, len(outputData)), outputData)
        else:
            self.loadingStatusLabel.setText('Getting the timing data')
            self.outputPlot.loadData(os.path.dirname(fname[0]), self.progress, self.app)
            self.app.processEvents()
            for i in range(-1, 6):
                self.outputPlot.inc = i
                self.outputPlot.updatePlot()

        self.frameSlider.setMaximum(self.nnOutput.shape[0]-1)

        self.refreshGradients()
        self.onTimer()
        self.loadingStatusLabel.setText('Done')
        self.viewerProc.viewBox.setRange(xRange=(0, postSize), yRange=(0, postSize))

    def loadModel(self):
        """ Load a .h5 model generated using Keras """
        self.loadingStatusLabel.setText('Loading Model')
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/',
            "Keras models (*.h5)")
        print(fname[0])
        self.model = keras.models.load_model(fname[0], compile=True)
        self.loadingStatusLabel.setText('Done')

    def orderChange(self):
        """ React to a press of the order button to read interleaved data into the right order """
        if self.order == 0:
            self.order = 1
            orderStr = 'order: Drp first'
        else:
            self.order = 0
            orderStr = 'order: Mito first'
        print(self.order)
        self.orderButton.setText(orderStr)

    def onTimer(self):
        """ Reset the data in the GUI on the timer when button or slider is pressed """
        i = self.frameSlider.value()
        self.imageItemMitoOrig.setImage(self.imageMitoOrig[i])
        self.imageItemDrpOrig.setImage(self.imageDrpOrig[i])
        self.imageItemMitoProc.setImage(self.mitoDataFull[i])
        self.imageItemDrpProc.setImage(self.drpDataFull[i])
        self.imageItemNN.setImage(self.nnOutput[i])
        self.currentFrameLabel.setText(str(i))
        if self.mode == 'stack':
            self.frameLine.setValue(i)
        else:
            self.frameLine.setValue(self.outputPlot.elapsed[i])
        self.viewerOrig.cross.setPosition([self.maxPos[i][0]])
        self.viewerProc.cross.setPosition([self.maxPos[i][0]])
        self.viewerNN.cross.setPosition([self.maxPos[i][0]])

    def refreshGradients(self):
        """ refresh the images when a LUT was changed in the popup window """
        self.viewerOrig.setImage(self.imageMitoOrig[0], 1)
        self.viewerOrig.setImage(self.imageDrpOrig[0], 0)
        self.viewerProc.setImage(self.mitoDataFull[0], 1)
        self.viewerProc.setImage(self.drpDataFull[0], 0)
        self.viewerNN.setImage(self.nnOutput[0], 0)

    def startTimer(self):
        """ start Timer when slider is pressed """
        self.timer.start()

    def stopTimer(self):
        """ stop timer when slider is released"""
        self.timer.stop()

    def nextFrame(self):
        """ display next frame in all viewBoxes when '>' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i + 1)
        self.onTimer()

    def prevFrame(self):
        """ display previous frame in all viewBoxes when '<' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i - 1)
        self.onTimer()


def main():
    """ main method to run and display the NNGui """
    app = QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    stackViewer = NNGui(app)

    stackViewer.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
