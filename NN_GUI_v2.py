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
from nnIO import loadTifFolder, loadTifStack
from QtImageViewerMerge import QtImageViewerMerge
from SATS_GUI import SATS_GUI

# Adjust for different screen sizes
QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


class MultiPageTIFFViewerQt(QWidget):

    frameChanged = pyqtSignal([], [int])

    def __init__(self, app):
        QWidget.__init__(self)

        # Handle to the image stack tiffcapture object.
        self._tiffCaptureHandle = None
        self.currentFrameIndex = None

        # Image frame viewer.
        self.viewer_Orig = QtImageViewerMerge()
        self.imageDrpOrig = self.viewer_Orig.addImage()
        self.imageMitoOrig = self.viewer_Orig.addImage()
        self.viewer_Orig.setLUT(self.imageDrpOrig, 'reds')
        self.viewer_Orig.setLUT(self.imageMitoOrig, 'grey')

        self.viewer_Proc = QtImageViewerMerge()
        self.imageDrpProc = self.viewer_Proc.addImage()
        self.imageMitoProc = self.viewer_Proc.addImage()
        self.viewer_Proc.setLUT(self.imageDrpProc, 'reds')
        self.viewer_Proc.setLUT(self.imageMitoProc, 'grey')

        self.viewer_nn = QtImageViewerMerge()
        self.imageNN = self.viewer_nn.addImage()
        self.viewer_nn.setLUT(self.imageNN, 'inferno')
        self.loadBox = QGroupBox()

        # Connect the viewers
        self.viewer_Orig.vb.setXLink(self.viewer_Proc.vb)
        self.viewer_Orig.vb.setYLink(self.viewer_Proc.vb)
        self.viewer_nn.vb.setXLink(self.viewer_Proc.vb)
        self.viewer_nn.vb.setYLink(self.viewer_Proc.vb)

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
        self.LoadingStatusLabel = QLabel('')
        # progress bar for loading data
        self.progress = QProgressBar(self)

        self.outputPlot = SATS_GUI()

        pen = pg.mkPen(color='#AAAAAA', style=Qt.DashLine)
        self.frameLine = pg.InfiniteLine(pos=0.5, angle=90, pen=pen)
        self.outputPlot.pI.addItem(self.frameLine)

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
        grid.addWidget(self.viewer_Orig, 0, 0)
        grid.addWidget(self.viewer_Proc, 0, 1)
        grid.addWidget(self.viewer_nn, 0, 2)
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
        gridBox.addWidget(self.LoadingStatusLabel, 1, 0, 1, 3)
        gridBox.addWidget(self.progress, 2, 0, 1, 3)
        gridBox.addWidget(self.currentFrameLabel, 3, 0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        # init variables
        self.mode = None
        self.model = None
        self.image_drpOrig = None
        self.image_mitoOrig = None
        self.nnOutput = None
        self.mitoDataFull = None
        self.drpDataFull = None
        self.maxPos = None
        self.nnRecalculated = None


        self.app = app
        self.order = 1
        self.linePen = pg.mkPen(color='#AAAAAA')

    def loadData(self):
        # load images
        self.progress.setValue(0)
        nnImageSize = 128
        pixelCalib = 56  # nm per pixel
        resizeParam = pixelCalib/81  # no unit

        # Loading different types of data
        # order 0 is mito first
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/')
        self.LoadingStatusLabel.setText('Loading images from files into arrays')
        if not re.match(r'img_channel\d+_position\d+_time', os.path.basename(fname[0])) is None:
            print("Folder mode")
            self.mode = 'folder'
            self.image_drpOrig, self.image_mitoOrig, self.nnOutput = loadTifFolder(
                os.path.dirname(fname[0]), resizeParam, self.order, self.progress, self.app)
        else:
            self.mode = 'stack'
            print("Stack mode")
            self.image_drpOrig, self.image_mitoOrig = loadTifStack(fname[0], self.order)

        print(fname[0])

        print(self.image_mitoOrig.shape)
        # Do NN for all images
        frameNum = self.image_mitoOrig.shape[0]
        postSize = round(self.image_mitoOrig.shape[1]*resizeParam)
        self.nnOutput = np.zeros((frameNum, postSize, postSize))
        self.mitoDataFull = np.zeros_like(self.nnOutput)
        self.drpDataFull = np.zeros_like(self.nnOutput)
        outputData = []
        self.maxPos = []
        self.nnRecalculated = np.zeros(frameNum)
        # set up the progress bar
        self.progress.setRange(0, frameNum-1)

        self.LoadingStatusLabel.setText('Processing frames and running the network')
        for frame in range(0, self.image_mitoOrig.shape[0]):
            inputData, positions = prepareNNImages(
                self.image_mitoOrig[frame, :, :],
                self.image_drpOrig[frame, :, :], nnImageSize)

            # Do the NN calculation if there is not already a file there
            if self.model == 'folder' and np.max(self.nnOutput[frame]) > 0:
                nnDataPres = 1
            else:
                nnDataPres = 0
                output_predict = self.model.predict_on_batch(inputData)
                self.nnRecalculated[frame] = 1

            i = 0
            st = positions['stitch']
            st1 = None if st == 0 else -st
            for p in positions['px']:
                if nnDataPres == 0:
                    self.nnOutput[frame, p[0]+st:p[2]-st, p[1]+st:p[3]-st] =\
                                output_predict[i, st:st1, st:st1, 0]
                self.mitoDataFull[frame, p[0]+st:p[2]-st, p[1]+st:p[3]-st] = \
                    inputData[i, st:st1, st:st1, 0]
                self.drpDataFull[frame, p[0]+st:p[2]-st, p[1]+st:p[3]-st] = \
                    inputData[i, st:st1, st:st1, 1]
                i += 1

            outputData.append(np.max(self.nnOutput[frame, :, :]))
            self.maxPos.append(list(zip(*np.where(self.nnOutput[frame] == outputData[-1]))))
            self.progress.setValue(frame)
            self.app.processEvents()

        self.LoadingStatusLabel.setText('Resize the original frames to fit the output')
        print(postSize)
        image_drpOrigScaled = np.zeros((self.image_drpOrig.shape[0], postSize, postSize))
        image_mitoOrigScaled = np.zeros((self.image_drpOrig.shape[0], postSize, postSize))
        for i in range(0, self.image_mitoOrig.shape[0]):
            image_drpOrigScaled[i] = transform.rescale(self.image_drpOrig[i], resizeParam)
            image_mitoOrigScaled[i] = transform.rescale(self.image_mitoOrig[i], resizeParam)
            self.progress.setValue(i)
            self.app.processEvents()
        self.image_drpOrig = np.array(image_drpOrigScaled)
        self.image_mitoOrig = np.array(image_mitoOrigScaled)
        print(self.image_mitoOrig.shape)

        # Make a rolling mean of the output Data
        # N = 10
        # outputDataSmooth = np.ones(len(outputData))
        # outputDataSmooth[0:N] = np.ones(N)*np.mean(outputData[0:N])
        # for x in range(N, len(outputData)):
        #     outputDataSmooth[x] = np.sum(outputData[x-N:x])/N
        self.app.processEvents()
        if self.mode == 'stack':
            self.outputPlot.deleteRects()
            self.outputPlot.frames.setData([])
            self.outputPlot.nnframeScatter.setData([])
            self.outputPlot.nnline.setData(outputData)
            self.outputPlot.scatter.setData(range(0, len(outputData)), outputData)
        else:
            self.LoadingStatusLabel.setText('Getting the timing data')
            self.outputPlot.loadData(os.path.dirname(fname[0]), self.progress, self.app)
            self.app.processEvents()
            for i in range(-1, 6):
                self.outputPlot.inc = i
                self.outputPlot.updatePlot()

        self.frameSlider.setMaximum(self.nnOutput.shape[0]-1)

        self.refreshGradients()
        self.onTimer()
        self.LoadingStatusLabel.setText('Done')
        self.viewer_Proc.vb.setRange(xRange=(0, postSize), yRange=(0, postSize))

    def loadModel(self):
        self.LoadingStatusLabel.setText('Loading Model')
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/',
            "Keras models (*.h5)")
        print(fname[0])
        self.model = keras.models.load_model(fname[0], compile=True)
        self.LoadingStatusLabel.setText('Done')

    def orderChange(self):
        if self.order == 0:
            self.order = 1
            orderStr = 'order: Drp first'
        else:
            self.order = 0
            orderStr = 'order: Mito first'
        print(self.order)
        self.orderButton.setText(orderStr)

    def onTimer(self):
        i = self.frameSlider.value()
        self.imageMitoOrig.setImage(self.image_mitoOrig[i])
        self.imageDrpOrig.setImage(self.image_drpOrig[i])
        self.imageMitoProc.setImage(self.mitoDataFull[i])
        self.imageDrpProc.setImage(self.drpDataFull[i])
        self.imageNN.setImage(self.nnOutput[i])
        self.currentFrameLabel.setText(str(i))
        if self.mode == 'stack':
            self.frameLine.setValue(i)
        else:
            self.frameLine.setValue(self.outputPlot.elapsed[i])
        self.viewer_Orig.cross.setPosition([self.maxPos[i][0]])
        self.viewer_Proc.cross.setPosition([self.maxPos[i][0]])
        self.viewer_nn.cross.setPosition([self.maxPos[i][0]])

    def refreshGradients(self):
        self.viewer_Orig.setImage(self.image_mitoOrig[0], 1)
        self.viewer_Orig.setImage(self.image_drpOrig[0], 0)
        self.viewer_Proc.setImage(self.mitoDataFull[0], 1)
        self.viewer_Proc.setImage(self.drpDataFull[0], 0)
        self.viewer_nn.setImage(self.nnOutput[0], 0)

    def startTimer(self):
        self.timer.start()

    def stopTimer(self):
        self.timer.stop()

    def nextFrame(self):
        i = self.frameSlider.value()
        self.frameSlider.setValue(i + 1)
        self.onTimer()

    def prevFrame(self):
        i = self.frameSlider.value()
        self.frameSlider.setValue(i - 1)
        self.onTimer()

def main():
    app = QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    stackViewer = MultiPageTIFFViewerQt(app)

    stackViewer.showMaximized()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
