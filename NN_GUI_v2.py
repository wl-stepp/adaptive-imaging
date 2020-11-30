# -*- coding: utf-8 -*-
"""
This is a GUI that runs a neural network on the input data and displays
original data and the result in a scrollable form.
v1 was made using matplotlib. Unfortunately that is very slow, so here
I try to use pyqtgraph for faster performance

Created on Mon Oct  5 12:18:48 2020

@author: stepp
"""
from PyQt5.QtCore import Qt, pyqtSignal, QT_VERSION_STR, QTimer
from PyQt5.QtWidgets import QWidget, QSlider, QPushButton, QLabel,\
    QGridLayout, QFileDialog, QProgressBar, QGroupBox, QApplication
from PyQt5.QtGui import QColor, QBrush, QPen, QMovie
import pyqtgraph as pg
from skimage import io
from QtImageViewer import QtImageViewer
from qimage2ndarray import array2qimage
import sys
import time
import numpy as np
from NNfeeder import prepareNNImages
from imageTiles import getTilePositions_v2
from tensorflow import keras
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
        self.viewer_mitoOrig = QtImageViewer()
        self.viewer_drpOrig = QtImageViewer()
        self.viewer_mito = QtImageViewer()
        self.viewer_drp = QtImageViewer()
        self.viewer_nn = QtImageViewer()
        self.nnMaxPos = self.viewer_nn.scene.addEllipse(
            0, 0, 3, 3, pen=Qt.transparent,
            brush=QBrush(QColor(255, 0, 0)))
        self.nnMaxPos.setZValue(100)
        self.loadBox = QGroupBox()

        # Connect events for zooming in all output data
        self.viewer_mito.rightMouseButtonReleased.connect(self.mouseRelease)
        self.viewer_mito.rightMouseButtonDoubleClicked.connect(
            self.mouseDouble)

        # Slider and arrow buttons for frame traversal.
        self.sliderBox = QGroupBox()
        self.frameSlider = QSlider(Qt.Horizontal)
        self.prevFrameButton = QPushButton("<")
        self.nextFrameButton = QPushButton(">")

        # loadBox content: Buttons for load model and data
        self.modelButton = QPushButton("load model")
        self.dataButton = QPushButton("load data")
        self.currentFrameLabel = QLabel('Frame')
        # progress bar for loading data
        self.progress = QProgressBar(self)
        self.outputPlot = pg.PlotWidget()
        pen = pg.mkPen(color='#AAAAAA', style=Qt.DashLine)
        self.frameLine = pg.InfiniteLine(pos=0.5, angle=90, pen=pen)
        self.outputPlot.addItem(self.frameLine)

        # Connect functions to the interactive elements
        self.modelButton.clicked.connect(self.loadModel)
        self.dataButton.clicked.connect(self.loadData)
        # self.prevFrameButton.clicked.connect(self.prevFrame)
        # self.nextFrameButton.clicked.connect(self.nextFrame)
        self.frameSlider.sliderPressed.connect(self.startTimer)
        self.frameSlider.sliderReleased.connect(self.stopTimer)

        # Layout.
        grid = QGridLayout(self)
        grid.addWidget(self.viewer_mitoOrig, 0, 0)
        grid.addWidget(self.viewer_drpOrig, 0, 1)
        grid.addWidget(self.loadBox, 0, 2)
        grid.addWidget(self.viewer_mito, 1, 0)
        grid.addWidget(self.viewer_drp, 1, 1)
        grid.addWidget(self.viewer_nn, 1, 2)
        grid.addWidget(self.sliderBox, 2, 0, 1, 3)

        gridprogress = QGridLayout(self.sliderBox)
        gridprogress.addWidget(self.prevFrameButton, 0, 0)
        gridprogress.addWidget(self.frameSlider, 0, 1)
        gridprogress.addWidget(self.nextFrameButton, 0, 3)
        gridprogress.setContentsMargins(0, 0, 0, 0)

        gridBox = QGridLayout(self.loadBox)
        gridBox.addWidget(self.currentFrameLabel, 1, 0)
        gridBox.addWidget(self.progress, 1, 1)
        gridBox.addWidget(self.modelButton, 0, 0)
        gridBox.addWidget(self.dataButton, 0, 1)
        gridBox.addWidget(self.outputPlot, 2, 0, 1, 2)

        self.resize(1500, 1000)

        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        self.app = app

    def loadData(self):
        # load images, this goes into a button later
        self.progress.setValue(0)
        nnImageSize = 128
        pixelCalib = 56  # nm per pixel
        resizeParam = pixelCalib/81  # no unit

        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/')
        print(fname[0])
        image_mitoOrig = io.imread(fname[0])
        image_drpOrig = image_mitoOrig[1::2]
        image_mitoOrig = image_mitoOrig[0::2]

        print(image_mitoOrig.shape)
        # Do NN for all images
        frameNum = image_mitoOrig.shape[0]
        postSize = round(image_mitoOrig.shape[1]*resizeParam)
        nnOutput = np.zeros((frameNum, postSize, postSize))
        mitoDataFull = np.zeros_like(nnOutput)
        drpDataFull = np.zeros_like(nnOutput)
        outputData = []
        self.max_pos = []
        # set up the progress bar
        self.progress.setRange(0, frameNum-1)

        for frame in range(0, image_mitoOrig.shape[0]):
            inputData, positions = prepareNNImages(
                image_mitoOrig[frame, :, :],
                image_drpOrig[frame, :, :], nnImageSize)
            output_predict = self.model.predict_on_batch(inputData)
            i = 0
            st = positions['stitch']
            st1 = None if st == 0 else -st
            for p in positions['px']:
                nnOutput[frame, p[0]+st:p[2]-st, p[1]+st:p[3]-st] =\
                            output_predict[i, st:st1, st:st1, 0]
                mitoDataFull[frame, p[0]+st:p[2]-st, p[1]+st:p[3]-st] = \
                    inputData[i, st:st1, st:st1, 0]
                drpDataFull[frame, p[0]+st:p[2]-st, p[1]+st:p[3]-st] = \
                    inputData[i, st:st1, st:st1, 1]
                i += 1

            outputData.append(np.max(nnOutput[frame, :, :]))
            self.max_pos.append(np.where(nnOutput[frame, :, :] ==
                                np.max(nnOutput[frame, :, :])))
            self.progress.setValue(frame)
            self.app.processEvents()

        # Make a rolling mean of the output Data
        N = 10
        t1 = time.perf_counter()
        outputDataSmooth = np.ones(len(outputData))
        outputDataSmooth[0:N] = np.ones(N)*np.mean(outputData[0:N])
        for x in range(N, len(outputData)):
            outputDataSmooth[x] = np.sum(outputData[x-N:x])/N
        print("dt", time.perf_counter() - t1)

        # Make QImages and put in list to display later
        self.nnMovie = []
        self.mitoMovie = []
        self.drpMovie = []
        self.mitoOrigMovie = []
        self.drpOrigMovie = []
        for frame in range(0, nnOutput.shape[0]):
            self.nnMovie.append(
                array2qimage(nnOutput[frame, :, :],
                             normalize=(0, np.mean(outputData))))
            self.mitoMovie.append(
                array2qimage(mitoDataFull[frame, :, :], normalize=True))
            self.drpMovie.append(
                array2qimage(drpDataFull[frame, :, :], normalize=True))
            self.mitoOrigMovie.append(
                array2qimage(image_mitoOrig[frame, :, :], normalize=True))
            self.drpOrigMovie.append(
                array2qimage(image_drpOrig[frame, :, :], normalize=True))
            self.progress.setValue(frame)

        pen = pg.mkPen(color=(192, 251, 255), width=1)
        pen2 = pg.mkPen(color=(151, 0, 26), width=3)
        pen3 = pg.mkPen(color=(148, 155, 0), width=3)
        self.outputPlot.plot(outputData, pen=pen)
        self.outputPlot.plot(outputDataSmooth, pen=pen2)
        # self.outputPlot.plot(outputData-outputDataSmooth, pen=pen3)
        self.frameSlider.setMaximum(nnOutput.shape[0]-1)
        self.onTimer()

        print('Done with everything')

    def loadModel(self):
        folder = 'C:/Users/stepp/Documents/data_raw/SmartMito/'
        model_path = folder + 'model_Dora.h5'
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/',
            "Keras models (*.h5)")
        print(fname[0])
        self.model = keras.models.load_model(fname[0], compile=True)
        print('Model compiled')

    def onTimer(self):
        i = self.frameSlider.value()
        self.viewer_mito.setImage(self.mitoMovie[i])
        self.viewer_drp.setImage(self.drpMovie[i])
        self.viewer_drpOrig.setImage(self.drpOrigMovie[i])
        self.viewer_mitoOrig.setImage(self.mitoOrigMovie[i])
        self.viewer_nn.setImage(self.nnMovie[i])
        self.currentFrameLabel.setText(str(i))
        self.frameLine.setValue(i)
        self.nnMaxPos.setPos(self.max_pos[i][1][0]-1.5,
                             self.max_pos[i][0][0]-1.5)
        self.viewer_mito.updateViewer()

    def startTimer(self, i=0):
        self.timer.start()

    def stopTimer(self, i=0):
        self.timer.stop()

    def mouseRelease(self, x, y):
        for remoteViewer in [self.viewer_drp, self.viewer_nn]:
            if remoteViewer.canZoom:
                viewBBox = remoteViewer.zoomStack[-1] if\
                    len(remoteViewer.zoomStack) else remoteViewer.sceneRect()
                selectionBBox = self.viewer_mito.scene.selectionArea().\
                    boundingRect().intersected(viewBBox)
                if selectionBBox.isValid() and (selectionBBox != viewBBox):
                    remoteViewer.zoomStack.append(selectionBBox)
                    remoteViewer.updateViewer()

    def mouseDouble(self, x, y):
        for remoteViewer in [self.viewer_drp, self.viewer_nn]:
            if remoteViewer.canZoom:
                remoteViewer.zoomStack = []
                remoteViewer.updateViewer()


app = QApplication(sys.argv)
# app.setAttribute(QtCore.Qt.AA_Use96Dpi)
stackViewer = MultiPageTIFFViewerQt(app)

stackViewer.show()
sys.exit(app.exec_())
