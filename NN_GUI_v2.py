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
import pyqtgraph as pg
from skimage import io
from QtImageViewer import QtImageViewer
import qimage2ndarray
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
        self.timer.setInterval(40)

        self.app = app

    def loadData(self):
        # load images, this goes into a button later
        self.progress.setValue(0)
        folder = 'C:/Users/stepp/Documents/data_raw/SmartMito/'
        nnImageSize = 128
        pixelCalib = 56  # nm per pixel
        resizeParam = pixelCalib/81  # no unit

        mito_file = folder + 's6_c12_p0_mito.tif'
        drp_file = folder + 's6_c12_p0_drp1.tif'
        self.image_mitoOrig = io.imread(mito_file).transpose(0, 2, 1)
        self.image_drpOrig = io.imread(drp_file).transpose(0, 2, 1)

        print(self.image_mitoOrig.shape)
        # Do NN for all images
        frameNum = self.image_mitoOrig.shape[0]
        postSize = round(self.image_mitoOrig.shape[1]*resizeParam)
        nnOutput = np.zeros((frameNum, postSize, postSize))
        mitoDataFull = np.zeros_like(nnOutput)
        drpDataFull = np.zeros_like(nnOutput)
        outputData = []
        # set up the progress bar
        self.progress.setRange(0, frameNum-1)

        for frame in range(0, self.image_mitoOrig.shape[0]):
            inputData, positions = prepareNNImages(
                self.image_mitoOrig[frame, :, :],
                self.image_drpOrig[frame, :, :], nnImageSize)
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
            self.progress.setValue(frame)
            self.app.processEvents()

        self.outputPlot.plot(outputData)
        self.image_nn = nnOutput
        self.image_mito = mitoDataFull
        self.image_drp = drpDataFull
        self.frameSlider.setMaximum(nnOutput.shape[0]-1)
        self.onTimer()

        print('Done with everything')

    def loadModel(self):
        folder = 'C:/Users/stepp/Documents/data_raw/SmartMito/'
        print('Starting the model up')
        model_path = folder + 'model_Dora.h5'
        self.model = keras.models.load_model(model_path, compile=True)
        print('Model compiled')

    def onTimer(self):
        i = self.frameSlider.value()
        qimage_mitoOrig = qimage2ndarray.array2qimage(
            self.image_mitoOrig[i, :, :], normalize=True)
        qimage_drpOrig = qimage2ndarray.array2qimage(
            self.image_drpOrig[i, :, :], normalize=True)
        qimage_drp = qimage2ndarray.array2qimage(
            self.image_drp[i, :, :], normalize=False)
        qimage_mito = qimage2ndarray.array2qimage(
            self.image_mito[i, :, :], normalize=False)
        qimage_nn = qimage2ndarray.array2qimage(
            self.image_nn[i, :, :], normalize=(0, 50))
        self.viewer_mito.setImage(qimage_mito)
        self.viewer_drp.setImage(qimage_drp)
        self.viewer_drpOrig.setImage(qimage_drpOrig)
        self.viewer_mitoOrig.setImage(qimage_mitoOrig)
        self.viewer_nn.setImage(qimage_nn)
        self.currentFrameLabel.setText(str(i))
        self.frameLine.setValue(i)
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
