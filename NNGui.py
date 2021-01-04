# -*- coding: utf-8 -*-
"""
This is a GUI that runs a neural network on the input data and displays
original data and the result in a scrollable form.
v1 was made using matplotlib. Unfortunately that is very slow, so here
I try to use pyqtgraph for faster performance
Sources:
https://stackoverflow.com/questions/41526832/pyqt5-qthread-signal-not-working-gui-freeze

Created on Mon Oct  5 12:18:48 2020

@author: stepp
"""
import os
import re
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox,
                             QLabel, QPlainTextEdit, QProgressBar, QPushButton,
                             QSlider, QWidget)
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
        self.frameSlider.setValue(0)
        self.frameSlider.setDisabled(True)
        self.prevFrameButton = QPushButton("<")
        self.nextFrameButton = QPushButton(">")

        # loadBox content: Buttons for load model and data
        self.modelButton = QPushButton("load model")
        self.dataButton = QPushButton("load data")
        self.dataButton.setDisabled(True)
        self.orderButton = QPushButton("order: Drp first")
        self.currentFrameLabel = QLabel('Frame')
        self.loadingStatusLabel = QLabel('')
        # progress bar for loading data
        self.progress = QProgressBar(self)
        self.log = QPlainTextEdit(self)

        self.outputPlot = SatsGUI()

        pen = pg.mkPen(color='#AAAAAA', style=Qt.DashLine)
        self.frameLine = pg.InfiniteLine(pos=0.5, angle=90, pen=pen)
        self.outputPlot.plotItem1.addItem(self.frameLine)

        # Connect functions to the interactive elements
        self.modelButton.clicked.connect(self.loadModel)
        self.dataButton.clicked.connect(self.loadDataThread)
        self.orderButton.clicked.connect(self.orderChange)
        self.prevFrameButton.clicked.connect(self.prevFrame)
        self.nextFrameButton.clicked.connect(self.nextFrame)

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
        gridBox.addWidget(self.log, 4, 0, 1, 3)

        self.threads = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        self.frameSlider.sliderPressed.connect(self.startTimer)
        self.frameSlider.sliderReleased.connect(self.stopTimer)

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
        self.folder = 'C:/Users/stepp/Documents/data_raw/SmartMito/'

        self.app = app
        self.order = 1
        self.linePen = pg.mkPen(color='#AAAAAA')

    def setProgressValue(self, progress):
        """ Set the progress value from the loading thread """
        self.progress.setValue(progress)

    def setLabelString(self, message):
        """ Set the Satus Label when called from loading thread """
        self.loadingStatusLabel.setText(message)

    def setLogString(self, message):
        """ Append to the log signaled from loading thread"""
        self.log.appendPlainText(message)

    def receiveData(self, data):
        """ Receive the data from the loading worker """
        self.mode = data.mode
        self.imageMitoOrig = data.imageMitoOrig
        self.imageDrpOrig = data.imageDrpOrig
        self.mitoDataFull = data.mitoDataFull
        self.drpDataFull = data.drpDataFull
        self.nnOutput = data.nnOutput
        self.maxPos = data.maxPos
        self.frameSlider.setMaximum(self.nnOutput.shape[0]-1)
        self.refreshGradients()
        self.onTimer()
        self.viewerOrig.resetRanges()
        self.viewerProc.resetRanges()
        self.viewerNN.resetRanges()
        self.loadingStatusLabel.setText('Done')
        self.viewerProc.viewBox.setRange(xRange=(0, data.postSize), yRange=(0, data.postSize))

        # set up the progress bar
        self.frameSlider.setDisabled(False)

    def updateProgress(self, rangeMax):
        """ Set the range of the progress bar """
        self.progress.setMaximum(rangeMax)

    def loadDataThread(self):
        """ Create and start the thread to load data """
        worker = LoadingThread(self)
        thread = QThread()
        thread.setObjectName('Data Loader')
        self.threads.append((thread, worker))
        worker.moveToThread(thread)
        worker.change_progress.connect(self.setProgressValue)
        worker.setLabel.connect(self.setLabelString)
        worker.setLog.connect(self.setLogString)
        worker.loadingDone.connect(self.receiveData)
        worker.updateProgressRange.connect(self.updateProgress)
        thread.started.connect(worker.work)
        thread.start()

    def loadModel(self):
        """ Load a .h5 model generated using Keras """
        self.loadingStatusLabel.setText('Loading Model')
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', 'C:/Users/stepp/Documents/data_raw/SmartMito/',
            "Keras models (*.h5)")
        self.model = keras.models.load_model(fname[0], compile=True)
        self.loadingStatusLabel.setText('Done')
        self.log.appendPlainText(fname[0])
        self.dataButton.setDisabled(False)

    def orderChange(self):
        """ React to a press of the order button to read interleaved data into the right order """
        if self.order == 0:
            self.order = 1
            orderStr = 'order: Drp first'
        else:
            self.order = 0
            orderStr = 'order: Mito first'
        self.setLogString('Set ' + orderStr)
        self.orderButton.setText(orderStr)

    def onTimer(self):
        """ Reset the data in the GUI on the timer when button or slider is pressed """
        i = self.frameSlider.value()
        self.viewerOrig.setImage(self.imageMitoOrig[i], 1)
        self.viewerOrig.setImage(self.imageDrpOrig[i], 0)
        self.viewerProc.setImage(self.mitoDataFull[i], 1)
        self.viewerProc.setImage(self.drpDataFull[i], 0)
        self.viewerNN.setImage(self.nnOutput[i], 0)
        self.currentFrameLabel.setText(str(i))
        if self.mode == 'stack':
            self.frameLine.setValue(i)
        else:
            self.frameLine.setValue(self.outputPlot.elapsed[i])
        self.viewerOrig.cross.setPosition([self.maxPos[i][0]])
        self.viewerProc.cross.setPosition([self.maxPos[i][0]])
        self.viewerNN.cross.setPosition([self.maxPos[i][0]])

    def startTimer(self):
        """ start Timer when slider is pressed """
        self.timer.start()

    def stopTimer(self):
        """ stop timer when slider is released"""
        self.timer.stop()

    def refreshGradients(self):
        """ refresh the images when a LUT was changed in the popup window """
        self.viewerOrig.setImage(self.imageMitoOrig[0], 1)
        self.viewerOrig.setImage(self.imageDrpOrig[0], 0)
        self.viewerProc.setImage(self.mitoDataFull[0], 1)
        self.viewerProc.setImage(self.drpDataFull[0], 0)
        self.viewerNN.setImage(self.nnOutput[0], 0)

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


class LoadingThread(QObject):
    """ Extra Thread for loading data """
    change_progress = pyqtSignal(int)
    setLabel = pyqtSignal(str)
    loadingDone = pyqtSignal(QObject)
    setLog = pyqtSignal(str)
    updateProgressRange = pyqtSignal(int)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.mode = None
        self.imageDrpOrig = None
        self.imageMitoOrig = None
        self.nnOutput = None
        self.nnOutput = None
        self.mitoDataFull = None
        self.drpDataFull = None
        self.nnRecalculated = None
        self.postSize = None
        self.frameNum = None
        self.maxPos = []
        self.folder = None
        self.nnImageSize = 128
        self.pixelCalib = 56  # nm per pixel
        self.resizeParam = self.pixelCalib/81  # no unit

    @pyqtSlot()
    def work(self):
        """ Function used to start the loading in the thread load """
        self.setLog.emit('\n\nLoading thread started')
        self._getFolder(self.data)
        newData = self._loadData(self.data)
        self.loadingDone.emit(newData)

    def _getFolder(self, data):
        """Getting GUI folder input, determine data type and load accordingly
        order 0 is mito first"""
        fname = QFileDialog.getOpenFileName(QWidget(), 'Open file', data.folder)
        self.setLabel.emit('Loading images from files into arrays')
        if not re.match(r'img_channel\d+_position\d+_time', os.path.basename(fname[0])) is None:
            # Check for microManager file like name and if so load as files in folder
            self.setLog.emit("Folder mode")
            self.mode = 'folder'
            self.imageDrpOrig, self.imageMitoOrig, self.nnOutput = loadTifFolder(
                os.path.dirname(fname[0]), self.resizeParam, data.order, data.progress)
            # Save this to go back to when the user wants to load another file
            data.folder = os.path.dirname(os.path.dirname(fname[0]))
            # Save this to use in SATS_gui
            self.folder = os.path.dirname(fname[0])
            self.setLog.emit(os.path.dirname(fname[0]))
        else:
            # If not singular files in folder, load as interleaved stack
            self.mode = 'stack'
            self.setLog.emit("Stack mode")
            self.imageDrpOrig, self.imageMitoOrig = loadTifStack(fname[0], data.order)
            # Save this to go back to when the user wants to load another file
            data.folder = os.path.dirname(fname[0])
            self.folder = data.folder
            self.setLog.emit(fname[0])
        self.updateProgressRange.emit(self.imageDrpOrig.shape[0])

    def _loadData(self, data):
        """load tif stack or ATS folder into the GUI. Reports progress using a textbox and the
        progress bar.
        """

        self.change_progress.emit(0)

        self.setLog.emit('input shape {}'.format(self.imageMitoOrig.shape))
        # Do NN for all images
        self.frameNum = self.imageMitoOrig.shape[0]
        self.postSize = round(self.imageMitoOrig.shape[1]*self.resizeParam)
        self.nnOutput = np.zeros((self.frameNum, self.postSize, self.postSize))
        self.mitoDataFull = np.zeros_like(self.nnOutput)
        self.drpDataFull = np.zeros_like(self.nnOutput)
        outputData = []
        self.nnRecalculated = np.zeros(self.frameNum)

        self.setLabel.emit('Processing frames and running the network')
        for frame in range(0, self.imageMitoOrig.shape[0]):
            inputData, positions = prepareNNImages(
                self.imageMitoOrig[frame, :, :],
                self.imageDrpOrig[frame, :, :], self.nnImageSize)

            # Do the NN calculation if there is not already a file there
            if self.mode == 'folder' and np.max(self.nnOutput[frame]) > 0:
                nnDataPres = 1
            else:
                nnDataPres = 0
                outputPredict = data.model.predict_on_batch(inputData)
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
            self.change_progress.emit(frame)
            QApplication.processEvents()

        self.setLabel.emit('Resize the original frames to fit the output')
        self.setLog.emit('Size after network: {}x{}'.format(self.postSize, self.postSize))
        imageDrpOrigScaled = np.zeros((self.imageDrpOrig.shape[0], self.postSize, self.postSize))
        imageMitoOrigScaled = np.zeros((self.imageDrpOrig.shape[0], self.postSize, self.postSize))
        for i in range(0, self.imageMitoOrig.shape[0]):
            imageDrpOrigScaled[i] = transform.rescale(self.imageDrpOrig[i], self.resizeParam,
                                                      anti_aliasing=True, preserve_range=True)
            imageMitoOrigScaled[i] = transform.rescale(self.imageMitoOrig[i], self.resizeParam,
                                                       anti_aliasing=True, preserve_range=True)
            self.change_progress.emit(i)
            QApplication.processEvents()
        self.imageDrpOrig = np.array(imageDrpOrigScaled).astype(np.uint8)
        self.imageMitoOrig = np.array(imageMitoOrigScaled).astype(np.uint8)

        # Make a rolling mean of the output Data
        # N = 10
        # outputDataSmooth = np.ones(len(outputData))
        # outputDataSmooth[0:N] = np.ones(N)*np.mean(outputData[0:N])
        # for x in range(N, len(outputData)):
        #     outputDataSmooth[x] = np.sum(outputData[x-N:x])/N
        QApplication.processEvents()
        data.outputPlot.deleteRects()
        data.outputPlot.frames.setData([])
        data.outputPlot.nnframeScatter.setData([])
        if self.mode == 'stack':
            data.outputPlot.nnline.setData(outputData)
            data.outputPlot.scatter.setData(range(0, len(outputData)), outputData)
        else:
            self.setLabel.emit('Getting the timing data')
            data.outputPlot.loadData(self.folder, data.progress, data.app)
            QApplication.processEvents()
            for i in range(-1, 6):
                data.outputPlot.inc = i
                data.outputPlot.updatePlot()

        return self


def main():
    """ main method to run and display the NNGui """
    app = QApplication(sys.argv)
    # setAttribute(QtCore.Qt.AA_Use96Dpi)
    stackViewer = NNGui(app)

    stackViewer.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
