# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:32:38 2020

@author: stepp

Watchdog application for a tif file on the network that reads the image and
updates it in the associated figure window. Will be modified to make visual
output optional and adding a neural network that is used to mark interesting
events in the image.

Used in an environment where microManager running on an other machine saves an
image to a NAS location, that should be observed, read and analysed. The binary
output will go to the same NAS and is expected to be read by a Matlab program
that runs on the same machine as microManager.
"""
# Standard
import time
import numpy as np
import os
import glob
import re  # Regular expression module
from datetime import datetime
import sys

# Watchdog
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# Plotting
from matplotlib import cm
from skimage import io, exposure, filters

# Tensorflow
from tensorflow import keras
import tensorflow as tf
# from PIL import Image

# Own modules
from NNfeeder import prepareNNImages, prepareNNImages_01, prepareNNImages_02
from imageTiles import getTilePositions_v2
from binOutput import write_bin

# Qt display imports
from PyQt5.QtCore import Qt, pyqtSignal, QT_VERSION_STR, QTimer
from PyQt5.QtWidgets import QWidget, QSlider, QPushButton, QLabel,\
    QGridLayout, QFileDialog, QProgressBar, QGroupBox, QApplication
from PyQt5.QtGui import QColor, QBrush, QPen, QMovie, qRgb, QImage
import pyqtgraph as pg
from QtImageViewer import QtImageViewer
from qimage2ndarray import array2qimage, gray2qimage


class NetworkWatchdog(QWidget):

    frameChanged = pyqtSignal([], [int])
    refreshGUI = pyqtSignal([], [int])
    reinitGUI = pyqtSignal(int)

    def __init__(self, app):

        # Setting for the watchdogs
        patterns = ["*.tiff"]
        ignore_patterns = ["*.txt", "*.tif"]
        ignore_directories = True
        case_sensitive = True
        my_event_handler = PatternMatchingEventHandler(
            patterns, ignore_patterns, ignore_directories, case_sensitive)

        # Loading the NN model
        self.nnImageSize = 128
        if os.environ['COMPUTERNAME'] == 'LEBPC20':
            self.modelPath = 'E:/Watchdog/SmartMito/model_Dora.h5'
        elif os.environ["COMPUTERNAME"] == 'LEBPC34':
            self.modelPath = (
                'C:/Users/stepp/Documents/data_raw/SmartMito/model_Dora.h5')
        self.model = keras.models.load_model(self.modelPath, compile=True)
        # Prep some structure to hold data

        # Set iSIM specific values
        pixelCalib = 56  # nm per pixel
        self.sig = 121.5/81  # in pixel
        self.resizeParam = pixelCalib/81  # no unit
        # Preprocess the images

        # Figure for the outputs
        QWidget.__init__(self)
        self.viewerMito = QtImageViewer()
        self.viewerDrp = QtImageViewer()
        self.viewerNN = QtImageViewer()
        self.viewerOutput = pg.PlotWidget()
        self.outputPlot = self.viewerOutput.plot()

        grid = QGridLayout(self)
        grid.addWidget(self.viewerMito, 0, 0)
        grid.addWidget(self.viewerDrp, 0, 1)
        grid.addWidget(self.viewerNN, 1, 0)
        grid.addWidget(self.viewerOutput, 1, 1)
        self.linePen = pg.mkPen(color='#AAAAAA')

        # Get colormaps into format for QImages
        colormap = cm.get_cmap("hot")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.lutHot = [qRgb(i[0], i[1], i[2]) for i in lut]
        colormap = cm.get_cmap("inferno")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        self.lutInferno = [qRgb(i[0], i[1], i[2]) for i in lut]

        # Assign the event handlers
        my_event_handler.on_created = self.on_created
        my_event_handler.on_deleted = self.on_deleted
        my_event_handler.on_modified = self.on_modified
        my_event_handler.on_moved = self.on_moved
        self.refreshGUI.connect(self.refresh_GUI)
        self.reinitGUI.connect(self.reinitialize_GUI)

        # More settings for the Watchdos
        path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
        go_recursively = True
        self.my_observer = Observer()
        self.my_observer.schedule(
            my_event_handler, path, recursive=go_recursively)

        # Init the model by running it once
        print('Initialize the model')
        self.model(np.random.randint(
            10, size=[1, self.nnImageSize, self.nnImageSize, 2]))

        self.my_observer.start()

        print('All loaded, running...')

        self.frameNumOld = 100
        self.inputSizeOld = 0
        self.outputDataFull = np.zeros([512, 512])
        self.imageViewer = [self.viewerMito, self.viewerDrp, self.viewerNN]
        for viewer in self.imageViewer:
            viewer.lines = []

    def on_created(self, event):
        pass

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        size = os.path.getsize(event.src_path)

        # Extract the frame number from the filename
        splitStr = re.split(r'img_channel\d+_position\d+_time',
                            os.path.basename(event.src_path))
        splitStr = re.split(r'_z\d+', splitStr[1])
        frameNum = int(splitStr[0])

        # Only go on if frame number is odd and this frame was not analysed yet
        if not frameNum % 2 or frameNum == self.frameNumOld:
            return

        # check framNumwrite.py output from the binary
        file = open(os.path.join(
            os.path.dirname(self.modelPath), 'binary_output.dat'), mode='rb')
        content = file.read()
        file.close()

        # Skip file if a newer one is already in the folder
        if not frameNum >= len(content)-1:
            print(int((frameNum-1)/2), ' passed because newer file is there')
            return

        # Construct paths
        # DO THIS IN A NICER WAY! not required to have that exact format
        mitoFile = 'img_channel000_position000_time' \
            + str((frameNum-1)).zfill(9) + '_z000.tiff'
        nnFile = 'img_channel000_position000_time' \
            + str((frameNum)).zfill(9) + 'nn.tif'
        mito_path = os.path.join(os.path.dirname(event.src_path), mitoFile)
        nn_path = os.path.join(os.path.dirname(event.src_path), nnFile)
        txtFile = os.path.join(os.path.dirname(event.src_path), 'output.txt')

        # Mito is already written, so check size
        mitoSize = os.path.getsize(mito_path)

        if not size > mitoSize*0.95:
            return

        print('frame', int((frameNum-1)/2))
        # Read the mito image first, as it should already be written
        mitoFull = io.imread(mito_path)
        drpFull = io.imread(event.src_path)

        # If this is the first frame, reinitialize the plot
        inputSize = mitoFull.shape[0]
        if frameNum == 1 and not inputSize == self.inputSizeOld:
            inputSize = round(drpFull.shape[0]*self.resizeParam) if not \
                inputSize == 128 else 128
            self.outputDataFull = np.zeros([inputSize, inputSize])
            self.mitoDataFull = np.zeros([inputSize, inputSize])
            self.drpDataFull = np.zeros([inputSize, inputSize])
            self.outputHistogram = []
            # Redraw the lines
            self.reinitGUI.emit(inputSize)
            print('Reinitialized plot')
        elif frameNum == 1:
            # Make the txt file to write the output data to
            print('New txt file written')
            open(txtFile, 'w+')

        # Preprocess the data and make tiles if necessary
        inputData, positions = prepareNNImages(
            mitoFull, drpFull, self.nnImageSize)

        # Calculate the prediction on the full batch of images
        output_predict = self.model.predict_on_batch(inputData)

        # Stitch the tiles back together (~2ms 512x512)
        i = 0
        stitch = positions['stitch']
        stitch1 = None if stitch == 0 else -stitch
        for position in positions['px']:
            self.outputDataFull[position[0]+stitch:position[2]-stitch,
                                position[1]+stitch:position[3]-stitch] = \
                output_predict[i, stitch:stitch1, stitch:stitch1, 0]
            self.mitoDataFull[position[0]+stitch:position[2]-stitch,
                              position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 0]
            self.drpDataFull[position[0]+stitch:position[2]-stitch,
                             position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 1]
            i = i + 1

        # OUTPUT Calculation
        approach = 3
        if approach == 1:
            mask = self.outputDataFull > 10
            outputDataThresh = np.zeros_like(mask).astype(int)
            outputDataThresh[mask] = self.outputDataFull[mask]
            output = int(round(np.sum(outputDataThresh)))
        elif approach == 2:
            n = 4
            output = 0
            for i in range(0, n):
                output = output + np.max(self.outputDataFull)
                maxX = np.argmax(self.outputDataFull, axis=0)
                maxY = np.argmax(self.outputDataFull, axis=1)
                self.outputDataFull[maxX, maxY] = 0
            outputDataThresh = self.outputDataFull
            output = round(output)
        elif approach == 3:
            output = int(round(np.max(self.outputDataFull)))
            outputDataThresh = self.outputDataFull

        lengthCache = 30
        if len(self.outputHistogram) > lengthCache:
            x = np.arange(frameNum-lengthCache-1, frameNum)
            self.outputHistogram = self.outputHistogram[1:]
        else:
            x = np.arange(0, len(self.outputHistogram)+1)
        self.outputHistogram.append(output)

        # Write output to binary for Matlab to read
        write_bin(output, 0)
        # Write output to txt file for later
        f = open(txtFile, 'a')
        f.write('%d, %d\n' % (frameNum, output))
        f.close()
        # Save the nn image
        io.imsave(nn_path, self.outputDataFull, check_contrast=False)

        # Prepare images for plot and emit plotting event
        self.mitoDisp = gray2qimage(self.mitoDataFull, normalize=True)
        self.mitoDisp.setColorTable(self.lutHot)
        self.drpDisp = gray2qimage(self.drpDataFull, normalize=True)
        self.nnDisp = gray2qimage(self.outputDataFull, normalize=(0, 50))
        # np.mean(self.outputHistogram)*2 instead of 50?!
        self.nnDisp.setColorTable(self.lutInferno)
        self.refreshGUI.emit()

        # Prepare for next cycle
        self.frameNumOld = frameNum
        self.inputSizeOld = inputSize
        print('output generated   ', int(output), '\n')

    def refresh_GUI(self, event=0):
        self.viewerMito.setImage(self.mitoDisp)
        self.viewerDrp.setImage(self.drpDisp)
        self.viewerNN.setImage(self.nnDisp)
        self.outputPlot.setData(self.outputHistogram)

    def reinitialize_GUI(self, inputSize):
        positions = getTilePositions_v2(
                np.ones((inputSize, inputSize)), self.nnImageSize)

        for viewer in self.imageViewer:
            for line in viewer.lines:
                viewer.scene.removeItem(line)
            viewer.lines = []
            viewer.lines.append(
                viewer.scene.addLine(
                                     inputSize - positions['stitch'],
                                     0,
                                     inputSize - positions['stitch'],
                                     inputSize))
            viewer.lines.append(
                viewer.scene.addLine(0,
                                     inputSize - positions['stitch'],
                                     inputSize,
                                     inputSize - positions['stitch']))

            for x in positions['px']:
                viewer.lines.append(
                    viewer.scene.addLine(x[0] + positions['stitch'],
                                         0,
                                         x[0] + positions['stitch'],
                                         inputSize))
                viewer.lines.append(
                    viewer.scene.addLine(0,
                                         x[1] + positions['stitch'],
                                         inputSize,
                                         x[1] + positions['stitch']))

            for line in viewer.lines:
                line.setPen(self.linePen)
                line.setZValue(100)

    def on_moved(self, event):
        pass

    def closeEvent(self, event):
        print('Watchdogs stopped')
        self.my_observer.stop()
        self.my_observer.join()


app = QApplication(sys.argv)
Watchdog = NetworkWatchdog(app)

Watchdog.show()
sys.exit(app.exec_())
