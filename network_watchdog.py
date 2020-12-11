# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:32:38 2020

@author: stepp

Watchdog application for a tif file on the network that reads the image and
updates it in the associated figure window. Will be modified to make visual
output optional and adding a neural network that is used to mark interesting
events in the image.

Used in an environment where microManager running on another machine saves an
image to a NAS location, that should be observed, read and analysed. The binary
output will go to the same NAS and is expected to be read by a Matlab program
that runs on the same machine as microManager.
"""

import os
import re  # Regular expression module
import sys
from datetime import datetime

import numpy as np
import pyqtgraph as pg
# Qt display imports
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget
from skimage import io
# Tensorflow
from tensorflow import keras
from watchdog.events import PatternMatchingEventHandler
# Watchdog
from watchdog.observers import Observer

from binOutput import write_bin
# from imageTiles import getTilePositions_v2
# Own modules
from NNfeeder import prepareNNImages
from QtImageViewerMerge import QtImageViewerMerge


class NetworkWatchdog(QWidget):

    frameChanged = pyqtSignal([], [int])
    refreshGUI = pyqtSignal([], [int])
    reinitGUI = pyqtSignal(int)

    def __init__(self):

        # Setting for the watchdogs
        patterns = ["*.tif"]
        ignore_patterns = ["*.txt", "*.tiff"]
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
        self.viewerMerge = QtImageViewerMerge()
        self.imageDrp = self.viewerMerge.addImage()
        self.imageMito = self.viewerMerge.addImage()
        self.viewerNN = QtImageViewerMerge()
        self.imageNN = self.viewerNN.addImage()

        self.viewerOutput = pg.PlotWidget()
        self.outputPlot = self.viewerOutput.plot()
        self.viewerMerge.setLookupTable(self.imageDrp, 'viridis', False)
        self.viewerMerge.setLookupTable(self.imageMito, 'hot', True)
        self.viewerMerge.setLookupTable(self.imageNN, 'inferno', False)

        grid = QGridLayout(self)
        grid.addWidget(self.viewerMerge, 1, 0)
        grid.addWidget(self.viewerNN, 1, 1)
        grid.addWidget(self.viewerOutput, 0, 0, 1, 2)
        self.linePen = pg.mkPen(color='#AAAAAA')

        # Assign the event handlers
        my_event_handler.on_created = self.on_created
        my_event_handler.on_deleted = self.on_deleted
        my_event_handler.on_modified = self.on_modified
        my_event_handler.on_moved = self.on_moved
        self.refreshGUI.connect(self.refresh_GUI)
        self.reinitGUI.connect(self.reinitialize_GUI)
        self.viewerNN.vb.setXLink(self.viewerMerge.vb)
        self.viewerNN.vb.setYLink(self.viewerMerge.vb)

        # More settings for the Watchdos
        path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
        goRecursively = True
        self.my_observer = Observer()
        self.my_observer.schedule(
            my_event_handler, path, recursive=goRecursively)

        # Init the model by running it once
        print('Initialize the model')
        self.model(np.random.randint(
            10, size=[1, self.nnImageSize, self.nnImageSize, 2]))

        self.my_observer.start()

        print('All loaded, running...')

        #Init variables
        self.mitoDataFull = None
        self.drpDataFull = None
        self.outputHistogram = None
        self.outputX = None
        self.maxPos = None


        self.frameNumOld = 100
        self.inputSizeOld = 0
        self.folderNameOld = 'noFolder'
        self.outputDataFull = np.zeros([512, 512])
        self.imageViewer = [self.viewerMerge, self.viewerNN]
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
            + str((frameNum-1)).zfill(9) + '_z000.tif'
        nnFile = 'img_channel000_position000_time' \
            + str((frameNum)).zfill(9) + '_nn.tiff'
        mito_path = os.path.join(os.path.dirname(event.src_path), mitoFile)
        nn_path = os.path.join(os.path.dirname(event.src_path), nnFile)
        txtFile = os.path.join(os.path.dirname(event.src_path), 'output.txt')

        # Mito is already written, so check size
        mitoSize = os.path.getsize(mito_path)

        if not size > mitoSize*0.95:
            return

        print('frame', int((frameNum-1)/2))
        # Read the mito image first, as it should already be written
        # mitoFull = io.imread(mito_path)
        # drpFull = io.imread(event.src_path)
        # switched for Drp > green mito > red
        mitoFull = io.imread(event.src_path)
        drpFull = io.imread(mito_path)
        print(mitoFull.shape)

        # If this is the first frame, reinitialize the plot
        inputSize = mitoFull.shape[0]
        folderName = os.path.dirname(event.src_path)
        if not folderName == self.folderNameOld and not inputSize == self.inputSizeOld:
            inputSize = round(drpFull.shape[0]*self.resizeParam) if not \
                inputSize == 128 else 128
            self.outputDataFull = np.zeros([inputSize, inputSize])
            self.mitoDataFull = np.zeros([inputSize, inputSize])
            self.drpDataFull = np.zeros([inputSize, inputSize])
            # Redraw the lines
            self.reinitGUI.emit(inputSize)
            self.viewerNN.cross.show()
            print('Reinitialized plot')
        if not folderName == self.folderNameOld:
            # Make the txt file to write the output data to
            print(folderName)
            print('New txt file written')
            open(txtFile, 'w+')
            self.outputHistogram = []
            self.outputX = []

        # Preprocess the data and make tiles if necessary
        inputData, positions = prepareNNImages(
            mitoFull, drpFull, self.nnImageSize)
        print(inputData.shape)
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
        # Define where the position was found

        self.maxPos = list(zip(*np.where(self.outputDataFull == np.max(self.outputDataFull))))

        now_str = datetime.now()
        h = now_str.strftime("%H")
        m = now_str.strftime("%M")
        s = now_str.strftime("%S")
        ms = now_str.strftime("%f")
        print(ms)
        tX = (int(h)*60*60*1000 + int(m)*60*1000 + int(s)*1000 + int(ms[:2]))/60/60/1000
        print(tX)

        lengthCache = 100
        if len(self.outputHistogram) > lengthCache:
            self.outputX = self.outputX[1:]
            self.outputHistogram = self.outputHistogram[1:]
        self.outputHistogram.append(output)
        self.outputX.append(tX)  # (frameNum-1)/2)

        # Write output to binary for Matlab to read
        write_bin(output, 0)
        # Write output to txt file for later
        f = open(txtFile, 'a')
        f.write('%d, %d\n' % (frameNum, output))
        f.close()
        # Save the nn image
        io.imsave(nn_path, self.outputDataFull.astype(np.uint8), check_contrast=False)

        # Prepare images for plot and emit plotting event
        # self.mitoDisp = gray2qimage(self.mitoDataFull, normalize=True)
        # self.mitoDisp.setColorTable(self.lutHot)
        # self.drpDisp = gray2qimage(self.drpDataFull, normalize=True)
        # self.nnDisp = gray2qimage(self.outputDataFull, normalize=(0, 50))
        # np.mean(self.outputHistogram)*2 instead of 50?!
        # self.nnDisp.setColorTable(getQtcolormap('inferno'))
        self.refreshGUI.emit()

        # Prepare for next cycle
        self.frameNumOld = frameNum
        self.inputSizeOld = inputSize
        self.folderNameOld = folderName
        print('output generated   ', int(output), '\n')

    def refresh_GUI(self):
        self.imageMito.setImage(self.mitoDataFull)
        self.imageDrp.setImage(self.drpDataFull)
        self.imageNN.setImage(self.outputDataFull)
        self.outputPlot.setData(self.outputX, self.outputHistogram)
        self.viewerMerge.cross.setPosition(self.maxPos)
        self.viewerNN.cross.setPosition(self.maxPos)
        self.viewerOutput.enableAutoRange()

    def reinitialize_GUI(self, inputSize):
        # This was used to draw a grid on the ViewBox that shows the tiling. Will have to redo for
        # the new plotting environment
        # positions = getTilePositions_v2(
        #         np.ones((inputSize, inputSize)), self.nnImageSize)

        # for viewer in self.imageViewer:
        #     for line in viewer.lines:
        #         viewer.scene.removeItem(line)
        #     viewer.lines = []

        #     for line in viewer.lines:
        #         line.setPen(self.linePen)
        #         line.setZValue(100)
        self.viewerMerge.vb.setRange(xRange=(0, inputSize), yRange=(0, inputSize))

    def on_moved(self, event):
        pass

    def closeEvent(self):
        print('Watchdogs stopped')
        self.my_observer.stop()
        self.my_observer.join()


def main():
    app = QApplication(sys.argv)
    Watchdog = NetworkWatchdog()

    Watchdog.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
