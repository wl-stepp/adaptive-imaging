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

import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import os
import glob
from binOutput import write_bin
from tensorflow import keras
import tensorflow as tf
import re  # Regular expression module
from PIL import Image
from skimage import io, exposure, filters

from datetime import datetime
from matplotlib.axes import Axes
from NNfeeder import prepareNNImages, prepareNNImages_01, prepareNNImages_02
from imageTiles import getTilePositions_v2

if __name__ == "__main__":

    # Setting for the watchdogs
    patterns = ["*.tif", "*.tiff"]
    ignore_patterns = ["*.txt"]
    ignore_directories = True
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns,
                                                   ignore_directories,
                                                   case_sensitive)

    # Loading the NN model
    nnImageSize = 128
    model_path = 'E:/Watchdog/SmartMito/model_Dora.h5'
    model = keras.models.load_model(model_path, compile=True)

    # Prep some structure to hold data

    # Set iSIM specific values
    pixelCalib = 56  # nm per pixel
    sig = 121.5/81  # in pixel
    resizeParam = pixelCalib/81  # no unit
    # Preprocess the images

    inputSize = 708
    inputSize = round(inputSize*resizeParam)

    # Figure for the histogram
    figHist = plt.figure()
    axHist = plt.axes()
    pltHist = axHist.plot([1, 2, 1])

    # Figure for the outputs
    fig = plt.figure(facecolor='black', tight_layout=True)

    data = np.random.randint(10, size=[inputSize, inputSize])
    im = []
    ax = []
    lines = []

    ax.append(fig.add_subplot(1, 3, 1))
    im.append(plt.imshow(data, vmax=5, cmap='Greys_r'))  # scaling

    # mito input display
    ax.append(fig.add_subplot(1, 3, 2))
    im.append(plt.imshow(data, vmax=255, alpha=1))
    ax.append(fig.add_subplot(1, 3, 3))
    im.append(plt.imshow(data, vmax=255, cmap='hot'))

    positions = getTilePositions_v2(
        np.ones((inputSize, inputSize)), nnImageSize)
    setWidth = 0.5
    setColor = [0.7, 0.7, 0.7]
    for axes in ax:
        lines.append(Axes.axhline(axes, y=inputSize - positions['stitch'],
                                  linewidth=setWidth, color=setColor))
        lines.append(Axes.axvline(axes, x=inputSize - positions['stitch'],
                                  linewidth=setWidth, color=setColor))
        for x in positions['px']:
            lines.append(Axes.axhline(axes, y=x[0] + positions['stitch'],
                                      linewidth=setWidth, color=setColor))
            lines.append(Axes.axvline(axes, x=x[1] + positions['stitch'],
                                      linewidth=setWidth, color=setColor))

    plt.pause(0.1)
    time.sleep(0.1)

    global frameNumOld, inputSizeOld
    frameNumOld = 100
    inputSizeOld = 0
    outputDataFull = np.zeros([512, 512])


def on_created(event):
    pass


def on_deleted(event):
    pass


def on_modified(event):
    global frameNumOld, inputSizeOld
    global outputDataFull, mitoDataFull, drpDataFull
    global outputHistogram
    size = os.path.getsize(event.src_path)

    # Extract the frame number from the filename
    splitStr = re.split(r'img_channel\d+_position\d+_time',
                        os.path.basename(event.src_path))
    splitStr = re.split(r'_z\d+', splitStr[1])
    frameNum = int(splitStr[0])

    # Only go on if frame number is odd and this frame was not analysed yet
    if frameNum % 2 and not frameNum == frameNumOld:
        pass
    else:
        return

    # Get what on_created wrote to the binary
    file = open(os.path.join(
        os.path.dirname(model_path), 'binary_output.dat'), mode='rb')
    content = file.read()
    file.close()
    # Skip file if a newer one is already in the folder
    if frameNum >= len(content)-1:
        pass
    else:
        print(int((frameNum-1)/2), ' passed because newer file is there')
        return

    # Construct the mito path
    # DO THIS IN A NICER WAY! that is not required to have that exact format
    mitoFile = 'img_channel000_position000_time' \
        + str((frameNum-1)).zfill(9) + '_z000.tiff'
    mito_path = os.path.join(os.path.dirname(event.src_path), mitoFile)

    # Mito is already written, so check size
    mitoSize = os.path.getsize(mito_path)

    if size > mitoSize*0.95:
        global im, ax, lines

        print(int((frameNum-1)/2))
        # print('folder search', int(round((t2 - t1)*1000)))
        # Read the mito image first, as it should already be written
        mitoFull = io.imread(mito_path)

        # Now read the drp file, should really be done writing
        # This takes 15 to 30 ms for a 512x512 16bit tif from lebpc34
        drpFull = io.imread(event.src_path)

        # If this is the first frame, get the parameters for contrast
        inputSize = mitoFull.shape[0]
        print(frameNum)
        if frameNum == 1 and not inputSize == inputSizeOld:
            inputSize = round(drpFull.shape[0]*resizeParam) \
                if not inputSize == 128 else 128
            outputDataFull = np.zeros([inputSize, inputSize])
            outputDataThresh = np.zeros([inputSize, inputSize])
            mitoDataFull = np.zeros([inputSize, inputSize])
            drpDataFull = np.zeros([inputSize, inputSize])
            outputHistogram = []
            # Figure for the outputs
            data = np.random.randint(10, size=[inputSize, inputSize])
            im[0] = ax[0].imshow(data, vmax=80, cmap='Greys_r')  # scaling
            # mito input display
            im[1] = ax[1].imshow(data, vmax=255, alpha=1)
            im[2] = ax[2].imshow(data, vmax=255, cmap='hot')

            positions = getTilePositions_v2(
                np.ones((inputSize, inputSize)), nnImageSize)
            print('overlap is ', positions['overlap'])
            for line in lines:
                line.remove()

            lines = []
            setWidth = 0.5
            setC = [0.7, 0.7, 0.7]
            for axes in ax:
                lines.append(Axes.axhline(axes,
                                          y=inputSize - positions['stitch'],
                                          linewidth=setWidth, color=setC))
                lines.append(Axes.axvline(axes,
                                          x=inputSize - positions['stitch'],
                                          linewidth=setWidth, color=setC))
                for x in positions['px']:
                    lines.append(Axes.axhline(axes,
                                              y=x[0] + positions['stitch'],
                                              linewidth=setWidth, color=setC))
                    lines.append(Axes.axvline(axes,
                                              x=x[1] + positions['stitch'],
                                              linewidth=setWidth, color=setC))
            print('Reinitialized plot')

        # Preprocess the data and make tiles if necessary
        inputData, positions = prepareNNImages(mitoFull, drpFull, nnImageSize)

        # Calculate the prediction on the full batch of images
        output_predict = model.predict_on_batch(inputData)  # , training=True)

        # Stitch the tiles back together (~2ms 512x512)
        i = 0
        stitch = positions['stitch']
        stitch1 = None if stitch == 0 else -stitch
        for position in positions['px']:
            outputDataFull[position[0]+stitch:position[2]-stitch,
                           position[1]+stitch:position[3]-stitch] = \
                output_predict[i, stitch:stitch1, stitch:stitch1, 0]
            mitoDataFull[position[0]+stitch:position[2]-stitch,
                         position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 0]
            drpDataFull[position[0]+stitch:position[2]-stitch,
                        position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 1]

            i = i + 1

        # OUTPUT Calculation
        # print('                   ', int(round(np.max(outputDataFull))))
        approach = 3
        if approach == 1:
            mask = outputDataFull > 10
            outputDataThresh = np.zeros_like(mask).astype(int)
            outputDataThresh[mask] = outputDataFull[mask]
            output = int(round(np.sum(outputDataThresh)))
        elif approach == 2:
            n = 4
            output = 0
            for i in range(0, n):
                output = output + np.max(outputDataFull)
                maxX = np.argmax(outputDataFull, axis=0)
                maxY = np.argmax(outputDataFull, axis=1)
                outputDataFull[maxX, maxY] = 0
            outputDataThresh = outputDataFull
            output = round(output)
        elif approach == 3:
            output = int(round(np.max(outputDataFull)))
            outputDataThresh = outputDataFull

        # ***** remove this for real use
        im[0].set_data(outputDataThresh)
        im[2].set_data(mitoDataFull)
        im[1].set_data(drpDataFull)
        all_lines = []

        # output = frameNum+1
        write_bin(output, 0)

        lengthCache = 10
        if len(outputHistogram) > lengthCache:
            x = np.arange(frameNum-lengthCache-1, frameNum)
            outputHistogram = outputHistogram[1:]

        else:
            x = np.arange(0, len(outputHistogram)+1)
        outputHistogram.append(output)
        print(len(x), len(outputHistogram))
        pltHist[0].set_data(x, outputHistogram)
        axHist.relim()
        axHist.autoscale_view(True, True, True)
        tend = time.perf_counter()

        frameNumOld = frameNum
        inputSizeOld = inputSize
        print('output generated   ', int(output), '\n')


def on_moved(event):
    pass


# Assign the event handlers
my_event_handler.on_created = on_created
my_event_handler.on_deleted = on_deleted
my_event_handler.on_modified = on_modified
my_event_handler.on_moved = on_moved

# More settings for the Watchdos
path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
go_recursively = True
my_observer = Observer()
my_observer.schedule(my_event_handler, path, recursive=go_recursively)


# Init the model by running it once
print('Initialize the model')
model(np.random.randint(10, size=[1, nnImageSize, nnImageSize, 2]))

my_observer.start()

print('All loaded, running...')
# Keep running and let image update until Strg + C
try:
    while True:
        plt.pause(0.1)
        time.sleep(0.1)
except KeyboardInterrupt:
    my_observer.stop()
    my_observer.join()
