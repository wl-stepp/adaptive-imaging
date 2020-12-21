'''
Input/Ouput module for data generated using the network_watchdog approach for adaptive temporal
sampling on the iSIM
'''

import glob
import json
import os
import re

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from skimage import io


def loadiSIMmetadata(folder):
    """  Load the information written by matlab about the generated DAQ signals for all in folder
    """
    delay = []
    for name in sorted(glob.glob(folder + '/iSIMmetadata*.txt')):
        data = np.genfromtxt(name)
        # set minimum cycle time
        if data[1] == 0:
            data[1] = 0.2
        numFrames = int(data[2])
        delay = np.append(delay, np.ones(numFrames)*data[1])
    print(delay)
    return delay


def loadTIF(folder):
    """ deprecated was used for testing Metadata import """
    file = 'img_channel000_position000_time000000001_z000.tif'
    filePath = folder + '/' + file

    tif = tifffile.TiffFile(filePath)
    info = tif.imagej_metadata['Info']  # pylint: disable=E1136  # pylint/issues/3139
    infoDict = json.loads(info)
    print(infoDict['ElapsedTime-ms'])
    tif.close()


def loadElapsedTime(folder, progress=None, app=None):
    """ get the Elapsed time for all files in folder """
    elapsed = []
    fileList = glob.glob(folder + '/img_*.tif')
    numFrames = int(len(fileList)/2)
    if progress is not None:
        progress.setRange(0, numFrames*2)
    i = 0
    for filePath in glob.glob(folder + '/img_*.tif'):
        with tifffile.TiffFile(filePath) as tif:
            mdInfo = tif.imagej_metadata['Info']  # pylint: disable=E1136  # pylint/issues/3139
            mdInfoDict = json.loads(mdInfo)
            elapsed.append(mdInfoDict['ElapsedTime-ms'])
    if app is not None:
        app.processEvents()
    # Progress the bar if available
    if progress is not None:
        progress.setValue(i)
    i = i + 1
    return elapsed


def loadNNData(folder):
    """ load the csv file written by NetworkWatchdog when processing live ATS data """
    file = 'output.txt'
    filePath = folder + '/' + file
    nnData = np.genfromtxt(filePath, delimiter=',')
    return nnData


def resaveNN(folder):
    """ Function to resave files that have been written in float format """
    for filePath in glob.glob(folder + '/img_*_nn.tiff'):
        img = tifffile.imread(filePath)
        newFile = filePath[:-5] + '_fiji.tiff'
        tifffile.imsave(newFile, img.astype(np.uint8))
        print(filePath)


def loadTifFolder(folder, resizeParam=1, order=0, progress=None) -> np.ndarray:
    """Function to load SATS data from a folder with the individual tif files written by
    microManager. Inbetween there might be neural network images that are also loaded into
    an array. Mainly used with NN_GUI_v2.py

    Args:
        folder ([type]): Folder with the data inside
        resizeParam (int, optional): Parameter that the loaded images are shrunken to later. This
        is also the size of the neural network images. Defaults to 1.
        order (int, optional): Order of the data. In the original case mito/Drp. Defaults to 0.
        progress ([type], optional): Link to a progress bar in the calling app. Defaults to None.
        app ([type], optional): Link to the calling app in order to keep it responsive using
        app.processEvents after each frame. Defaults to None.

    Returns:
        stack1 (numpy.array): stack of data depending on order
        stack2 (numpy.array): stack of data depending on order
        stackNN (numpy.array): stack of available network output, with zeros where no file was found
    """
    fileList = sorted(glob.glob(folder + '/img_*.tif'))
    numFrames = int(len(fileList)/2)
    pixelSize = io.imread(fileList[0]).shape
    stack1 = np.zeros((numFrames, pixelSize[0], pixelSize[1]))
    stack2 = np.zeros((numFrames, pixelSize[0], pixelSize[1]))
    postSize = round(pixelSize[1]*resizeParam)
    stackNN = np.zeros((numFrames,  postSize, postSize))
    frame = 0
    if progress is not None:
        progress.setRange(0, numFrames*2)
    for filePath in fileList:
        splitStr = re.split(r'img_channel\d+_position\d+_time', os.path.basename(filePath))
        splitStr = re.split(r'_z\d+', splitStr[1])
        frameNum = int(splitStr[0])
        if frameNum % 2:
            # odd
            stack1[frame] = io.imread(filePath)
            nnPath = filePath[:-8] + 'nn.tiff'
            try:
                stackNN[frame] = io.imread(nnPath)
            except FileNotFoundError:
                pass
            frame = frame + 1
        else:
            stack2[frame] = io.imread(filePath)

        # Progress the bar if available
        if progress is not None:
            progress.setValue(frameNum)

    if order == 0:
        stack1 = np.array(stack1)
        stack2 = np.array(stack2)
    else:
        stack1Save = stack1
        stack1 = np.array(stack2)
        stack2 = np.array(stack1Save)

    return stack1, stack2, stackNN


def loadTifStack(stack, order=0):
    """ Load a tif stack and deinterleave depending on the order (0 or 1) """
    start1 = order
    start2 = np.abs(order-1)
    imageMitoOrig = io.imread(stack)
    stack1 = imageMitoOrig[start1::2]
    stack2 = imageMitoOrig[start2::2]
    return stack1, stack2


def main():
    """ Main method testing loadTifFolder """
    folder = (
        'W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_9_nodecon')
    stack1 = loadTifFolder(folder, 512/741, 0)[0]
    print(stack1.shape)
    plt.imshow(stack1[5])
    plt.show()
    print('Done')


if __name__ == '__main__':
    main()
