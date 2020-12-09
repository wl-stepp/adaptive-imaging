'''
Input/Ouput module for data generated using the network_watchdog approach for adaptive temporal
sampling on the iSIM
'''

import glob
import numpy as np
import os
import tifffile
import json
from skimage import io
import re
from matplotlib import pyplot as plt


def loadiSIMmetadata(folder):
    delay = []
    for name in sorted(glob.glob(folder + '/iSIMmetadata*.txt')):
        data = np.genfromtxt(name)
        # set minimum cycle time
        if data[1] == 0:
            data[1] = 0.2
        numFrames = int(data[2])
        delay = np.append(delay, np.ones(numFrames)*data[1])

    return delay


def loadTIF(folder):
    file = 'img_channel000_position000_time000000001_z000.tif'
    filePath = folder + '/' + file

    with tifffile.TiffFile(filePath) as tif:
        data3 = tif.imagej_metadata
        info = data3['Info']
        dic = tif.__dict__
        info_dict = json.loads(info)
        print(info_dict['ElapsedTime-ms'])

        tif.close()


def loadElapsedTime(folder):
    elapsed = []
    for filePath in glob.glob(folder + '/img_*.tif'):
        with tifffile.TiffFile(filePath) as tif:
            md = tif.imagej_metadata
            mdInfo = md['Info']
            mdInfoDict = json.loads(mdInfo)
            elapsed.append(mdInfoDict['ElapsedTime-ms'])
    return elapsed


def loadNNData(folder):
    file = 'output.txt'
    filePath = folder + '/' + file
    nnData = np.genfromtxt(filePath, delimiter=',')
    return nnData


def resaveNN(folder):
    for filePath in glob.glob(folder + '/img_*_nn.tiff'):
        img = tifffile.imread(filePath)
        new_file = filePath[:-5] + '_fiji.tiff'
        tifffile.imsave(new_file, img.astype(np.uint8))
        print(filePath)


def loadTifFolder(folder, resizeParam=1, order=0, progress=None, app=None):
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
    FileList = sorted(glob.glob(folder + '/img_*.tif'))
    numFrames = int(len(FileList)/2)
    pixelSize = io.imread(FileList[0]).shape
    stack1 = np.zeros((numFrames, pixelSize[0], pixelSize[1]))
    stack2 = np.zeros((numFrames, pixelSize[0], pixelSize[1]))
    postSize = round(pixelSize[0]*resizeParam)
    stackNN = np.zeros((numFrames,  postSize, postSize))
    frame = 0
    progress.setRange(0, numFrames*2)
    for filePath in FileList:
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

        if app is not None:
            app.processEvents()
        # Progress the bar if available
        if progress is not None:
            progress.setValue(frameNum)

    if order == 0:
        stack1 = np.array(stack1)
        stack2 = np.array(stack2)
    else:
        stack1_save = stack1
        stack1 = np.array(stack2)
        stack2 = np.array(stack1_save)
    print(stack1.shape)
    return stack1, stack2, stackNN


def loadTifStack(stack, order=0):
    start1 = order
    start2 = np.abs(order-1)
    image_mitoOrig = io.imread(stack)
    stack1 = image_mitoOrig[start1::2]
    stack2 = image_mitoOrig[start2::2]
    return stack1, stack2


if __name__ == '__main__':

    folder = (
        'W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_9')
    mito, drp, stackNN = loadTifFolder(folder, 0)
    print(mito.shape)
    plt.imshow(mito[5])
    plt.show()
    print('Done')
