'''
Input/Ouput module for data generated using the network_watchdog approach for adaptive temporal
sampling on the iSIM
'''

import glob
import json
import os
import re

import imageio
import numpy as np
import tifffile
import xmltodict
from skimage import io


def loadiSIMmetadata(folder):
    """  Load the information written by matlab about the generated DAQ signals for all in folder
    """
    delay = []
    for name in sorted(glob.glob(folder + '/iSIMmetadata*.txt')):
        txtFile = open(name)
        data = txtFile.readline().split('\t')
        data = [float(i) for i in data]
        txtFile.close()
        # set minimum cycle time
        if data[1] == 0:
            data[1] = 0.2
        numFrames = int(data[2])
        delay = np.append(delay, np.ones(numFrames)*data[1])
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
    """ get the Elapsed time for all .tif files in folder or from a stack """

    elapsed = []
    # Check for folder or stack mode
    # if not re.match(r'*.tif*', folder) is None:
    #     # get microManager elapsed times if available
    #     with tifffile.TiffFile(folder) as tif:
    #         for frame in range(0, len(tif.pages)):
    #             elapsed.append(
    #                 tif.pages[frame].tags['MicroManagerMetadata'].value['ElapsedTime-ms'])
    # else:
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


def loadTifFolder(folder, resizeParam=1, order=0, progress=None, cropSquare=True) -> np.ndarray:
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
        # if progress is not None:
            # progress.setValue(frameNum)
    stack1 = np.array(stack1)
    stack2 = np.array(stack2)


    # Check if this data is rectangular and crop if necessary
    if cropSquare:
        stack1 = cropToSquare(stack1)
        stack2 = cropToSquare(stack2)

    if order == 0:
        return stack1, stack2, stackNN
    else:
        return stack2, stack1, stackNN


def cropToSquare(stack):
    """ Crop a rectangular stack to a square. Should also work for single images """
    dimensions = len(stack.shape())
    diffShape = stack.shape[dimensions - 2] - stack.shape[dimensions - 1]
    if diffShape > 0:
        print('Shape is not rectangular going to crop')
        if dimensions == 3:
            stack[:, np.floor(diffShape/2):np.floor(stack.shape[dimensions-2]-diffShape/2), :]
        elif dimensions == 2:
            stack[:, np.floor(diffShape/2):np.floor(stack.shape[dimensions-2]-diffShape/2), :]
        else:
            print('Image has wrong dimensions')
    elif diffShape < 0:
        print('Shape is not rectangular going to crop')
        if dimensions == 3:
            stack[:, :, np.floor(diffShape/2):np.floor(stack.shape[dimensions-1]-diffShape/2)]
        elif dimensions == 2:
            stack[:, ;, np.floor(diffShape/2):np.floor(stack.shape[dimensions-1]-diffShape/2)]
        else:
            print('Image has wrong dimensions')

    return stack


def loadTifStack(stack, order=0, outputElapsed=False, cropSquare=True):
    """ Load a tif stack and deinterleave depending on the order (0 or 1) """
    start1 = order
    start2 = np.abs(order-1)
    imageMitoOrig = io.imread(stack)

    elapsed = []
    # get elapsed from tif file
    print(imageMitoOrig.shape)
    with tifffile.TiffFile(stack, fastij=False) as tif:
        mdInfo = tif.ome_metadata  # pylint: disable=E1136  # pylint/issues/3139
        if mdInfo is not None:
            mdInfoDict = xmltodict.parse(mdInfo)
            for frame in range(0, imageMitoOrig.shape[0]):
                elapsed.append(float(
                    mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))
            print(elapsed[-1])

    # Deinterleave data
    stack1 = imageMitoOrig[start1::2]
    elapsed1 = elapsed[start1::2]
    stack2 = imageMitoOrig[start2::2]
    elapsed2 = elapsed[start2::2]

    # Check if this data is rectangular and crop if necessary
    if cropSquare:
        stack1 = cropToSquare(stack1)
        stack2 = cropToSquare(stack2)

    return (stack1, stack2, elapsed1, elapsed2) if outputElapsed else (stack1, stack2)


def savegif(stack, times, fps):
    """ Save a gif that uses the right frame duration read from the files. This can be sped up
    using the fps option"""
    filePath = 'C:/Users/stepp/Documents/02_Raw/SmartMito/test.gif'
    times = np.divide(times, fps).tolist()
    print(stack.shape)
    imageio.mimsave(filePath, stack, duration=times)


def main():
    """ Main method testing savegif """

    from skimage import exposure
    folder = (
        "C:/Users/stepp/Documents/02_Raw/SmartMito/microM_test/"
        "201208_cell_Int0s_30pc_488_50pc_561_band_5")

    stack1 = loadTifFolder(folder, 512/741, 0)[0]
    times = loadElapsedTime(folder)
    times = times[0::2]
    times = (times - np.min(times))/1000
    times = np.diff(times).tolist()

    print(len(times))
    stack1 = exposure.rescale_intensity(
            stack1, (np.min(stack1), np.max(stack1)), out_range=(0, 255))
    stack1 = stack1.astype('uint8')
    savegif(stack1, times, 10)

    print('Done')


if __name__ == '__main__':
    main()
