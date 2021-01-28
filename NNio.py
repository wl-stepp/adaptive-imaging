'''
Input/Ouput module for data generated using the network_watchdog approach for adaptive temporal
sampling on the iSIM
'''

import glob
import json
import os
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import xmltodict
from PIL import Image
from skimage import io

import ImageTiles
import NNfeeder


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
            if mdInfo is None:
                mdInfo = tif.shaped_metadata[0]['Infos'] # pylint: disable=E1136
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
    dimensions = len(stack.shape)
    print(stack.shape)
    diffShape = stack.shape[dimensions - 2] - stack.shape[dimensions - 1]
    start = int(np.floor(abs(diffShape/2)))
    if diffShape > 0:
        print('Shape is not rectangular going to crop')
        end = int(np.floor(stack.shape[dimensions-2]-abs(diffShape/2)))
        if dimensions == 3:
            stack = stack[:, start:end, :]
        elif dimensions == 2:
            stack = stack[start:end, :]
        else:
            print('Image has wrong dimensions')
    elif diffShape < 0:
        print('Shape is not rectangular going to crop')
        end = int(np.floor(stack.shape[dimensions-1]-abs(diffShape/2)))
        if dimensions == 3:
            stack = stack[:, :, start:end]
        elif dimensions == 2:
            stack = stack[:, start:end]
        else:
            print('Image has wrong dimensions')

    return stack


def loadTifStack(stack, order=0, outputElapsed=False, cropSquare=True):
    """ Load a tif stack and deinterleave depending on the order (0 or 1). Also get the
    elapsed time on the images and give them back as list."""
    start1 = order
    start2 = np.abs(order-1)
    imageMitoOrig = io.imread(stack)

    print(imageMitoOrig.shape)
    if len(imageMitoOrig.shape) == 4:
        # This file probably has channels for the different data
        channels = imageMitoOrig.shape[1]
        stack1 = imageMitoOrig[:, start1, :, :]
        stack2 = imageMitoOrig[:, start2, :, :]
    else:
        channels = 1
        stack1 = imageMitoOrig[start1::2]
        stack2 = imageMitoOrig[start2::2]

    # This could be done also more flexible with a matrix or dict

    elapsed = []
    # get elapsed from tif file
    with tifffile.TiffFile(stack, fastij=False) as tif:
        mdInfo = tif.ome_metadata  # pylint: disable=E1136  # pylint/issues/3139
        # This should work for single series ome.tifs
        mdInfoDict = xmltodict.parse(mdInfo)
        for frame in range(0, imageMitoOrig.shape[0]*channels):
            # Check which unit the DeltaT is
            if mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit'] == 's':
                unitMultiplier = 1000
            else:
                unitMultiplier = 1
            elapsed.append(unitMultiplier*float(
                mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))

    elapsed1 = elapsed[start1::2]
    elapsed2 = elapsed[start2::2]

    # Check if this data is rectangular and crop if necessary
    if cropSquare:
        stack1 = cropToSquare(stack1)
        stack2 = cropToSquare(stack2)

    return (stack1, stack2, elapsed1, elapsed2) if outputElapsed else (stack1, stack2)


def loadTifStackElapsed(file, numFrames=None):
    """ this should ideally be added to loadElapsed """
    elapsed = []
    with tifffile.TiffFile(file) as tif:
        if numFrames is None:
            numFrames = len(tif.pages)
        mdInfo = tif.ome_metadata  # pylint: disable=E1136  # pylint/issues/3139
        # This should work for single series ome.tifs
        mdInfoDict = xmltodict.parse(mdInfo)
        for frame in range(0, numFrames):
            # Check which unit the DeltaT is
            if mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit'] == 's':
                unitMultiplier = 1000
            else:
                unitMultiplier = 1
            elapsed.append(unitMultiplier*float(
                mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))
    return elapsed


def savegif(stack, times, fps):
    """ Save a gif that uses the right frame duration read from the files. This can be sped up
    using the fps option"""
    filePath = 'C:/Users/stepp/Documents/02_Raw/SmartMito/test.gif'
    times = np.divide(times, fps).tolist()
    print(stack.shape)
    imageio.mimsave(filePath, stack, duration=times)


def extractTiffStack(file, frame, target):
    """ It should be possible to get the elapsed time also for a single image. Therefore especially
    when writing from ATSSim, I want to preserve both the OME metadata and the imagej metadata.
    Unfortunately tifffile can't write both at the same time. So the imagej metadata is transferred
    to the shaped_metadata. This also makes it easy to account for both possibilities when loading
    data elsewhere. """
    with tifffile.TiffFile(file, fastij=False) as tif:
        mdInfo = tif.ome_metadata
        ijInfo = tif.imagej_metadata

    image = tifffile.imread(file, pages=frame)
    mdInfoDict = xmltodict.parse(mdInfo)
    ijInfoDict = json.loads(ijInfo['Info'])  # pylint: disable=E1136  # pylint/issues/3139

    if mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit'] == 's':
        unitMultiplier = 1000
    else:
        unitMultiplier = 1

    elapsed = mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']*unitMultiplier

    ijInfoDict['ElapsedTime-ms'] = elapsed
    ijInfo['Info'] = json.dumps(ijInfoDict)  # pylint: disable=E1136  # pylint/issues/3139

    # Save the metadata for this page
    planeMD = mdInfoDict['OME']['Image']['Pixels']['Plane'][frame]
    # empty out the metadata for all the planes
    mdInfoDict['OME']['Image']['Pixels']['Plane'] = []
    # Set the old metadata as single object in a list
    mdInfoDict['OME']['Image']['Pixels']['Plane'] = [planeMD]

    # retranslate the dict to an xml string
    mdInfo = xmltodict.unparse(mdInfoDict)
    tifffile.imwrite(target, image, description=mdInfo.encode(encoding='UTF-8', errors='strict'),
                     metadata={'Info': ijInfo['Info']})  # pylint: disable=E1136


def calculateNNforStack(file, model=None):
    """ calculate neural network output for all files in a stack """
    dataOrder = 0 # 0 for drp/foci first, 1 for mito/structure first
    if model is None:
        from tensorflow import keras
        modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/model_Dora.h5'
        model = keras.models.load_model(modelPath, compile=True)

    # DrpOrig, MitoOrig, elapsed, _ = loadTifStack(file, dataOrder,  outputElapsed=True)

    nnPath = file[:-8] + '_nn.ome.tif'

    DrpOrig, MitoOrig = loadTifStack(file, dataOrder)

    # Prepare the Metadata for the new file by extracting the ome-metadata
    with tifffile.TiffFile(file, fastij=False) as tif:
        mdInfo = tif.ome_metadata.encode(encoding='UTF-8', errors='strict')
        #extract only the planes that was written to
        mdInfoDict = xmltodict.parse(mdInfo)
        mdInfoDict['OME']['Image']['Pixels']['Plane'] = mdInfoDict['OME']['Image']['Pixels']['Plane'][0::2]
        mdInfo = xmltodict.unparse(mdInfoDict).encode(encoding='UTF-8', errors='strict')

    # Get each frame and calculate the nn-output for it
    for frame in range(DrpOrig.shape[0]):
        printProgressBar(frame, DrpOrig.shape[0])
        inputData, positions = NNfeeder.prepareNNImages(MitoOrig[frame, :, :],
                                                        DrpOrig[frame, :, :], 128)
        outputPredict = model.predict_on_batch(inputData)
        if frame == 0:
            nnSize = int(np.sqrt(len(positions['px']))*(outputPredict.shape[1]
                                                        - positions['stitch']))
            nnImage = np.zeros((int(DrpOrig.shape[0]), nnSize, nnSize), dtype=np.uint8)
        nnImage[frame, :, :] = ImageTiles.stitchImage(outputPredict, positions)

    # Write the whole stack to a tif file with description
    tifffile.imwrite(nnPath, nnImage, description=mdInfo)


def main():
    """ Main method testing savegif """
    file = ('//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Dora/20180420_Dora_MitoGFP_Drp1mCh/'
            'sample1/sample1_cell_3/sample1_cell_3_MMStack_Pos0_1.ome.tif')
    calculateNNforStack(file)

    # target = 'C:/Users/stepp/Documents/05_Software/Analysis/test.ome.tiff'
    # extractTiffStack(file ,11 , target)

    # with tifffile.TiffFile(target, fastij=False) as tif:
    #     mdInfo = tif.ome_metadata
    #     mdInfoDict = xmltodict.parse(mdInfo)
    #     # ijInfoDict = json.loads(ijInfo)
    #     print('THIS IS WHAT WE GET OUT')
    #     print(mdInfoDict['OME']['Image']['Pixels']['Plane']['@DeltaT'])

    #     ijInfo = tif.shaped_metadata[0]
    #     ijInfoDict = json.loads(ijInfo['Info'])
    #     print(ijInfoDict['ElapsedTime-ms'])


    # from skimage import exposure
    # folder = (
    #     "C:/Users/stepp/Documents/02_Raw/SmartMito/microM_test/"
    #     "201208_cell_Int0s_30pc_488_50pc_561_band_5")

    # stack1 = loadTifFolder(folder, 512/741, 0)[0]
    # times = loadElapsedTime(folder)
    # times = times[0::2]
    # times = (times - np.min(times))/1000
    # times = np.diff(times).tolist()

    # print(len(times))
    # stack1 = exposure.rescale_intensity(
    #         stack1, (np.min(stack1), np.max(stack1)), out_range=(0, 255))
    # stack1 = stack1.astype('uint8')
    # savegif(stack1, times, 10)

    # print('Done')


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    main()
