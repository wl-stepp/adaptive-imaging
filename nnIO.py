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


def loadTifFolder(folder, order=0):
    stack1 = []
    stack2 = []
    stackNN = []
    for filePath in sorted(glob.glob(folder + '/img_*.tif')):
        splitStr = re.split(r'img_channel\d+_position\d+_time', os.path.basename(filePath))
        splitStr = re.split(r'_z\d+', splitStr[1])
        frameNum = int(splitStr[0])
        print(frameNum)
        if frameNum % 2:
            # odd
            stack1.append(io.imread(filePath))
            nnPath = filePath[:-8] + 'nn.tiff'
            try:
                stackNN.append(io.imread(nnPath))
            except FileNotFoundError:
                stackNN.append(np.zeros_like(stack1[0]))

        else:
            stack2.append(io.imread(filePath))
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
