'''
Input/Ouput module for data generated using the network_watchdog approach for adaptive temporal
sampling on the iSIM
'''

import glob
import numpy as np
import bioformats
import javabridge
import os
import tifffile
import json


def loadiSIMmetadata(folder):
    delay = []
    for name in glob.glob(folder + '/iSIMmetadata*.txt'):
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


if __name__ == '__main__':

    folder = (
        'C:/Users/stepp/Documents/data_raw/SmartMito/microM_test/cell_Int5s_30pc_488_50pc_561_5')
    print('Done')
