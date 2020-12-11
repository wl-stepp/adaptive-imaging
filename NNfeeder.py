
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from skimage import exposure, filters, io, transform

from imageTiles import getTilePositions_v2


def prepareNNImages(mitoFull, drpFull, nnImageSize):

    # Set iSIM specific values
    pixelCalib = 56  # nm per pixel
    sig = 121.5/81  # in pixel
    resizeParam = pixelCalib/81  # no unit
    preSize = mitoFull.shape[0]  # pixel
    # Preprocess the images

    if drpFull.shape[1] > nnImageSize:
        # Adjust to 81nm/px
        t0 = time.perf_counter()
        postSize = round(preSize*resizeParam)
        # print(postSize)
        mitoFull = transform.rescale(mitoFull, resizeParam)
        drpFull = transform.rescale(drpFull, resizeParam)
        # This leaves an image that is smaller then initially
        # x0 = int(np.floor((mitoFull.shape[0] - 512)/2))
        # mitoFull = mitoFull[x0:x0 + 512,x0:x0 + 512]
        # drpFull = drpFull[x0:x0 + 512,x0:x0 + 512]
        t1 = time.perf_counter()

        # gaussian and background subtraction
        mitoFull = filters.gaussian(mitoFull, sig, preserve_range=True)
        drpFull = (filters.gaussian(drpFull, sig, preserve_range=True)
                   - filters.gaussian(drpFull, sig*5, preserve_range=True))
        t2 = time.perf_counter()


        # Contrast
        drpFull = exposure.rescale_intensity(
            drpFull, (np.min(drpFull), np.max(drpFull)), out_range=(0, 255))
        mitoFull = exposure.rescale_intensity(
            mitoFull, (np.mean(mitoFull), np.max(mitoFull)),
            out_range=(0, 255))
        # drpFull = drpFull.astype(np.uint8)
        # mitoFull = mitoFull.astype(np.uint8)

        t3 = time.perf_counter()

        # Tiling
        positions = getTilePositions_v2(drpFull, nnImageSize)
    else:
        positions = {'px': [(0, 0, drpFull.shape[1], drpFull.shape[1])],
                     'n': 1, 'overlap': 0, 'stitch': 0}
    t4 = time.perf_counter()

    # Put into format for the network
    drpFull = drpFull.reshape(1, drpFull.shape[0], drpFull.shape[0], 1)
    mitoFull = mitoFull.reshape(1, mitoFull.shape[0], mitoFull.shape[0], 1)
    inputDataFull = np.concatenate((mitoFull, drpFull), axis=3)
    t5 = time.perf_counter()

    # Cycle through these tiles
    i = 0
    inputData = np.zeros((positions['n']**2, nnImageSize, nnImageSize, 2))
    for position in positions['px']:
        inputData[i, :, :, :] = inputDataFull[:,
                                              position[0]:position[2],
                                              position[1]:position[3],
                                              :]
        # inputData[i, :, :, 1] =  exposure.rescale_intensity(
        #    inputData[i, :, :, 1], (0, np.max(inputData[i, :, :, 1])),
        #    out_range=(0, 255))
        inputData[i, :, :, 0] = exposure.rescale_intensity(
            inputData[i, :, :, 0], (0, np.max(inputData[i, :, :, 0])),
            out_range=(0, 255))

        i = i+1

    inputData = inputData.astype('uint8')
    t6 = time.perf_counter()
    # print('resize__', t1-t0)
    # print('gaussian', t2-t1)
    # print('contrast', t3-t2)
    # print('tile pre', t4-t3)
    # print('reformat', t5-t4)
    # print('tiling__', t6-t5)
    # print('full____', t6-t0)
    return inputData, positions
